# SPDX-License-Identifier: Apache-2.0
"""Single-kernel fused ERNIE 5.5 split-feature sigmoid + bias-corrected top-k router.

Upstream ``vllm_ernie/models/ernie5_5/nvidia/moe.py`` executes routing in two
separate Triton launches with an intermediate ``[T, 512]`` fp32 global buffer:
  1. ``split_sigmoid_sum(logits, E=512)`` -> writes ``[T, 512]`` fp32 scores
  2. ``ernie5_5_routing(..., scores, ...)`` -> reads ``[T, 512]`` fp32 scores,
     adds ``bias``, selects top-k=10, reorders to ``torch.topk(sorted=False)``
     slot order, renormalizes, multiplies by learnable ``scale[ids]``, and
     applies optional CUDA-graph padding mask.

This module fuses both stages into a single Triton kernel so the ``[T, 512]``
score tensor stays strictly in registers — eliminating 42 kernel launches and
42 global write+read roundtrips per decode step while maintaining bit-exact
numerical parity with upstream.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from vllm.triton_utils import tldevice
except ImportError:
    from triton.language.extra.cuda import libdevice as tldevice


@triton.jit
def _sigmoid(x):
    # Matches torch's fp32 CUDA sigmoid: 1 / (1 + expf(-x)), IEEE round-nearest.
    return tldevice.div_rn(1.0, 1.0 + tldevice.exp(-x))


@triton.jit
def _fused_ernie5_5_routing_kernel(
    logits_ptr,
    bias_ptr,
    scale_ptr,
    pad_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    kofs = tl.arange(0, TOPK_PAD)

    base = logits_ptr + row * stride_logits
    l0 = tl.load(base + offs).to(tl.float32)
    l1 = tl.load(base + E + offs).to(tl.float32)
    scores = _sigmoid(l0) + _sigmoid(l1)

    sel = scores + tl.load(bias_ptr + offs)
    ids = tl.full((TOPK_PAD,), E, dtype=tl.int32)
    vals = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    weights = tl.zeros((TOPK_PAD,), dtype=tl.float32)

    for k in tl.static_range(TOPK):
        best = tl.max(sel, 0)
        idx = tl.minimum(tl.min(tl.where(sel == best, offs, E), 0), E - 1)
        hit = offs == idx
        ids = tl.where(kofs == k, idx, ids)
        vals = tl.where(kofs == k, best, vals)
        weights = tl.where(kofs == k, tl.sum(tl.where(hit, scores, 0.0), 0), weights)
        sel = tl.where(hit, float("-inf"), sel)

    # Reorder to torch.topk(sorted=False) slots: experts above the k-th value
    # by index, then those equal to it by index.
    kmask = kofs < TOPK
    above = kmask & (vals > best)
    key = tl.where(above, ids, ids + E)
    slot = tl.sum((key[None, :] < key[:, None]).to(tl.int32), 1)
    move = slot[None, :] == kofs[:, None]
    ids = tl.sum(tl.where(move, ids[None, :], 0), 1)
    weights = tl.sum(tl.where(move, weights[None, :], 0.0), 1)

    if RENORM:
        total = tl.maximum(tl.sum(weights, 0), 1e-20)
        weights = tldevice.div_rn(weights, total)

    weights = weights * tl.load(scale_ptr + ids, mask=kmask, other=0.0)

    if HAS_PAD:
        is_pad = tl.load(pad_ptr + row) != 0
        weights = tl.where(is_pad, 0.0, weights)
        ids = tl.where(is_pad, -1, ids)

    tl.store(weights_ptr + row * TOPK + kofs, weights, mask=kmask)
    tl.store(ids_ptr + row * TOPK + kofs, ids, mask=kmask)


@triton.jit
def _fused_ernie5_5_routing_fast_kernel(
    logits_ptr,
    bias_ptr,
    scale_ptr,
    pad_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    """Variant using argmax(tie_break_left=True) to cut 30 reductions -> 20."""
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    kofs = tl.arange(0, TOPK_PAD)

    base = logits_ptr + row * stride_logits
    l0 = tl.load(base + offs).to(tl.float32)
    l1 = tl.load(base + E + offs).to(tl.float32)
    scores = _sigmoid(l0) + _sigmoid(l1)

    sel = scores + tl.load(bias_ptr + offs)
    ids = tl.full((TOPK_PAD,), E, dtype=tl.int32)
    vals = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    weights = tl.zeros((TOPK_PAD,), dtype=tl.float32)

    for k in tl.static_range(TOPK):
        best, idx = tl.max(sel, axis=0, return_indices=True, return_indices_tie_break_left=True)
        idx = idx.to(tl.int32)
        hit = offs == idx
        ids = tl.where(kofs == k, idx, ids)
        vals = tl.where(kofs == k, best, vals)
        weights = tl.where(kofs == k, tl.sum(tl.where(hit, scores, 0.0), 0), weights)
        sel = tl.where(hit, float("-inf"), sel)

    kmask = kofs < TOPK
    above = kmask & (vals > best)
    key = tl.where(above, ids, ids + E)
    slot = tl.sum((key[None, :] < key[:, None]).to(tl.int32), 1)
    move = slot[None, :] == kofs[:, None]
    ids = tl.sum(tl.where(move, ids[None, :], 0), 1)
    weights = tl.sum(tl.where(move, weights[None, :], 0.0), 1)

    if RENORM:
        total = tl.maximum(tl.sum(weights, 0), 1e-20)
        weights = tldevice.div_rn(weights, total)

    weights = weights * tl.load(scale_ptr + ids, mask=kmask, other=0.0)

    if HAS_PAD:
        is_pad = tl.load(pad_ptr + row) != 0
        weights = tl.where(is_pad, 0.0, weights)
        ids = tl.where(is_pad, -1, ids)

    tl.store(weights_ptr + row * TOPK + kofs, weights, mask=kmask)
    tl.store(ids_ptr + row * TOPK + kofs, ids, mask=kmask)


def fused_ernie5_5_routing(
    logits: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    bias: torch.Tensor,
    scale: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    use_fast_argmax: bool = True,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch fused split-feature sigmoid + ERNIE 5.5 top-k routing.

    Args:
        logits: ``[num_tokens, 2 * num_experts]`` float32 router GEMM output.
        topk: Number of experts per token (10 for ERNIE 5.5).
        renormalize: Whether to L1-renormalize selected sigmoid weights.
        bias: ``[num_experts]`` float32 ``e_score_correction_bias``.
        scale: ``[num_experts]`` float32 ``routed_scaling_factor_param``.
        is_padding: Optional ``[num_tokens]`` bool/uint8 padding mask.
    """
    num_tokens, two_e = logits.shape
    num_experts = two_e // 2
    assert two_e == 2 * num_experts

    if (
        logits.dtype != torch.float32
        or logits.stride(1) != 1
        or triton.next_power_of_2(num_experts) != num_experts
    ):
        scores = logits[:, :num_experts].float().sigmoid() + logits[:, num_experts:].float().sigmoid()
        ids = torch.topk(scores + bias, topk, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        if renormalize:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        weights = weights * scale[ids]
        weights, ids = weights.float(), ids.int()
        if is_padding is not None:
            m = is_padding.unsqueeze(1)
            weights = weights.masked_fill(m, 0)
            ids = ids.masked_fill(m, -1)
        return weights, ids

    weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=logits.device)
    if num_tokens:
        kernel = _fused_ernie5_5_routing_fast_kernel if use_fast_argmax else _fused_ernie5_5_routing_kernel
        kernel[(num_tokens,)](
            logits,
            bias,
            scale,
            is_padding if is_padding is not None else ids,
            weights,
            ids,
            logits.stride(0),
            E=num_experts,
            TOPK=topk,
            TOPK_PAD=triton.next_power_of_2(topk),
            RENORM=renormalize,
            HAS_PAD=is_padding is not None,
            enable_reflect_ftz=False,
            num_warps=num_warps,
        )
    return weights, ids
