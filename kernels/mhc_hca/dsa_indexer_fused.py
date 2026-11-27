"""Fused Triton kernels for ERNIE 5.5 DSA Lightning Indexer & Sparse Index Prep on GB10.

1. `fused_merge_swa_topk_to_global`:
   Fuses `merge_swa_topk` + `triton_convert_req_index_to_global_index` into ONE
   Triton kernel, eliminating the intermediate `[T, 2176] int32` write+read (142 MB
   of DRAM traffic saved at T=8192) and removing 1 kernel launch per DSA layer.
2. `fused_hadamard_fp8_quant_q`:
   Fuses the 128-point orthonormal Walsh-Hadamard transform (`hadamard_rows(q)`) +
   `per_token_group_quant_fp8(q, 128, use_ue8m0=True)` + `weights * q_scale * weights_scale`
   into ONE single-pass register-resident Triton kernel for SM12.1's active FP8
   Lightning Indexer path! Eliminates 268 MB of DRAM roundtrips at T=8192 and fuses
   3 kernels into 1 at decode.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 1. Fused merge_swa_topk + req_index_to_global_index
# ---------------------------------------------------------------------------
@triton.jit
def _fused_merge_swa_to_global_kernel(
    topk_ptr,        # [T, TOPK] int32 (request-local token indices from indexer)
    positions_ptr,   # [T] int64/int32 (token positions -> seq_len = pos + 1)
    req_ids_ptr,     # [T] int32 (request id per token)
    block_table_ptr, # [num_reqs, max_blocks] int32
    out_ptr,         # [T, WIDTH] int32 (physical global token indices, -1 padded)
    topk_stride,
    bt_stride,
    out_stride,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE_ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1)
    cols = chunk_id * CHUNK + tl.arange(0, CHUNK)
    col_mask = cols < WIDTH

    seq_len = tl.load(positions_ptr + row).to(tl.int32) + 1
    req_id = tl.load(req_ids_ptr + row).to(tl.int64)
    indexer_len = tl.minimum(tl.maximum(seq_len - WINDOW, 0), TOPK)
    swa_len = tl.minimum(seq_len, WINDOW)

    is_topk = cols < indexer_len
    topk_val = tl.load(topk_ptr + row * topk_stride + cols, mask=is_topk, other=-1)
    swa_idx = seq_len - swa_len + cols - indexer_len
    local_idx = tl.where(
        is_topk,
        topk_val,
        tl.where(cols < indexer_len + swa_len, swa_idx, -1),
    )

    valid = (local_idx >= 0) & col_mask
    block_num = tl.where(valid, local_idx // BLOCK_SIZE, 0)
    block_off = tl.where(valid, local_idx % BLOCK_SIZE, 0)
    phys_block = tl.load(
        block_table_ptr + req_id * bt_stride + block_num,
        mask=valid,
        other=0,
    )
    global_idx = tl.where(valid, phys_block * BLOCK_STRIDE_ROWS + block_off, -1)
    tl.store(out_ptr + row * out_stride + cols, global_idx, mask=col_mask)


def fused_merge_swa_topk_to_global(
    topk_indices: torch.Tensor,
    positions: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    window: int = 128,
    block_size: int = 64,
    block_stride_rows: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-kernel merge of indexer top-k + sliding window + physical index conversion."""
    num_tokens, topk = topk_indices.shape
    width = topk + window
    if block_stride_rows is None:
        block_stride_rows = block_size
    if out is None:
        out = torch.empty((num_tokens, width), dtype=torch.int32, device=topk_indices.device)
    if num_tokens == 0:
        return out
    # For small T (decode), split WIDTH across 4 chunks of 512/1024 to increase SM parallelism;
    # for large T (prefill), 1 chunk per row (or 2) maximizes coalescing!
    chunk = 512 if num_tokens <= 32 else 1024
    num_chunks = triton.cdiv(width, chunk)
    _fused_merge_swa_to_global_kernel[(num_tokens, num_chunks)](
        topk_indices,
        positions,
        req_id_per_token,
        block_table,
        out,
        topk_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        WINDOW=window,
        TOPK=topk,
        WIDTH=width,
        BLOCK_SIZE=block_size,
        BLOCK_STRIDE_ROWS=block_stride_rows,
        CHUNK=chunk,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# 2. Fused 128-point Orthonormal Walsh-Hadamard + UE8M0 FP8 Quant + Weight Scale
#    Replaces: hadamard_rows(q) + per_token_group_quant_fp8(q) + weights * scale!
# ---------------------------------------------------------------------------
@triton.jit
def _fwht_128_step(x, stride: tl.constexpr, ROWS: tl.constexpr):
    """One butterfly stage of radix-2 FWHT on [ROWS, 128] using reshape+split+join."""
    # Reshape [ROWS, 128 // (2*stride), 2, stride] -> split dim 2
    n_groups: tl.constexpr = 128 // (2 * stride)
    v = tl.reshape(x, (ROWS, n_groups, 2, stride))
    v = tl.trans(v, (0, 1, 3, 2))  # [ROWS, n_groups, stride, 2]
    a, b = tl.split(v)             # each [ROWS, n_groups, stride]
    out = tl.join(a + b, a - b)    # [ROWS, n_groups, stride, 2]
    out = tl.trans(out, (0, 1, 3, 2))
    return tl.reshape(out, (ROWS, 128))


@triton.jit
def _fused_hadamard128_fp8_quant_kernel(
    q_ptr,          # [T, NH, 128] bf16
    w_in_ptr,       # [T, NH] bf16/fp32 (raw indexer head weights)
    q_out_ptr,      # [T, NH, 128] fp8_e4m3fn
    w_out_ptr,      # [T, NH] fp32 (scaled by q_scale * weights_scale)
    num_tokens,
    q_stride_tok,
    w_in_stride,
    w_out_stride,
    WEIGHTS_SCALE: tl.constexpr,
    INV_SQRT_128: tl.constexpr,
    FP8_MAX: tl.constexpr,
    NH: tl.constexpr,       # 64
    BLOCK_H: tl.constexpr,  # heads per CTA (e.g. 8 or 16)
):
    tok = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    h_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, 128)

    # Load [BLOCK_H, 128] bf16 -> fp32
    x = tl.load(
        q_ptr + tok * q_stride_tok + h_offs[:, None] * 128 + d_offs[None, :]
    ).to(tl.float32)

    # Exact 7-stage radix-2 orthonormal Walsh-Hadamard transform (D = 128 = 2^7)
    x = _fwht_128_step(x, 1, BLOCK_H)
    x = _fwht_128_step(x, 2, BLOCK_H)
    x = _fwht_128_step(x, 4, BLOCK_H)
    x = _fwht_128_step(x, 8, BLOCK_H)
    x = _fwht_128_step(x, 16, BLOCK_H)
    x = _fwht_128_step(x, 32, BLOCK_H)
    x = _fwht_128_step(x, 64, BLOCK_H)
    # Match unfused hadamard_rows bf16 boundary before UE8M0 quantization
    x = (x * INV_SQRT_128).to(tl.bfloat16).to(tl.float32)

    # Per-row (group=128) UE8M0 power-of-2 FP8 E4M3 quantization
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    raw_scale = amax * (1.0 / FP8_MAX)
    scale = tl.math.exp2(tl.ceil(tl.log2(raw_scale)))
    q_fp8 = tl.clamp(x / scale[:, None], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    tl.store(
        q_out_ptr + tok * (NH * 128) + h_offs[:, None] * 128 + d_offs[None, :],
        q_fp8,
    )
    w = tl.load(w_in_ptr + tok * w_in_stride + h_offs).to(tl.float32)
    tl.store(w_out_ptr + tok * w_out_stride + h_offs, w * scale * WEIGHTS_SCALE)


def fused_hadamard_fp8_quant_q(
    q: torch.Tensor,          # [T, 64, 128] bf16
    weights: torch.Tensor,    # [T, 64] bf16 or fp32
    weights_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 128-pt Hadamard + UE8M0 FP8 quant + weight scaling for SM12.1 indexer Q."""
    T, NH, D = q.shape
    assert D == 128 and NH == 64
    q_out = torch.empty((T, NH, D), dtype=torch.float8_e4m3fn, device=q.device)
    w_out = torch.empty((T, NH), dtype=torch.float32, device=q.device)
    if T == 0:
        return q_out, w_out
    block_h = 4 if T <= 8 else (8 if T <= 64 else 16)
    _fused_hadamard128_fp8_quant_kernel[(T, NH // block_h)](
        q,
        weights,
        q_out,
        w_out,
        T,
        q.stride(0),
        weights.stride(0),
        w_out.stride(0),
        WEIGHTS_SCALE=weights_scale,
        INV_SQRT_128=1.0 / math.sqrt(128.0),
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        NH=NH,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return q_out, w_out
