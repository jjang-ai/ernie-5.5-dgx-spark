#!/usr/bin/env python3
"""Single-layer MXFP4 MoE backend microbenchmark on NVIDIA GB10 (SM12.1).

Benchmarks all available vLLM MXFP4 MoE backends at exact ERNIE 5.5 Flash Preview
dimensions:
  - hidden_size (latent): 2048
  - moe_intermediate_size: 2048
  - n_routed_experts: 512, top_k: 10
  - activation: SiTU (beta=4.0, linear_beta=25.0)
  - EP=2 (E_local=256, N=2048) and EP=4 (E_local=128, N=2048)
  - M in [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]

Measures:
  - Cold-L2 median latency (us/layer)
  - Effective memory bandwidth (GB/s and % of GB10 273 GB/s peak)
  - Effective compute (TFLOPS)
  - Numerical parity (max_abs, rel_l2, cos_sim) vs DEEPGEMM_MXFP4 and FP32 ref
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
)

# ERNIE 5.5 Flash Preview exact constants
ERNIE_NUM_EXPERTS = 512
ERNIE_TOP_K = 10
ERNIE_LATENT_SIZE = 2048
ERNIE_MOE_INTERMEDIATE = 2048
ERNIE_SITU_BETA = 4.0
ERNIE_SITU_LINEAR_BETA = 25.0
ERNIE_NUM_MOE_LAYERS = 42
GB10_PEAK_BW_GBS = 273.0

# E2M1 FP4 lookup table (index 0..15)
FP4_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def dequantize_mxfp4_expert(
    w_packed: torch.Tensor, scale_e8m0: torch.Tensor
) -> torch.Tensor:
    """Dequantize one MXFP4 expert weight [N, K//2] uint8 + [N, K//32] uint8 -> [N, K] fp32."""
    lut = FP4_E2M1_LUT.to(w_packed.device)
    lo = (w_packed & 0x0F).to(torch.int64)
    hi = ((w_packed >> 4) & 0x0F).to(torch.int64)
    # Interleave lo (even k) and hi (odd k)
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(w_packed.shape[0], -1)
    # E8M0 scale = 2^(s - 127)
    scales = torch.pow(2.0, scale_e8m0.to(torch.float32) - 127.0)
    scales = scales.repeat_interleave(32, dim=-1)
    return vals * scales


def situ_and_mul_fp32(
    x: torch.Tensor, beta: float = ERNIE_SITU_BETA, linear_beta: float = ERNIE_SITU_LINEAR_BETA
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


@torch.no_grad()
def reference_moe_fp32(
    x: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Exact FP32 reference for active experts only."""
    M, H = x.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros((M, H), dtype=torch.float32, device=x.device)
    x_f32 = x.float()

    mapped_ids = expert_map[topk_ids] if expert_map is not None else topk_ids
    active_local = torch.unique(mapped_ids[mapped_ids >= 0]).tolist()
    for e_loc in active_local:
        mask = mapped_ids == e_loc  # [M, top_k]
        tok_idx, k_idx = torch.nonzero(mask, as_tuple=True)
        w13_f = dequantize_mxfp4_expert(w13_packed[e_loc], w13_scale[e_loc])  # [2N, H]
        w2_f = dequantize_mxfp4_expert(w2_packed[e_loc], w2_scale[e_loc])    # [H, N]
        h1 = x_f32[tok_idx] @ w13_f.t()
        act = situ_and_mul_fp32(h1)
        h2 = act @ w2_f.t()
        w = topk_weights[tok_idx, k_idx].float().unsqueeze(-1)
        out.index_add_(0, tok_idx, h2 * w)
    return out.to(x.dtype)


class DummyMoELayer(nn.Module):
    """Minimal RoutedExperts-compatible container for standalone kernel setup."""

    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        moe_config: FusedMoEConfig,
    ):
        super().__init__()
        self.w13_weight = nn.Parameter(w13.clone(), requires_grad=False)
        self.w2_weight = nn.Parameter(w2.clone(), requires_grad=False)
        self.w13_weight_scale = nn.Parameter(w13_scale.clone(), requires_grad=False)
        self.w2_weight_scale = nn.Parameter(w2_scale.clone(), requires_grad=False)
        self.moe_config = moe_config
        self.params_dtype = torch.bfloat16
        self.layer_name = "model.layers.1.mlp.experts"
        self.w13_bias = None
        self.w2_bias = None
        self.w13_input_scale = None
        self.w2_input_scale = None


def make_synthetic_weights(
    e_local: int, n_inter: int, hidden: int, device: torch.device, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(0, 256, (e_local, 2 * n_inter, hidden // 2), dtype=torch.uint8, device=device, generator=g)
    w2 = torch.randint(0, 256, (e_local, hidden, n_inter // 2), dtype=torch.uint8, device=device, generator=g)
    # Realistic UE8M0 exponents: 2^(118..122 - 127) = 2^-9 .. 2^-5 (~0.002 .. 0.03)
    # Keeps post-GEMM activations well within BF16 / FP8 dynamic range (std ~ 0.5-1.5)
    w13_scale = torch.randint(118, 123, (e_local, 2 * n_inter, hidden // 32), dtype=torch.uint8, device=device, generator=g)
    w2_scale = torch.randint(118, 123, (e_local, hidden, n_inter // 32), dtype=torch.uint8, device=device, generator=g)
    return w13, w2, w13_scale, w2_scale


def make_inputs(
    M: int,
    hidden: int,
    num_experts: int,
    top_k: int,
    e_local: int,
    device: torch.device,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    g = torch.Generator(device=device).manual_seed(seed + M)
    x = torch.randn((M, hidden), dtype=torch.bfloat16, device=device, generator=g) * 0.5
    # Generate unique top_k expert IDs per token
    scores = torch.rand((M, num_experts), dtype=torch.float32, device=device, generator=g)
    topk_vals, topk_ids = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)).to(torch.float32) * 3.0

    if e_local < num_experts:
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        expert_map[:e_local] = torch.arange(e_local, dtype=torch.int32, device=device)
        # For M=1, ensure exact expected number of local experts (e_local / num_experts * top_k)
        # so microbenchmarks don't suffer from random lucky/unlucky draw of 0 or 9 local hits!
        if M == 1:
            target_local = max(1, round(top_k * e_local / num_experts))
            local_ids = torch.randperm(e_local, device=device, generator=g)[:target_local]
            remote_ids = e_local + torch.randperm(num_experts - e_local, device=device, generator=g)[: top_k - target_local]
            topk_ids[0] = torch.cat([local_ids, remote_ids]).to(torch.int32)
    else:
        expert_map = None

    return x, topk_ids, topk_weights, expert_map


def compute_metrics(
    time_us: float,
    M: int,
    H: int,
    N: int,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> dict[str, float]:
    mapped = expert_map[topk_ids] if expert_map is not None else topk_ids
    valid_mask = mapped >= 0
    local_pairs = int(valid_mask.sum().item())
    active_experts = int(torch.unique(mapped[valid_mask]).numel()) if local_pairs > 0 else 0

    # Bytes per active expert: w13 (2*N*H/2 + 2*N*H/32) + w2 (H*N/2 + H*N/32)
    bytes_per_expert = (2 * N * H // 2) + (2 * N * H // 32) + (H * N // 2) + (H * N // 32)
    weight_bytes = active_experts * bytes_per_expert
    # Activation traffic (input read + output write, BF16)
    act_bytes = (M * H * 2) * 2
    total_bytes = weight_bytes + act_bytes

    # FLOPs: each (token, local_expert) pair does W13 (2 * 2N * H) + W2 (2 * H * N) = 6 * H * N
    flops = local_pairs * 6 * H * N

    sec = time_us * 1e-6
    bw_gbs = (total_bytes / sec) / 1e9 if sec > 0 else 0.0
    tflops = (flops / sec) / 1e12 if sec > 0 else 0.0
    return {
        "active_experts": active_experts,
        "local_pairs": local_pairs,
        "weight_mb": weight_bytes / 1e6,
        "bw_gbs": bw_gbs,
        "bw_pct": 100.0 * bw_gbs / GB10_PEAK_BW_GBS,
        "tflops": tflops,
    }


def parity_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a = out.float()
    b = ref.float()
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    rel_l2 = float(torch.linalg.norm(diff) / (torch.linalg.norm(b) + 1e-12))
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
    return {"max_abs": max_abs, "rel_l2": rel_l2, "cos": cos}


@dataclass
class BackendSpec:
    name: str
    backend: Mxfp4MoeBackend
    experts_cls_name: str | None
    activation_key: QuantKey | None = None
    humming_gemm_type: str | None = None
    humming_input_quant: str | None = None
    use_batched: bool = False
    activation_override: MoEActivation | None = None


def build_backend_runner(
    spec: BackendSpec,
    e_local: int,
    n_part: int,
    ep_size: int,
    tp_size: int,
    w13_orig: torch.Tensor,
    w2_orig: torch.Tensor,
    w13_scale_orig: torch.Tensor,
    w2_scale_orig: torch.Tensor,
    max_m: int,
) -> tuple[Any, DummyMoELayer, FusedMoEConfig]:
    # Set env vars for Humming variants if specified
    if spec.humming_gemm_type is not None:
        os.environ["VLLM_HUMMING_MOE_GEMM_TYPE"] = spec.humming_gemm_type
    else:
        os.environ.pop("VLLM_HUMMING_MOE_GEMM_TYPE", None)

    if spec.humming_input_quant is not None:
        os.environ["VLLM_HUMMING_INPUT_QUANT_CONFIG"] = spec.humming_input_quant
    else:
        os.environ.pop("VLLM_HUMMING_INPUT_QUANT_CONFIG", None)

    use_ep = ep_size > 1
    pconf = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=ep_size if use_ep else 1,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="naive",
        enable_eplb=False,
    )
    if spec.use_batched:
        # Force batched activation format flag if supported
        pconf.use_batched_activation_format = True

    activation = spec.activation_override or MoEActivation.SITU
    moe_cfg = FusedMoEConfig(
        num_experts=ERNIE_NUM_EXPERTS,
        experts_per_token=ERNIE_TOP_K,
        hidden_dim=ERNIE_LATENT_SIZE,
        intermediate_size=ERNIE_MOE_INTERMEDIATE,
        intermediate_size_per_partition=n_part,
        num_logical_experts=ERNIE_NUM_EXPERTS,
        num_local_experts=e_local,
        activation=activation,
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.Custom,
        moe_parallel_config=pconf,
        in_dtype=torch.bfloat16,
        max_num_tokens=max_m,
        activation_situ_beta=ERNIE_SITU_BETA,
        activation_situ_linear_beta=ERNIE_SITU_LINEAR_BETA,
    )

    layer = DummyMoELayer(w13_orig, w2_orig, w13_scale_orig, w2_scale_orig, moe_cfg)

    candidates = backend_to_kernel_cls(spec.backend)
    if spec.experts_cls_name is not None:
        matched = [c for c in candidates if c.__name__ == spec.experts_cls_name]
        if not matched:
            raise ValueError(f"Class {spec.experts_cls_name} not in {candidates}")
        experts_cls = matched[0]
    else:
        experts_cls = candidates[0]

    w13, w2, w13_s, w2_s, w13_b, w2_b = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=spec.backend,
        layer=layer,
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=layer.w13_weight_scale,
        w2_weight_scale=layer.w2_weight_scale,
        w13_bias=None,
        w2_bias=None,
        _cache_permute_indices={},
        activation=activation,
        activation_key=spec.activation_key,
    )
    layer.w13_weight = nn.Parameter(w13, requires_grad=False) if not isinstance(w13, nn.Parameter) else w13
    layer.w2_weight = nn.Parameter(w2, requires_grad=False) if not isinstance(w2, nn.Parameter) else w2
    if isinstance(w13_s, torch.Tensor):
        layer.w13_weight_scale = nn.Parameter(w13_s, requires_grad=False) if not isinstance(w13_s, nn.Parameter) else w13_s
        layer.w2_weight_scale = nn.Parameter(w2_s, requires_grad=False) if not isinstance(w2_s, nn.Parameter) else w2_s

    quant_cfg = make_mxfp4_moe_quant_config(
        mxfp4_backend=spec.backend,
        w1_scale=w13_s,
        w2_scale=w2_s,
        layer=layer,
    )

    if spec.use_batched:
        experts = experts_cls(
            moe_config=moe_cfg,
            quant_config=quant_cfg,
            max_num_tokens=max_m,
            num_dispatchers=1,
        )
        experts.process_weights_after_loading(layer)
        return experts, layer, moe_cfg

    experts = experts_cls(
        moe_config=moe_cfg,
        quant_config=quant_cfg,
    )
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
    kernel = mk.FusedMoEKernel(prepare_finalize, experts)
    kernel.fused_experts.process_weights_after_loading(layer)
    return kernel, layer, moe_cfg


def time_callable(
    fn,
    warmup: int = 8,
    iters: int = 25,
    flush_l2: bool = True,
    l2_buf: torch.Tensor | None = None,
) -> float:
    """Return median GPU execution time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if flush_l2 and l2_buf is not None:
            # Read-only L2 eviction avoids dirty writeback bus contention on LPDDR5X
            _ = l2_buf.sum()
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms * 1000.0  # microseconds


def get_backend_specs() -> list[BackendSpec]:
    return [
        BackendSpec(
            name="DEEPGEMM_MXFP4",
            backend=Mxfp4MoeBackend.DEEPGEMM_MXFP4,
            experts_cls_name="DeepGemmFP4Experts",
            activation_key=kFp8Dynamic128Sym,
        ),
        BackendSpec(
            name="MARLIN",
            backend=Mxfp4MoeBackend.MARLIN,
            experts_cls_name="MarlinExperts",
            activation_key=None,
        ),
        BackendSpec(
            name="HUMMING_GROUPED_BF16",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingGroupedExperts",
            activation_key=None,
            humming_gemm_type="grouped_contiguous",
        ),
        BackendSpec(
            name="HUMMING_INDEXED_BF16",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingIndexedExperts",
            activation_key=None,
            humming_gemm_type="indexed",
        ),
        BackendSpec(
            name="HUMMING_GROUPED_FP8TOK",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingGroupedExperts",
            activation_key=kFp8DynamicTokenSym,
            humming_gemm_type="grouped_contiguous",
        ),
        BackendSpec(
            name="HUMMING_INDEXED_FP8TOK",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingIndexedExperts",
            activation_key=kFp8DynamicTokenSym,
            humming_gemm_type="indexed",
        ),
        BackendSpec(
            name="HUMMING_GROUPED_MXFP8",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingGroupedExperts",
            activation_key=kMxfp8Dynamic,
            humming_gemm_type="grouped_contiguous",
        ),
        BackendSpec(
            name="HUMMING_INDEXED_MXFP8",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingIndexedExperts",
            activation_key=kMxfp8Dynamic,
            humming_gemm_type="indexed",
        ),
        BackendSpec(
            name="HUMMING_GROUPED_MXFP4",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingGroupedExperts",
            activation_key=kMxfp4Dynamic,
            humming_gemm_type="grouped_contiguous",
        ),
        BackendSpec(
            name="HUMMING_INDEXED_MXFP4",
            backend=Mxfp4MoeBackend.HUMMING,
            experts_cls_name="HummingIndexedExperts",
            activation_key=kMxfp4Dynamic,
            humming_gemm_type="indexed",
        ),
        BackendSpec(
            name="FLASHINFER_CUTLASS_MXFP8_SILU",
            backend=Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
            experts_cls_name="FlashInferExperts",
            activation_key=kMxfp8Dynamic,
            activation_override=MoEActivation.SILU,
        ),
        BackendSpec(
            name="FLASHINFER_CUTLASS_BF16_SILU",
            backend=Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
            experts_cls_name="FlashInferExperts",
            activation_key=None,
            activation_override=MoEActivation.SILU,
        ),
    ]


def run_config_sweep(
    label: str,
    e_local: int,
    n_part: int,
    ep_size: int,
    tp_size: int,
    m_list: list[int],
    specs: list[BackendSpec],
    flush_l2: bool,
    iters: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    print(f"\n{'='*90}")
    print(f"CONFIG: {label} | E_local={e_local}, N={n_part}, H={ERNIE_LATENT_SIZE}, top_k={ERNIE_TOP_K}")
    print(f"{'='*90}")

    w13_orig, w2_orig, w13_s_orig, w2_s_orig = make_synthetic_weights(
        e_local, n_part, ERNIE_LATENT_SIZE, device
    )
    # Pre-generate identical inputs for all M
    inputs_by_m = {
        M: make_inputs(M, ERNIE_LATENT_SIZE, ERNIE_NUM_EXPERTS, ERNIE_TOP_K, e_local, device)
        for M in m_list
    }
    # Compute exact FP32 reference for small/medium M for parity check
    fp32_refs: dict[int, torch.Tensor] = {}
    for M in [m for m in m_list if m <= 64]:
        x, tids, tw, emap = inputs_by_m[M]
        fp32_refs[M] = reference_moe_fp32(
            x, w13_orig, w13_s_orig, w2_orig, w2_s_orig, tids, tw, emap
        ).cpu()

    # 64 MB L2 flush buffer (GB10 L2 is 24 MB)
    l2_buf = torch.empty((64 * 1024 * 1024 // 4,), dtype=torch.float32, device=device) if flush_l2 else None

    dg_outputs: dict[int, torch.Tensor] = {}
    results: dict[str, Any] = {
        "label": label,
        "e_local": e_local,
        "n_part": n_part,
        "ep_size": ep_size,
        "tp_size": tp_size,
        "backends": {},
    }

    vllm_cfg = VllmConfig()
    with set_current_vllm_config(vllm_cfg), set_forward_context(None, vllm_cfg):
        for spec in specs:
            print(f"\n--- Backend: {spec.name} ---")
            try:
                kernel, layer, moe_cfg = build_backend_runner(
                    spec=spec,
                    e_local=e_local,
                    n_part=n_part,
                    ep_size=ep_size,
                    tp_size=tp_size,
                    w13_orig=w13_orig,
                    w2_orig=w2_orig,
                    w13_scale_orig=w13_s_orig,
                    w2_scale_orig=w2_s_orig,
                    max_m=max(m_list),
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"  SKIP/FAIL setup: {msg}")
                results["backends"][spec.name] = {"status": "setup_failed", "error": msg}
                gc.collect()
                torch.cuda.empty_cache()
                continue

            b_res: dict[str, Any] = {"status": "ok", "per_m": {}}
            for M in m_list:
                x, tids, tw, emap = inputs_by_m[M]
                act = spec.activation_override or MoEActivation.SITU

                def _run():
                    return kernel.apply(
                        hidden_states=x,
                        w1=layer.w13_weight,
                        w2=layer.w2_weight,
                        topk_weights=tw,
                        topk_ids=tids,
                        activation=act,
                        global_num_experts=ERNIE_NUM_EXPERTS,
                        expert_map=emap,
                        apply_router_weight_on_input=False,
                    )

                try:
                    # Single correctness run first
                    out = _run()
                    torch.cuda.synchronize()
                    if spec.name == "DEEPGEMM_MXFP4" and M <= 64:
                        dg_outputs[M] = out.detach().cpu()

                    # Adaptive iters for huge M to keep runtime brisk
                    cur_iters = iters if M <= 128 else max(10, iters // 2)
                    cur_warmup = 8 if M <= 128 else 4
                    time_us = time_callable(
                        _run,
                        warmup=cur_warmup,
                        iters=cur_iters,
                        flush_l2=flush_l2,
                        l2_buf=l2_buf,
                    )
                    m_info = compute_metrics(time_us, M, ERNIE_LATENT_SIZE, n_part, tids, emap)
                    entry: dict[str, Any] = {
                        "time_us": round(time_us, 2),
                        "ms_42layers": round(time_us * ERNIE_NUM_MOE_LAYERS / 1000.0, 3),
                        **{k: round(v, 3) if isinstance(v, float) else v for k, v in m_info.items()},
                    }
                    if M in fp32_refs and spec.activation_override is None:
                        entry["vs_fp32"] = {
                            k: round(v, 6) for k, v in parity_metrics(out.cpu(), fp32_refs[M]).items()
                        }
                    if M in dg_outputs and spec.activation_override is None:
                        entry["vs_dg"] = {
                            k: round(v, 6) for k, v in parity_metrics(out.cpu(), dg_outputs[M]).items()
                        }
                    b_res["per_m"][str(M)] = entry

                    par_str = ""
                    if "vs_dg" in entry:
                        par_str = (
                            f" | vs_DG rel_l2={entry['vs_dg']['rel_l2']:.4f} "
                            f"max={entry['vs_dg']['max_abs']:.4f} cos={entry['vs_dg']['cos']:.5f}"
                        )
                    elif "vs_fp32" in entry:
                        par_str = (
                            f" | vs_FP32 rel_l2={entry['vs_fp32']['rel_l2']:.4f} "
                            f"max={entry['vs_fp32']['max_abs']:.4f}"
                        )
                    print(
                        f"  M={M:4d} | {time_us:8.1f} us | {entry['ms_42layers']:6.2f} ms/42L | "
                        f"E_act={entry['active_experts']:3d} | {entry['bw_gbs']:6.1f} GB/s ({entry['bw_pct']:5.1f}%) | "
                        f"{entry['tflops']:5.2f} TFLOPS{par_str}"
                    )
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    print(f"  M={M:4d} | FAIL: {msg}")
                    b_res["per_m"][str(M)] = {"error": msg}
                    if M == m_list[0]:
                        # Likely unsupported at runtime; bail early
                        traceback.print_exc(limit=3)
                        break

            results["backends"][spec.name] = b_res
            del kernel, layer
            gc.collect()
            torch.cuda.empty_cache()

    del w13_orig, w2_orig, w13_s_orig, w2_s_orig, inputs_by_m, l2_buf
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="ep2,ep4", help="Comma-separated: ep2,ep4,tp2,tp4")
    parser.add_argument("--ms", default="1,2,4,8,16,32,64,128,512,2048")
    parser.add_argument("--backends", default="all", help="Comma-separated filter or 'all'")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--no-flush-l2", action="store_true")
    parser.add_argument("--include-flashinfer", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    init_workspace_manager(torch.device("cuda"))

    m_list = [int(x) for x in args.ms.split(",")]
    all_specs = get_backend_specs()
    if not args.include_flashinfer:
        all_specs = [s for s in all_specs if not s.name.startswith("FLASHINFER_")]
    if args.backends != "all":
        wanted = set(args.backends.split(","))
        all_specs = [s for s in all_specs if s.name in wanted]

    cfg_map = {
        "ep2": ("EP2 (E_local=256, N=2048)", 256, 2048, 2, 1),
        "ep4": ("EP4 (E_local=128, N=2048)", 128, 2048, 4, 1),
        "tp2": ("TP2 (E_local=512, N=1024)", 512, 1024, 1, 2),
        "tp4": ("TP4 (E_local=512, N=512)", 512, 512, 1, 4),
    }

    all_results = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "flush_l2": not args.no_flush_l2,
        "configs": {},
    }

    for key in args.configs.split(","):
        label, e_local, n_part, ep_size, tp_size = cfg_map[key.strip()]
        all_results["configs"][key.strip()] = run_config_sweep(
            label=label,
            e_local=e_local,
            n_part=n_part,
            ep_size=ep_size,
            tp_size=tp_size,
            m_list=m_list,
            specs=all_specs,
            flush_l2=not args.no_flush_l2,
            iters=args.iters,
        )

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved JSON results to {args.out_json}")


if __name__ == "__main__":
    main()
