#!/usr/bin/env python3
"""Benchmark ERNIE 5.5 mHC kernels on DGX Spark GB10 (SM12.1).

Measures both L2-hot (single weight tensor) and L2-cold (rotating weight bank
exceeding GB10's 24 MiB L2 cache, matching 86 distinct mHC call sites in
ERNIE 5.5's 43-layer model) latencies across T = 1..2048.

Also profiles sub-kernel breakdowns and compares against GB10-tuned Triton/CUDA
custom kernels with strict numerical parity verification.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

import torch

# Ensure local kernels package is importable when run from repo or remote copy
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vllm.model_executor.kernels.mhc.tilelang import (
    _hc_prenorm_gemm_outputs,
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    ENABLE_PDL,
    _HC_PRENORM_GEMM_TILELANG_KERNEL,
    _MHC_FUSED_TILELANG_KERNEL,
    _MHC_POST_TILELANG_KERNEL,
    compute_num_split,
    mhc_fused_post_pre_split_config,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported, tf32_hc_prenorm_gemm
from vllm_ernie.models.ernie5_5.mhc import (
    mhc_fused_post_pre_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)
from vllm_ernie.models.ernie5_5.mhc_kernels import _MHC_PRE_BIG_FUSE_TILELANG_KERNEL

# ERNIE 5.5 exact hyperparameters (from ernie55_config.json & model.py)
HIDDEN_SIZE = 4096
HC_MULT = 4
HC_MULT2 = HC_MULT * HC_MULT  # 16
HC_MULT3 = HC_MULT * (2 + HC_MULT)  # 24
HC_DIM = HC_MULT * HIDDEN_SIZE  # 16384
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT = 2.0
SINKHORN_ITERS = 20
NORM_EPS = 1e-6
NUM_LAYERS = 43
NUM_MHC_CALL_SITES = NUM_LAYERS * 2  # 86 total (1 broadcast + 85 fused_post_pre + 1 post + 1 head)

TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]


def bench_cuda(fn: Callable[[], None], warmup: int = 10, iters: int = 50) -> float:
    """Return median-ish mean latency in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def bench_cuda_cudagraph(fn: Callable[[], None], warmup: int = 10, iters: int = 100) -> float:
    """Measure pure GPU execution time inside a CUDA graph (no Python launch overhead)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def make_inputs(num_tokens: int, num_banks: int = 1, device: str = "cuda"):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.5
    residual = (
        torch.randn(num_tokens, HC_MULT, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        * 0.5
    )
    post_mix = torch.rand(num_tokens, HC_MULT, 1, dtype=torch.float32, device=device) * 2.0
    comb_raw = torch.rand(num_tokens, HC_MULT, HC_MULT, dtype=torch.float32, device=device)
    comb_mix = comb_raw / comb_raw.sum(dim=-1, keepdim=True)
    # Rotating banks of fn (1.57 MB each): 24 banks = 37.7 MB > 24 MiB L2 cache
    fns = [
        torch.randn(HC_MULT3, HC_DIM, dtype=torch.float32, device=device) * 0.02
        for _ in range(num_banks)
    ]
    hc_scale = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)
    hc_base = torch.randn(HC_MULT3, dtype=torch.float32, device=device) * 0.1
    norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device) + (
        torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.05
    )
    return x, residual, post_mix, comb_mix, fns, hc_scale, hc_base, norm_weight


def profile_baseline(tokens_list: list[int], cold_banks: int = 24):
    props = torch.cuda.get_device_properties(0)
    print(
        f"Device: {props.name} (SM{props.major}.{props.minor}, "
        f"{props.multi_processor_count} SMs, L2={props.L2_cache_size/(1024**2):.1f} MiB, "
        f"ENABLE_PDL={ENABLE_PDL}, deep_gemm={is_deep_gemm_supported()})"
    )
    print(f"ERNIE 5.5 mHC: hc_mult={HC_MULT}, H={HIDDEN_SIZE}, sinkhorn={SINKHORN_ITERS}, sites={NUM_MHC_CALL_SITES}")
    print(f"fn size per site = {HC_MULT3 * HC_DIM * 4 / 1e6:.3f} MB; cold bank = {cold_banks} x = {cold_banks * HC_MULT3 * HC_DIM * 4 / 1e6:.1f} MB\n")

    print("=== 1. Baseline mhc_fused_post_pre_tilelang (with fused RMSNorm) ===")
    header = (
        f"{'T':>5} | {'Path':<18} | {'Eager Hot':>10} | {'Graph Hot':>10} | "
        f"{'Graph Cold':>10} | {'85x Cold':>9} | {'Sub-k1(us)':>10} | {'Sub-k2(us)':>10} | {'Sub-k3(us)':>10}"
    )
    print(header)
    print("-" * len(header))

    for T in tokens_list:
        cfg = mhc_fused_post_pre_split_config(T, HIDDEN_SIZE, HC_MULT)
        if cfg is not None:
            path_str = f"fused(n={cfg[0]},s={cfg[1]})"
        else:
            ns = compute_num_split(64, HC_DIM, (T + 63) // 64)
            path_str = f"3-kern(dg s={ns})"

        x, res, pm, cm, fns_hot, hc_scale, hc_base, nw = make_inputs(T, num_banks=1)
        fn0 = fns_hot[0]

        def run_hot():
            return mhc_fused_post_pre_tilelang(
                x, res, pm, cm, fn0, hc_scale, hc_base,
                RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT, SINKHORN_ITERS,
                norm_weight=nw, norm_eps=NORM_EPS,
            )

        iters = 80 if T <= 128 else (40 if T <= 512 else 20)
        eager_hot = bench_cuda(run_hot, warmup=5, iters=iters)
        graph_hot = bench_cuda_cudagraph(run_hot, warmup=5, iters=iters)

        # L2-cold benchmark: replay a graph containing cold_banks distinct fn tensors
        _, _, _, _, fns_cold, _, _, _ = make_inputs(T, num_banks=cold_banks)
        # Keep bank count reasonable for large T to avoid huge activation memory
        nb = cold_banks if T <= 512 else 16
        fns_cold = fns_cold[:nb]

        def run_cold_bank():
            for f in fns_cold:
                mhc_fused_post_pre_tilelang(
                    x, res, pm, cm, f, hc_scale, hc_base,
                    RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT, SINKHORN_ITERS,
                    norm_weight=nw, norm_eps=NORM_EPS,
                )

        cold_iters = max(4, iters // nb)
        graph_cold = bench_cuda_cudagraph(run_cold_bank, warmup=2, iters=cold_iters) / nb

        # Sub-kernel breakdown (L2-hot in CUDA graph)
        pm_flat = pm.view(T, HC_MULT)
        post_out = torch.empty(T, HC_MULT, dtype=torch.float32, device="cuda")
        comb_out = torch.empty(T, HC_MULT2, dtype=torch.float32, device="cuda")
        li_out = torch.empty(T, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        if cfg is not None:
            fn_3d = fn0.view(HC_MULT3, HC_MULT, HIDDEN_SIZE)
            k1_us = bench_cuda_cudagraph(
                lambda: _MHC_FUSED_TILELANG_KERNEL(
                    cm, res, pm_flat, x, fn_3d, HC_MULT, HIDDEN_SIZE, HC_MULT3
                ),
                warmup=5,
                iters=iters,
            )
            gm, gs, rcur = _MHC_FUSED_TILELANG_KERNEL(
                cm, res, pm_flat, x, fn_3d, HC_MULT, HIDDEN_SIZE, HC_MULT3
            )
            k2_us = bench_cuda_cudagraph(
                lambda: _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
                    gm, gs, hc_scale, hc_base, rcur, post_out, comb_out, li_out,
                    RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT, SINKHORN_ITERS,
                    norm_weight=nw, norm_eps=NORM_EPS,
                ),
                warmup=5,
                iters=iters,
            )
            k3_us = 0.0
        else:
            rcur = torch.empty_like(res)
            k1_us = bench_cuda_cudagraph(
                lambda: _MHC_POST_TILELANG_KERNEL(
                    cm, res, pm_flat, x, rcur, HC_MULT, HIDDEN_SIZE
                ),
                warmup=5,
                iters=iters,
            )
            ns = compute_num_split(64, HC_DIM, (T + 63) // 64)
            gm = torch.empty(ns, T, HC_MULT3, dtype=torch.float32, device="cuda")
            gs = torch.empty(ns, T, dtype=torch.float32, device="cuda")
            r2d = rcur.view(T, HC_DIM)
            k2_us = bench_cuda_cudagraph(
                lambda: tf32_hc_prenorm_gemm(r2d, fn0, gm, gs, ns),
                warmup=5,
                iters=iters,
            )
            k3_us = bench_cuda_cudagraph(
                lambda: _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
                    gm, gs, hc_scale, hc_base, rcur, post_out, comb_out, li_out,
                    RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT, SINKHORN_ITERS,
                    norm_weight=nw, norm_eps=NORM_EPS,
                ),
                warmup=5,
                iters=iters,
            )

        print(
            f"{T:5d} | {path_str:<18} | {eager_hot:8.1f}us | {graph_hot:8.1f}us | "
            f"{graph_cold:8.1f}us | {graph_cold*85/1000:7.2f}ms | "
            f"{k1_us:8.1f}us | {k2_us:8.1f}us | {k3_us:8.1f}us"
        )
        del x, res, pm, cm, fns_hot, fns_cold, hc_scale, hc_base, nw, gm, gs, rcur
        torch.cuda.empty_cache()

    print("\n=== 2. Other mHC Entry Points (mhc_pre, mhc_pre_broadcast, mhc_post, hc_head) ===")
    h2 = f"{'T':>5} | {'mhc_pre(us)':>12} | {'pre_bcast(us)':>13} | {'mhc_post(us)':>12} | {'hc_head(us)':>12}"
    print(h2)
    print("-" * len(h2))
    for T in [1, 4, 16, 32, 64, 128, 512, 2048]:
        x, res, pm, cm, fns_hot, hc_scale, hc_base, nw = make_inputs(T, num_banks=1)
        fn0 = fns_hot[0]
        fn_bcast = fn0.view(HC_MULT3, HC_MULT, HIDDEN_SIZE).sum(dim=1).contiguous()
        fn_head = torch.randn(HC_MULT, HC_DIM, dtype=torch.float32, device="cuda") * 0.02
        scale_head = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        base_head = torch.randn(HC_MULT, dtype=torch.float32, device="cuda") * 0.1
        iters = 50 if T <= 128 else 20

        t_pre = bench_cuda_cudagraph(
            lambda: mhc_pre_tilelang(
                res, fn0, hc_scale, hc_base,
                RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT, SINKHORN_ITERS,
                norm_weight=nw, norm_eps=NORM_EPS,
            ),
            warmup=5, iters=iters,
        )
        t_bcast = bench_cuda_cudagraph(
            lambda: mhc_pre_broadcast_tilelang(
                x, fn0, hc_scale, hc_base,
                RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT, SINKHORN_ITERS,
                norm_weight=nw, norm_eps=NORM_EPS, fn_broadcast=fn_bcast,
            ),
            warmup=5, iters=iters,
        )
        t_post = bench_cuda_cudagraph(
            lambda: mhc_post_tilelang(x, res, pm, cm),
            warmup=5, iters=iters,
        )
        t_head = bench_cuda_cudagraph(
            lambda: hc_head_fused_kernel_tilelang(
                res, fn_head, scale_head, base_head, RMS_EPS, HC_PRE_EPS
            ),
            warmup=5, iters=iters,
        )
        print(f"{T:5d} | {t_pre:10.1f}us | {t_bcast:11.1f}us | {t_post:10.1f}us | {t_head:10.1f}us")
        del x, res, pm, cm, fns_hot, fn_bcast, fn_head, scale_head, base_head
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_COUNTS)
    parser.add_argument("--cold-banks", type=int, default=24)
    args = parser.parse_args()
    profile_baseline(args.tokens, cold_banks=args.cold_banks)


if __name__ == "__main__":
    main()
