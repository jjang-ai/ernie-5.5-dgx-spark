#!/usr/bin/env python3
"""Benchmark ERNIE 5.5 router fusion & FP8 block-128 linear backends on GB10 SM12.1.

Covers:
  1. Fused ERNIE 5.5 router (single Triton kernel) vs upstream 2-kernel
     (split_sigmoid_sum + ernie5_5_routing) across M = 1..8192 + bit-exact parity.
  2. Block-128 FP8 linear backends (DeepGEMM, Humming, Marlin, Triton, Cutlass,
     FlashInfer) on ERNIE 5.5 shared expert + latent projections across M = 1..2048.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.linear import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    CutlassFp8BlockScaledMMKernel,
    DeepGemmFp8BlockScaledMMKernel,
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
    FP8ScaledMMLinearLayerConfig,
    HummingFP8ScaledMMLinearKernel,
    MarlinFP8ScaledMMLinearKernel,
    TritonFp8BlockScaledMMKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.v1.worker.workspace import init_workspace_manager

# Upstream ERNIE 5.5 router functions
from vllm_ernie.models.ernie5_5.nvidia.moe import (
    ernie5_5_routing,
    split_sigmoid_sum,
)

GB10_PEAK_BW_GBS = 273.0
ERNIE_MOE_LAYERS = 42


def time_callable(fn, warmup=15, iters=50, flush_l2=False, l2_buf=None) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush_l2 and l2_buf is not None:
            _ = l2_buf.sum()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2] * 1000.0  # us


def bench_router(fused_mod, m_list: list[int]) -> dict:
    device = torch.device("cuda")
    E = 512
    TOPK = 10
    bias = torch.randn(E, dtype=torch.float32, device=device) * 0.1
    scale = torch.rand(E, dtype=torch.float32, device=device) * 2.0 + 1.0

    print("\n" + "=" * 90)
    print("BENCHMARK 1: ERNIE 5.5 FUSED ROUTER (split_sigmoid_sum + ernie5_5_routing)")
    print("=" * 90)

    res = {}
    for M in m_list:
        g = torch.Generator(device=device).manual_seed(1000 + M)
        logits = torch.randn((M, 2 * E), dtype=torch.float32, device=device, generator=g)
        dummy_h = torch.empty((M, 1), dtype=torch.bfloat16, device=device)

        # Upstream reference
        def run_upstream():
            scores = split_sigmoid_sum(logits, E)
            return ernie5_5_routing(dummy_h, scores, TOPK, True, bias=bias, scale=scale)

        def run_fused_exact():
            return fused_mod.fused_ernie5_5_routing(
                logits, TOPK, True, bias=bias, scale=scale, use_fast_argmax=False
            )

        def run_fused_fast():
            return fused_mod.fused_ernie5_5_routing(
                logits, TOPK, True, bias=bias, scale=scale, use_fast_argmax=True
            )

        # Parity check (including ties and padding!)
        ref_w, ref_ids = run_upstream()
        ex_w, ex_ids = run_fused_exact()
        fa_w, fa_ids = run_fused_fast()

        ex_id_eq = bool(torch.equal(ref_ids, ex_ids))
        ex_w_max = float((ref_w - ex_w).abs().max().item())
        fa_id_eq = bool(torch.equal(ref_ids, fa_ids))
        fa_w_max = float((ref_w - fa_w).abs().max().item())

        # Also verify tie-breaking explicitly
        if M == 16:
            tie_logits = torch.zeros((M, 2 * E), dtype=torch.float32, device=device)
            tie_bias = torch.zeros(E, dtype=torch.float32, device=device)
            tw_ref, ti_ref = ernie5_5_routing(
                dummy_h, split_sigmoid_sum(tie_logits, E), TOPK, True, bias=tie_bias, scale=scale
            )
            tw_fa, ti_fa = fused_mod.fused_ernie5_5_routing(
                tie_logits, TOPK, True, bias=tie_bias, scale=scale, use_fast_argmax=True
            )
            assert torch.equal(ti_ref, ti_fa), f"Tie-break mismatch! {ti_ref[0]} vs {ti_fa[0]}"
            assert float((tw_ref - tw_fa).abs().max().item()) == 0.0

        t_up = time_callable(run_upstream, warmup=20, iters=100)
        t_ex = time_callable(run_fused_exact, warmup=20, iters=100)
        t_fa = time_callable(run_fused_fast, warmup=20, iters=100)

        saved_42l_ms = (t_up - min(t_ex, t_fa)) * ERNIE_MOE_LAYERS / 1000.0
        print(
            f"  M={M:4d} | upstream={t_up:6.2f} us | fused_exact={t_ex:6.2f} us ({t_up/t_ex:.2f}x, id_eq={ex_id_eq}, max={ex_w_max:.1e}) | "
            f"fused_fast={t_fa:6.2f} us ({t_up/t_fa:.2f}x, id_eq={fa_id_eq}, max={fa_w_max:.1e}) | save/42L={saved_42l_ms:.3f} ms"
        )
        res[str(M)] = {
            "upstream_us": round(t_up, 2),
            "fused_exact_us": round(t_ex, 2),
            "fused_fast_us": round(t_fa, 2),
            "speedup_fast": round(t_up / t_fa, 3),
            "saved_42l_ms": round(saved_42l_ms, 3),
            "exact_id_eq": ex_id_eq,
            "exact_max_abs": ex_w_max,
            "fast_id_eq": fa_id_eq,
            "fast_max_abs": fa_w_max,
        }
    return res


class DummyFp8LinearLayer(nn.Module):
    def __init__(self, n_out: int, k_in: int, w_fp8: torch.Tensor, s_inv: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(w_fp8.clone(), requires_grad=False)
        self.weight_scale_inv = nn.Parameter(s_inv.clone(), requires_grad=False)
        self.weight_scale = self.weight_scale_inv
        self.input_scale = None
        self.weight_block_size = [128, 128]
        self.input_size_per_partition = k_in
        self.output_size_per_partition = n_out
        self.output_partition_sizes = [n_out]
        self.logical_widths = [n_out]
        self.orig_dtype = torch.bfloat16
        self.params_dtype = torch.bfloat16
        self.has_bias = False
        self.bias = None
        self.layer_name = "model.layers.1.mlp.shared_experts.gate_up_proj"


def bench_fp8_linears(m_list: list[int], flush_l2: bool) -> dict:
    device = torch.device("cuda")
    print("\n" + "=" * 90)
    print("BENCHMARK 2: ERNIE 5.5 BLOCK-128 FP8 LINEARS (shared_expert & latent_proj)")
    print("=" * 90)

    shapes = [
        ("shared_gate_up_TP1 (4096->4096)", 4096, 4096),
        ("shared_gate_up_TP2 (4096->2048)", 2048, 4096),
        ("shared_down_TP1    (2048->4096)", 4096, 2048),
        ("shared_down_TP2    (1024->4096)", 4096, 1024),
        ("fc1_latent         (4096->2048)", 2048, 4096),
    ]

    candidate_kernels = [
        ("DeepGemm", DeepGemmFp8BlockScaledMMKernel),
        ("FlashInferDeepGEMM", FlashInferFp8DeepGEMMDynamicBlockScaledKernel),
        ("CutlassBlock", CutlassFp8BlockScaledMMKernel),
        ("MarlinFP8", MarlinFP8ScaledMMLinearKernel),
        ("HummingFP8", HummingFP8ScaledMMLinearKernel),
        ("TritonBlock", TritonFp8BlockScaledMMKernel),
    ]

    l2_buf = torch.empty((64 * 1024 * 1024 // 4,), dtype=torch.float32, device=device) if flush_l2 else None
    from types import SimpleNamespace
    vllm_cfg = VllmConfig()
    hf_cfg = SimpleNamespace(model_type="ernie5_5_moe")
    vllm_cfg.model_config = SimpleNamespace(
        hf_config=hf_cfg,
        hf_text_config=hf_cfg,
        dtype=torch.bfloat16,
        quantization="fp8",
    )
    results = {}

    with set_current_vllm_config(vllm_cfg), set_forward_context(None, vllm_cfg):
        for s_name, N, K in shapes:
            print(f"\n--- Shape: {s_name} [N={N}, K={K}] ---")
            g = torch.Generator(device=device).manual_seed(777)
            w_f32 = torch.randn((N, K), dtype=torch.float32, device=device, generator=g) * 0.05
            # Block-128 quantization with UE8M0 power-of-2 scales (matching ERNIE 5.5 config)
            w_blk = w_f32.view(N // 128, 128, K // 128, 128)
            amax = w_blk.abs().amax(dim=(1, 3)).clamp(min=1e-6)
            raw_scale = amax / 448.0
            # Round scale up to power of 2 (ue8m0)
            s_inv = torch.pow(2.0, torch.ceil(torch.log2(raw_scale))).to(torch.float32)
            s_exp = s_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
            w_fp8 = (w_f32 / s_exp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            w_deq_bf16 = (w_fp8.float() * s_exp).to(torch.bfloat16)

            xs = {
                M: torch.randn((M, K), dtype=torch.bfloat16, device=device, generator=g) * 0.5
                for M in m_list
            }
            refs = {M: (xs[M].float() @ w_deq_bf16.float().t()).to(torch.bfloat16) for M in m_list}

            # Check AUTO selection
            auto_k = init_fp8_linear_kernel(
                activation_quant_key=kFp8Dynamic128Sym,
                weight_quant_key=kFp8Static128BlockSym,
                weight_shape=(N, K),
                input_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                module_name="Fp8LinearMethod",
            )
            print(f"  vLLM AUTO kernel: {type(auto_k).__name__}")

            s_res = {"N": N, "K": K, "auto": type(auto_k).__name__, "kernels": {}}
            for k_name, k_cls in candidate_kernels:
                cfg = FP8ScaledMMLinearLayerConfig(
                    weight_quant_key=kFp8Static128BlockSym,
                    activation_quant_key=kFp8Dynamic128Sym,
                    weight_shape=(N, K),
                    input_dtype=torch.bfloat16,
                    out_dtype=torch.bfloat16,
                )

                ok, reason = k_cls.can_implement(cfg)
                if not ok:
                    print(f"  {k_name:20s} UNSUPPORTED: {reason}")
                    s_res["kernels"][k_name] = {"status": "unsupported", "reason": str(reason)}
                    continue

                try:
                    layer = DummyFp8LinearLayer(N, K, w_fp8, s_inv)
                    try:
                        k_inst = k_cls(
                            cfg,
                            layer_param_names=["weight", "weight_scale_inv", "input_scale", "input_scale_ub"],
                        )
                    except TypeError:
                        k_inst = k_cls(cfg)
                    k_inst.process_weights_after_loading(layer)

                    k_per_m = {}
                    for M in m_list:
                        x = xs[M]
                        out = k_inst.apply_weights(layer, x, None)
                        torch.cuda.synchronize()
                        diff = (out.float() - refs[M].float()).abs()
                        rel_l2 = float(torch.linalg.norm(diff) / (torch.linalg.norm(refs[M].float()) + 1e-12))

                        t_us = time_callable(
                            lambda: k_inst.apply_weights(layer, x, None),
                            warmup=10,
                            iters=40 if M <= 128 else 20,
                            flush_l2=flush_l2,
                            l2_buf=l2_buf,
                        )
                        # Weight bytes (N*K*1B) + scales + act
                        w_bytes = N * K + (N // 128) * (K // 128) * 4
                        tot_bytes = w_bytes + (M * K + M * N) * 2
                        bw = (tot_bytes / (t_us * 1e-6)) / 1e9
                        k_per_m[str(M)] = {
                            "time_us": round(t_us, 2),
                            "bw_gbs": round(bw, 1),
                            "bw_pct": round(100.0 * bw / GB10_PEAK_BW_GBS, 1),
                            "rel_l2": round(rel_l2, 5),
                        }
                    summary = " | ".join(
                        f"M={M}:{k_per_m[str(M)]['time_us']:.1f}us({k_per_m[str(M)]['bw_gbs']:.0f}GB/s)"
                        for M in m_list if str(M) in k_per_m
                    )
                    print(f"  {k_name:20s} {summary} (rel_l2@M1={k_per_m[str(m_list[0])]['rel_l2']:.4f})")
                    s_res["kernels"][k_name] = {"status": "ok", "per_m": k_per_m}
                    del layer, k_inst
                except Exception as e:
                    print(f"  {k_name:20s} FAIL: {type(e).__name__}: {e}")
                    s_res["kernels"][k_name] = {"status": "fail", "error": f"{type(e).__name__}: {e}"}
                gc.collect()
                torch.cuda.empty_cache()

            # Also benchmark BF16 cuBLAS baseline (dequantized weight) for comparison!
            bf16_per_m = {}
            for M in m_list:
                x = xs[M]
                t_us = time_callable(
                    lambda: torch.nn.functional.linear(x, w_deq_bf16),
                    warmup=10,
                    iters=40 if M <= 128 else 20,
                    flush_l2=flush_l2,
                    l2_buf=l2_buf,
                )
                w_bytes = N * K * 2
                bw = ((w_bytes + (M * K + M * N) * 2) / (t_us * 1e-6)) / 1e9
                bf16_per_m[str(M)] = {"time_us": round(t_us, 2), "bw_gbs": round(bw, 1)}
            summary = " | ".join(f"M={M}:{bf16_per_m[str(M)]['time_us']:.1f}us" for M in m_list)
            print(f"  {'BF16_cuBLAS_ref':20s} {summary}")
            s_res["kernels"]["BF16_cuBLAS_ref"] = {"status": "ok", "per_m": bf16_per_m}
            results[s_name] = s_res

    del l2_buf
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-ms", default="1,2,4,8,16,32,64,128,512,2048,8192")
    parser.add_argument("--linear-ms", default="1,2,4,8,16,32,64,128,512")
    parser.add_argument("--fused-router-dir", default="/tmp")
    parser.add_argument("--out-json", default="/tmp/fp8_linear_and_router.json")
    args = parser.parse_args()

    sys.path.insert(0, args.fused_router_dir)
    import fused_ernie55_router

    init_workspace_manager(torch.device("cuda"))
    out = {
        "router": bench_router(fused_ernie55_router, [int(x) for x in args.router_ms.split(",")]),
        "fp8_linear": bench_fp8_linears([int(x) for x in args.linear_ms.split(",")], flush_l2=True),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
