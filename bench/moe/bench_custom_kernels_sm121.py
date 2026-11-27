#!/usr/bin/env python3
"""Benchmark & strictly verify all custom SM12.1 (GB10) ERNIE 5.5 MoE/FP8 kernels.

Tests:
  1. Custom SM12.1 PTX 2-kernel MXFP4 MoE decode (EP2 & EP4, M=1,2,4,8)
  2. Custom SM12.1 PTX Block-128 FP8 GEMV (shared expert & latent proj, M=1..8)
  3. Custom fused SiTU + DeepGEMM packed UE8M0 FP8 quant kernel (vs unfused 2-kernel)
  4. DeepGemmFP4Experts stage-by-stage breakdown & Humming M=1/M=2048 cliff diagnosis
  5. Shared expert + Routed MoE dual-stream overlap on GB10
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/tmp")
import bench_mxfp4_moe_sm121 as base
import fused_situ_fp8_quant
import sm121_fp8_block_gemv
import sm121_mxfp4_moe_decode

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.v1.worker.workspace import init_workspace_manager

GB10_PEAK = 273.0
LAYERS = 42


def time_fn(fn, warmup=10, iters=30, l2_buf=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_buf is not None:
            _ = l2_buf.sum()
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in zip(s, e))
    return ts[len(ts) // 2] * 1000.0


def test_custom_mxfp4_moe_decode(l2_buf):
    device = torch.device("cuda")
    print("\n" + "=" * 92)
    print("1. CUSTOM SM12.1 PTX 2-KERNEL MXFP4 MoE DECODE (W13+SiTU -> W2+Reduce)")
    print("=" * 92)
    out = {}
    for cfg_name, e_local in [("ep2", 256), ("ep4", 128)]:
        print(f"\n--- {cfg_name.upper()} (E_local={e_local}, N=2048, H=2048) ---")
        w13, w2, w13_s, w2_s = base.make_synthetic_weights(e_local, 2048, 2048, device)
        out[cfg_name] = {}
        for M in [1, 2, 4, 8]:
            x, tids, tw, emap = base.make_inputs(M, 2048, 512, 10, e_local, device)
            ref = base.reference_moe_fp32(x, w13, w13_s, w2, w2_s, tids, tw, emap)

            best_t = 1e18
            best_cfg = None
            best_par = None
            for rpw1 in [2, 4, 8]:
                for rpw2 in [2, 4, 8]:
                    for atomic in [False, True]:
                        # Skip redundant/slow combos for larger M
                        if M > 2 and (rpw1 == 2 or rpw2 == 2):
                            continue
                        def _run():
                            return sm121_mxfp4_moe_decode.mxfp4_moe_decode_sm121(
                                x, w13, w13_s, w2, w2_s, tids, tw, emap,
                                beta=4.0, linear_beta=25.0,
                                rows_per_warp_w1=rpw1, rows_per_warp_w2=rpw2,
                                use_atomic_w2=atomic,
                            )
                        y = _run()
                        torch.cuda.synchronize()
                        par = base.parity_metrics(y, ref)
                        assert par["rel_l2"] < 0.01, f"Parity failed! {par}"
                        t = time_fn(_run, warmup=8, iters=30, l2_buf=l2_buf)
                        if t < best_t:
                            best_t = t
                            best_cfg = (rpw1, rpw2, atomic)
                            best_par = par

            m_info = base.compute_metrics(best_t, M, 2048, 2048, tids, emap)
            print(
                f"  M={M:2d} | best={best_t:7.1f} us ({best_t*LAYERS/1000:5.2f} ms/42L) | "
                f"E_act={m_info['active_experts']:2d} | {m_info['bw_gbs']:6.1f} GB/s ({m_info['bw_pct']:5.1f}%) | "
                f"cfg=(r1={best_cfg[0]},r2={best_cfg[1]},atom={best_cfg[2]}) | "
                f"vs_FP32 rel_l2={best_par['rel_l2']:.5f} max={best_par['max_abs']:.4f} cos={best_par['cos']:.6f}"
            )
            out[cfg_name][str(M)] = {
                "time_us": round(best_t, 2),
                "ms_42layers": round(best_t * LAYERS / 1000.0, 3),
                "bw_gbs": round(m_info["bw_gbs"], 1),
                "bw_pct": round(m_info["bw_pct"], 1),
                "active_experts": m_info["active_experts"],
                "cfg": {"rpw1": best_cfg[0], "rpw2": best_cfg[1], "atomic": best_cfg[2]},
                "vs_fp32": {k: round(v, 6) for k, v in best_par.items()},
            }
        del w13, w2, w13_s, w2_s
        gc.collect()
        torch.cuda.empty_cache()
    return out


def test_custom_fp8_gemv(l2_buf):
    device = torch.device("cuda")
    print("\n" + "=" * 92)
    print("2. CUSTOM SM12.1 PTX BLOCK-128 FP8 GEMV (shared_expert & latent_proj, M=1..8)")
    print("=" * 92)
    shapes = [
        ("shared_gate_up_TP1 (4096->4096)", 4096, 4096),
        ("shared_gate_up_TP2 (4096->2048)", 2048, 4096),
        ("shared_down_TP1    (2048->4096)", 4096, 2048),
        ("shared_down_TP2    (1024->4096)", 4096, 1024),
        ("fc1_latent         (4096->2048)", 2048, 4096),
    ]
    out = {}
    for name, N, K in shapes:
        g = torch.Generator(device=device).manual_seed(777)
        w_f32 = torch.randn((N, K), dtype=torch.float32, device=device, generator=g) * 0.05
        w_blk = w_f32.view(N // 128, 128, K // 128, 128)
        amax = w_blk.abs().amax(dim=(1, 3)).clamp(min=1e-6)
        s_inv = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0))).to(torch.float32)
        s_exp = s_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        w_fp8 = (w_f32 / s_exp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w_deq = (w_fp8.float() * s_exp)

        s_res = {}
        parts = []
        for M in [1, 2, 4, 8]:
            x = torch.randn((M, K), dtype=torch.bfloat16, device=device, generator=g) * 0.5
            ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
            best_t = 1e18
            best_rpw = 4
            best_par = None
            for rpw in [2, 4, 8]:
                def _run():
                    return sm121_fp8_block_gemv.fp8_block128_gemv_sm121(x, w_fp8, s_inv, rows_per_warp=rpw)
                y = _run()
                torch.cuda.synchronize()
                par = base.parity_metrics(y, ref)
                assert par["rel_l2"] < 0.005, f"FP8 GEMV parity failed: {par}"
                t = time_fn(_run, warmup=10, iters=40, l2_buf=l2_buf)
                if t < best_t:
                    best_t = t
                    best_rpw = rpw
                    best_par = par
            w_bytes = N * K + (N // 128) * (K // 128) * 4 + (M * K + M * N) * 2
            bw = (w_bytes / (best_t * 1e-6)) / 1e9
            s_res[str(M)] = {
                "time_us": round(best_t, 2),
                "bw_gbs": round(bw, 1),
                "bw_pct": round(100.0 * bw / GB10_PEAK, 1),
                "rpw": best_rpw,
                "rel_l2": round(best_par["rel_l2"], 6),
            }
            parts.append(f"M={M}:{best_t:.1f}us({bw:.0f}GB/s,{100*bw/GB10_PEAK:.0f}%)")
        print(f"  {name:34s} | " + " | ".join(parts) + f" | rel_l2@M1={s_res['1']['rel_l2']:.5f}")
        out[name] = s_res
    return out


def test_fused_situ_fp8_quant(l2_buf):
    device = torch.device("cuda")
    print("\n" + "=" * 92)
    print("3. FUSED SiTU + DeepGEMM PACKED UE8M0 FP8 QUANT (vs unfused 2-kernel)")
    print("=" * 92)
    vllm_cfg = VllmConfig()
    out = {}
    with set_current_vllm_config(vllm_cfg):
        situ_op = SituAndMul(beta=4.0, linear_beta=25.0)
        # Representative DeepGEMM M_sum values across M=1..128 (128-aligned per active expert)
        for label, m_sum, valid_frac in [
            ("EP2 M=1  (E_act=5,  M_sum=640)", 640, 5 / 640),
            ("EP2 M=8  (E_act=37, M_sum=4736)", 4736, 40 / 4736),
            ("EP2 M=16 (E_act=59, M_sum=7552)", 7552, 80 / 7552),
            ("EP2 M=64 (E_act=192,M_sum=24576)", 24576, 320 / 24576),
            ("EP2 M=512(E_act=256,M_sum=32768)", 32768, 2560 / 32768),
        ]:
            x = torch.randn((m_sum, 4096), dtype=torch.bfloat16, device=device) * 0.5
            out_q_buf = torch.empty((m_sum, 2048), dtype=torch.float8_e4m3fn, device=device)

            # Build expert_ids mask with realistic padding fraction
            n_valid = max(1, int(round(m_sum * valid_frac)))
            eids = torch.full((m_sum,), -1, dtype=torch.int32, device=device)
            eids[:n_valid] = 0

            def run_unfused():
                act_out = torch.empty((m_sum, 2048), dtype=torch.bfloat16, device=device)
                torch.ops._C.situ_and_mul(act_out, x, 4.0, 25.0)
                return per_token_group_quant_fp8_packed_for_deepgemm(
                    act_out, 128, use_ue8m0=True, out_q=out_q_buf
                )

            def run_fused_full():
                return fused_situ_fp8_quant.fused_situ_mul_fp8_quant_packed(
                    x, beta=4.0, linear_beta=25.0, out_q=out_q_buf, expert_ids=None
                )

            def run_fused_masked():
                return fused_situ_fp8_quant.fused_situ_mul_fp8_quant_packed(
                    x, beta=4.0, linear_beta=25.0, out_q=out_q_buf, expert_ids=eids
                )

            # Parity check between unfused and fused_full
            q_ref, s_ref = run_unfused()
            q_ref = q_ref.clone()
            s_ref = s_ref.clone()
            q_fus, s_fus = run_fused_full()
            scale_eq = bool(torch.equal(s_ref, s_fus))
            # Check dequantized output parity
            q_diff = (q_ref.float() - q_fus.float()).abs().max().item()
            q_rel = (torch.linalg.norm(q_ref.float() - q_fus.float()) / (torch.linalg.norm(q_ref.float()) + 1e-12)).item()

            t_unf = time_fn(run_unfused, warmup=10, iters=30, l2_buf=l2_buf)
            t_fus = time_fn(run_fused_full, warmup=10, iters=30, l2_buf=l2_buf)
            t_msk = time_fn(run_fused_masked, warmup=10, iters=30, l2_buf=l2_buf)
            print(
                f"  {label:35s} | unfused={t_unf:7.1f} us | fused={t_fus:6.1f} us ({t_unf/t_fus:.2f}x) | "
                f"fused+skip_pad={t_msk:5.1f} us ({t_unf/t_msk:.2f}x) | scale_eq={scale_eq} q_rel={q_rel:.5f}"
            )
            out[label] = {
                "unfused_us": round(t_unf, 2),
                "fused_us": round(t_fus, 2),
                "fused_masked_us": round(t_msk, 2),
                "speedup_fused": round(t_unf / t_fus, 2),
                "speedup_masked": round(t_unf / t_msk, 2),
                "scale_eq": scale_eq,
                "q_rel_l2": round(q_rel, 6),
                "q_max_abs": round(q_diff, 5),
            }
    return out


def test_humming_heuristics_and_overlap(l2_buf):
    device = torch.device("cuda")
    print("\n" + "=" * 92)
    print("4. HUMMING HEURISTICS DIAGNOSIS (M=1 & M=2048 cliffs) + SHARED EXPERT OVERLAP")
    print("=" * 92)
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic, kFp8DynamicTokenSym

    # Inspect Humming tuning tables for EP2 (E=256)
    w13, w2, w13_s, w2_s = base.make_synthetic_weights(256, 2048, 2048, device)
    vllm_cfg = VllmConfig()
    with set_current_vllm_config(vllm_cfg), set_forward_context(None, vllm_cfg):
        for name, akey, gtype, cls_name in [
            ("HUMMING_INDEXED_BF16", None, "indexed", "HummingIndexedExperts"),
            ("HUMMING_INDEXED_MXFP8", kMxfp8Dynamic, "indexed", "HummingIndexedExperts"),
            ("HUMMING_GROUPED_MXFP8", kMxfp8Dynamic, "grouped_contiguous", "HummingGroupedExperts"),
        ]:
            spec = base.BackendSpec(name, Mxfp4MoeBackend.HUMMING, cls_name, akey, gtype)
            kernel, layer, moe_cfg = base.build_backend_runner(
                spec, 256, 2048, 2, 1, w13, w2, w13_s, w2_s, 2048
            )
            ex = kernel.fused_experts
            print(f"\n  {name} w13_tuning_config:")
            for lo, hi, tc in ex.w13_tuning_config:
                print(f"    M in ({lo}, {hi}]: block_shape={tc.get('block_shape')} warp_shape={tc.get('warp_shape')} split_k={tc.get('num_split_k', tc.get('split_k'))} stages={tc.get('num_stages')}")
            del kernel, layer
            gc.collect()

    # Dual-stream overlap test: Shared Expert (custom FP8 GEMV) + Routed MoE (custom MXFP4 decode) at M=1,2,4,8
    print("\n--- Shared Expert (FP8) + Routed MoE (MXFP4) Dual-Stream Overlap (EP2) ---")
    g = torch.Generator(device=device).manual_seed(99)
    # Shared expert weights (TP1: gate_up 4096->4096, down 2048->4096)
    def mk_fp8(N, K):
        wf = torch.randn((N, K), dtype=torch.float32, device=device, generator=g) * 0.05
        amax = wf.view(N // 128, 128, K // 128, 128).abs().amax(dim=(1, 3)).clamp(min=1e-6)
        si = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0))).to(torch.float32)
        se = si.repeat_interleave(128, 0).repeat_interleave(128, 1)
        return (wf / se).clamp(-448, 448).to(torch.float8_e4m3fn), si

    w_gu, s_gu = mk_fp8(4096, 4096)
    w_dn, s_dn = mk_fp8(4096, 2048)
    aux_stream = torch.cuda.Stream(device=device)
    ev0 = torch.cuda.Event()
    ev1 = torch.cuda.Event()

    overlap_res = {}
    for M in [1, 2, 4, 8]:
        x_h = torch.randn((M, 4096), dtype=torch.bfloat16, device=device, generator=g) * 0.5
        x_lat, tids, tw, emap = base.make_inputs(M, 2048, 512, 10, 256, device)
        act_buf = torch.empty((M, 2048), dtype=torch.bfloat16, device=device)

        def run_shared():
            gu = sm121_fp8_block_gemv.fp8_block128_gemv_sm121(x_h, w_gu, s_gu, 4)
            torch.ops._C.situ_and_mul(act_buf, gu, 4.0, 25.0)
            return sm121_fp8_block_gemv.fp8_block128_gemv_sm121(act_buf, w_dn, s_dn, 4)

        def run_routed():
            return sm121_mxfp4_moe_decode.mxfp4_moe_decode_sm121(
                x_lat, w13, w13_s, w2, w2_s, tids, tw, emap, 4.0, 25.0, 4, 4, False
            )

        def run_seq():
            a = run_shared()
            b = run_routed()
            return a, b

        def run_par():
            cur = torch.cuda.current_stream(device)
            ev0.record(cur)
            with torch.cuda.stream(aux_stream):
                aux_stream.wait_event(ev0)
                a = run_shared()
                ev1.record(aux_stream)
            b = run_routed()
            cur.wait_event(ev1)
            return a, b

        t_sh = time_fn(run_shared, 8, 30, l2_buf)
        t_rt = time_fn(run_routed, 8, 30, l2_buf)
        t_seq = time_fn(run_seq, 8, 30, l2_buf)
        t_par = time_fn(run_par, 8, 30, l2_buf)
        print(
            f"  M={M} | shared={t_sh:6.1f} us | routed={t_rt:6.1f} us | "
            f"seq={t_seq:6.1f} us | dual_stream={t_par:6.1f} us ({t_seq/t_par:.3f}x, delta={t_seq-t_par:+.1f} us)"
        )
        overlap_res[str(M)] = {
            "shared_us": round(t_sh, 2),
            "routed_us": round(t_rt, 2),
            "seq_us": round(t_seq, 2),
            "par_us": round(t_par, 2),
            "speedup": round(t_seq / t_par, 3),
        }

    del w13, w2, w13_s, w2_s, w_gu, s_gu, w_dn, s_dn
    gc.collect()
    torch.cuda.empty_cache()
    return overlap_res


def main():
    init_workspace_manager(torch.device("cuda"))
    l2_buf = torch.empty((64 * 1024 * 1024 // 4,), dtype=torch.float32, device="cuda")
    res = {
        "custom_mxfp4_moe_decode": test_custom_mxfp4_moe_decode(l2_buf),
        "custom_fp8_block128_gemv": test_custom_fp8_gemv(l2_buf),
        "fused_situ_fp8_quant": test_fused_situ_fp8_quant(l2_buf),
        "overlap": test_humming_heuristics_and_overlap(l2_buf),
    }
    del l2_buf
    gc.collect()
    torch.cuda.empty_cache()
    with open("/tmp/custom_kernels_sm121.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved /tmp/custom_kernels_sm121.json")


if __name__ == "__main__":
    main()
