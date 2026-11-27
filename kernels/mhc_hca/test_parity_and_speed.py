#!/usr/bin/env python3
"""Parity & speedup verification suite for all GB10 (SM12.1) mHC + HCA/DSA kernels."""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import flashinfer.mla as fi_mla
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm_ernie.models.ernie5_5.mhc import mhc_fused_post_pre_tilelang
from vllm_ernie.models.ernie5_5.nvidia.attention import (
    hadamard_rows,
    merge_swa_topk,
)

from kernels.mhc_hca.dsa_indexer_fused import (
    fused_hadamard_fp8_quant_q,
    fused_merge_swa_topk_to_global,
)
from kernels.mhc_hca.mhc_sm121 import mhc_fused_post_pre_sm121
from kernels.mhc_hca.sm120_attn_dispatch import (
    dsa_sparse_mla_sm121,
    hca_sparse_mla_sm121,
)


def bench_cg(fn, warmup=3, iters=40) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def bench_cuda(fn, warmup=3, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float(), b.float()
    return (af - bf).norm().item() / (bf.norm().item() + 1e-12)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def test_mhc():
    print("=== 1. mHC Fused Post+Pre+RMSNorm Parity & Speed (L2-cold rotating bank) ===")
    H = 4096
    print(f"  {'T':>5} | {'Mode':<11} | {'Base(us)':>9} | {'SM121(us)':>10} | {'Speedup':>7} | {'85x Saved':>10} | {'max_abs (res/post/comb/li)':<34} | {'rel_l2 (li)':>10}")
    print("  " + "-" * 116)
    for T in [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]:
        torch.manual_seed(42)
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5
        res = torch.randn(T, 4, H, dtype=torch.bfloat16, device="cuda") * 0.5
        pm = torch.rand(T, 4, 1, dtype=torch.float32, device="cuda") * 2.0
        cm = torch.rand(T, 4, 4, dtype=torch.float32, device="cuda")
        cm = cm / cm.sum(dim=-1, keepdim=True)
        nb = 24 if T <= 128 else (16 if T <= 512 else 8)
        fns = [torch.randn(24, 4 * H, dtype=torch.float32, device="cuda") * 0.02 for _ in range(nb)]
        hc_scale = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device="cuda")
        hc_base = torch.randn(24, dtype=torch.float32, device="cuda") * 0.1
        nw = torch.ones(H, dtype=torch.bfloat16, device="cuda") + torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.05

        ref = mhc_fused_post_pre_tilelang(
            x, res, pm, cm, fns[0], hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, norm_weight=nw, norm_eps=1e-6
        )
        cold_iters = 6 if T <= 128 else 4
        t_base = bench_cg(
            lambda: [
                mhc_fused_post_pre_tilelang(
                    x, res, pm, cm, f, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, norm_weight=nw, norm_eps=1e-6
                )
                for f in fns
            ],
            warmup=2,
            iters=cold_iters,
        ) / nb

        modes = ["auto"]
        for m in modes:
            out = mhc_fused_post_pre_sm121(
                x, res, pm, cm, fns[0], hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                norm_weight=nw, norm_eps=1e-6, mode=m,
            )
            t_opt = bench_cg(
                lambda mm=m: [
                    mhc_fused_post_pre_sm121(
                        x, res, pm, cm, f, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                        norm_weight=nw, norm_eps=1e-6, mode=mm,
                    )
                    for f in fns
                ],
                warmup=2,
                iters=cold_iters,
            ) / nb
            ma = [max_abs(a, b) for a, b in zip(out, ref)]
            rl_li = rel_l2(out[3], ref[3])
            saved_ms = (t_base - t_opt) * 85 / 1000.0
            ma_s = f"{ma[0]:.1e}/{ma[1]:.1e}/{ma[2]:.1e}/{ma[3]:.1e}"
            print(f"  {T:5d} | {m:<11} | {t_base:7.1f}us | {t_opt:8.1f}us | {t_base/t_opt:6.2f}x | {saved_ms:+8.2f}ms | {ma_s:<34} | {rl_li:10.2e}")
            assert rl_li < 1e-2, f"mHC parity failed at T={T}, mode={m}: rel_l2={rl_li}"
        del x, res, pm, cm, fns, ref
        torch.cuda.empty_cache()
    print()


def test_hca_dsa_dispatch():
    print("=== 2. HCA (37L) & DSA (6L) SM120 Calibrated Dispatch Parity & Speed ===")
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sinks = torch.randn(64, dtype=torch.float32, device="cuda") * 0.5

    # HCA: finite non-trivial byte values across DSv4's 584B page-footer layout (448B FP8 + 128B BF16 + 8B scales)
    swa_cache = torch.randint(48, 60, (1024, 64, 1, 584), dtype=torch.uint8, device="cuda")
    comp_cache = torch.randint(48, 60, (1024, 64, 1, 584), dtype=torch.uint8, device="cuda")
    scale_hca = 512 ** -0.5

    print("  [2a. HCA dsv4 (37 layers)]")
    print(f"  {'B':>4} | {'kv_len':>6} | {'Base(us)':>9} | {'SM121(us)':>10} | {'Speedup':>7} | {'37L Saved':>10} | {'max_abs_err':>11}")
    print("  " + "-" * 72)
    for B, kv_len in [(1, 512), (1, 2048), (1, 8192), (1, 16384), (1, 32768), (4, 8192), (4, 32768), (8, 8192), (16, 8192), (64, 32768)]:
        comp_k = max(1, kv_len // 128)
        comp_w = ((comp_k + 63) // 64) * 64
        q = torch.randn(B, 64, 512, dtype=torch.bfloat16, device="cuda") * 0.1
        swa_idx = torch.randint(0, 1024 * 64, (B, 1, 128), dtype=torch.int32, device="cuda")
        swa_lens = torch.full((B,), min(kv_len, 128), dtype=torch.int32, device="cuda")
        comp_idx = torch.randint(0, 1024 * 64, (B, 1, comp_w), dtype=torch.int32, device="cuda")
        if comp_k < comp_w:
            comp_idx[:, :, comp_k:] = -1
        comp_lens = torch.full((B,), comp_k, dtype=torch.int32, device="cuda")
        out_ref = torch.empty_like(q)
        out_opt = torch.empty_like(q)

        def run_ref():
            fi_mla.trtllm_batch_decode_sparse_mla_dsv4(
                query=q, swa_kv_cache=swa_cache, workspace_buffer=workspace,
                sparse_indices=swa_idx, compressed_kv_cache=comp_cache,
                out=out_ref, bmm1_scale=scale_hca, sinks=sinks, kv_layout="NHD",
                swa_topk_lens=swa_lens, extra_sparse_indices=comp_idx,
                extra_sparse_topk_lens=comp_lens,
            )

        def run_opt():
            hca_sparse_mla_sm121(
                q, swa_cache, swa_idx, swa_lens, comp_cache, comp_idx, comp_lens, sinks, scale_hca, out=out_opt
            )

        run_ref()
        run_opt()
        torch.cuda.synchronize()
        err = max_abs(out_opt, out_ref)
        t_ref = bench_cg(run_ref, warmup=2, iters=30)
        t_opt = bench_cg(run_opt, warmup=2, iters=30)
        saved = (t_ref - t_opt) * 37 / 1000.0
        print(f"  {B:4d} | {kv_len:6d} | {t_ref:7.1f}us | {t_opt:8.1f}us | {t_ref/t_opt:6.2f}x | {saved:+8.2f}ms | {err:11.2e}")
        assert err < 1e-2, f"HCA parity failed: {err}"
    del swa_cache, comp_cache

    # DSA: realistic FP8 E4M3 KV cache values
    num_pages = 2048
    kv_cache = torch.zeros((num_pages, 1, 64, 656), dtype=torch.uint8, device="cuda")
    kv_cache[..., :512] = (
        (torch.randn(num_pages, 1, 64, 512, dtype=torch.bfloat16, device="cuda") * 0.25)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    kv_cache[..., 512:528].view(torch.float32).fill_(1.0)
    kv_cache[..., 528:] = (
        (torch.randn(num_pages, 1, 64, 64, dtype=torch.bfloat16, device="cuda") * 0.25)
        .view(torch.uint8)
    )
    scale_dsa = 256 ** -0.5

    print("\n  [2b. DSA dsv3_2 (6 layers, topk=2176, sinks)]")
    print(f"  {'T':>5} | {'Base(us)':>9} | {'SM121(us)':>10} | {'Speedup':>7} | {'6L Saved':>10} | {'max_abs_err':>11}")
    print("  " + "-" * 65)
    for T in [1, 2, 4, 8, 16, 32, 64, 512]:
        q = torch.randn(T, 64, 576, dtype=torch.bfloat16, device="cuda") * 0.1
        idx = torch.randint(0, num_pages * 64, (T, 1, 2176), dtype=torch.int32, device="cuda")
        out_ref = torch.empty(T, 1, 64, 512, dtype=torch.bfloat16, device="cuda")
        out_opt = torch.empty(T, 64, 512, dtype=torch.bfloat16, device="cuda")

        def run_ref():
            fi_mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1), kv_cache=kv_cache, workspace_buffer=workspace,
                qk_nope_head_dim=192, kv_lora_rank=512, qk_rope_head_dim=64,
                block_tables=idx, seq_lens=None, max_seq_len=2176,
                out=out_ref, bmm1_scale=scale_dsa, bmm2_scale=1.0,
                sparse_mla_top_k=2176, kv_scale_format="pow2_fp32", sinks=sinks,
            )

        def run_opt():
            dsa_sparse_mla_sm121(q, kv_cache, idx, sinks, scale_dsa, out=out_opt)

        run_ref()
        run_opt()
        torch.cuda.synchronize()
        err = max_abs(out_opt, out_ref.squeeze(1))
        fn_b = bench_cg if T <= 64 else bench_cuda
        iters = 25 if T <= 16 else 12
        t_ref = fn_b(run_ref, warmup=2, iters=iters)
        t_opt = fn_b(run_opt, warmup=2, iters=iters)
        saved = (t_ref - t_opt) * 6 / 1000.0
        print(f"  {T:5d} | {t_ref:7.1f}us | {t_opt:8.1f}us | {t_ref/t_opt:6.2f}x | {saved:+8.2f}ms | {err:11.2e}")
        assert err < 1e-2, f"DSA parity failed: {err}"
    del kv_cache, workspace
    torch.cuda.empty_cache()
    print()


def test_dsa_indexer_fused():
    print("=== 3. Fused DSA Indexer Triton Kernels Parity & Speed (6 layers) ===")
    print("  [3a. fused_merge_swa_topk_to_global vs merge_swa_topk + req2glob]")
    print(f"  {'T':>5} | {'Base(us)':>9} | {'Fused(us)':>10} | {'Speedup':>7} | {'6L Saved':>9} | {'ExactMatch':>10}")
    print("  " + "-" * 64)
    bt = torch.arange(512, dtype=torch.int32, device="cuda").unsqueeze(0)
    for T in [1, 8, 16, 64, 512, 2048, 8192]:
        topk_in = torch.randint(0, 4096, (T, 2048), dtype=torch.int32, device="cuda")
        pos = torch.full((T,), 8191, dtype=torch.int64, device="cuda")
        req_ids = torch.zeros(T, dtype=torch.int32, device="cuda")
        swa_buf = torch.empty((T, 2176), dtype=torch.int32, device="cuda")

        def run_ref():
            merge_swa_topk(topk_in, swa_buf, pos, 128)
            return triton_convert_req_index_to_global_index(
                req_ids, bt, swa_buf, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2176
            )

        out_ref = run_ref()
        out_fused = fused_merge_swa_topk_to_global(topk_in, pos, req_ids, bt, window=128, block_size=64)
        exact = torch.equal(out_ref, out_fused)
        assert exact, f"merge+req2glob mismatch at T={T}"

        fn_b = bench_cg if T <= 64 else bench_cuda
        iters = 40 if T <= 64 else 15
        t_ref = fn_b(run_ref, warmup=3, iters=iters)
        t_opt = fn_b(
            lambda: fused_merge_swa_topk_to_global(topk_in, pos, req_ids, bt, window=128, block_size=64, out=swa_buf),
            warmup=3,
            iters=iters,
        )
        print(f"  {T:5d} | {t_ref:7.1f}us | {t_opt:8.1f}us | {t_ref/t_opt:6.2f}x | {(t_ref-t_opt)*6/1000:+7.2f}ms | {str(exact):>10}")
        del topk_in, pos, req_ids, swa_buf, out_ref, out_fused

    print("\n  [3b. fused_hadamard_fp8_quant_q vs hadamard_rows + per_token_group_quant_fp8 + scale]")
    print(f"  {'T':>5} | {'Base(us)':>9} | {'Fused(us)':>10} | {'Speedup':>7} | {'6L Saved':>9} | {'q_rel_l2':>9} | {'w_rel_l2':>9}")
    print("  " + "-" * 74)
    w_scale = (128 ** -0.5) * (64 ** -0.5)
    for T in [1, 8, 16, 64, 512, 2048, 8192]:
        q = torch.randn(T, 64, 128, dtype=torch.bfloat16, device="cuda") * 0.5
        w = torch.randn(T, 64, dtype=torch.bfloat16, device="cuda")

        def run_ref():
            qh = hadamard_rows(q.view(-1, 128))
            qq, qs = per_token_group_quant_fp8(qh, 128, column_major_scales=False, use_ue8m0=True)
            qq = qq.view(T, 64, 128)
            ww = w * qs.view(T, 64) * w_scale
            return qq, ww

        qq_ref, ww_ref = run_ref()
        qq_opt, ww_opt = fused_hadamard_fp8_quant_q(q, w, w_scale)
        q_err = rel_l2(qq_opt, qq_ref)
        w_err = rel_l2(ww_opt, ww_ref)

        fn_b = bench_cg if T <= 64 else bench_cuda
        iters = 40 if T <= 64 else 15
        t_ref = fn_b(run_ref, warmup=3, iters=iters)
        t_opt = fn_b(lambda: fused_hadamard_fp8_quant_q(q, w, w_scale), warmup=3, iters=iters)
        print(f"  {T:5d} | {t_ref:7.1f}us | {t_opt:8.1f}us | {t_ref/t_opt:6.2f}x | {(t_ref-t_opt)*6/1000:+7.2f}ms | {q_err:9.2e} | {w_err:9.2e}")
        del q, w, qq_ref, ww_ref, qq_opt, ww_opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_mhc()
    test_hca_dsa_dispatch()
    test_dsa_indexer_fused()
    print("ALL PARITY & SPEED VERIFICATIONS COMPLETED!")
