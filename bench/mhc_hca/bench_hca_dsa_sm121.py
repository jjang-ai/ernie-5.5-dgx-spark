#!/usr/bin/env python3
"""Benchmark ERNIE 5.5 HCA (37 layers), DSA (6 layers) & Lightning Indexer on GB10 (SM12.1).

Profiles:
1. HCA FlashInfer SM120 sparse MLA (`trtllm_batch_decode_sparse_mla_dsv4`) across
   decode (B = 1, 8, 16, 64; kv_len = 512, 2048, 8192, 16384, 32768) and
   prefill (seq_len = 512, 2048, 8192), plus FlashInfer SM120 calibration status.
2. HCA epilogue ops: `vha_inv_rope_fp8_quant`, `sigmoid_mul_quant_fp8` (HCA 8192 & DSA 16384),
   and `VHAPostMix` (`torch.matmul`).
3. DSA FlashInfer SM120 sparse MLA (`trtllm_batch_decode_with_kv_cache_mla` with
   `sparse_mla_top_k=2176`, `sinks`), `triton_convert_req_index_to_global_index`,
   and `merge_swa_topk`.
4. Lightning Indexer ops: `hadamard_rows`, `per_token_group_quant_fp8` (active SM12.1 path),
   `downcast_to_mxfp` + `_store_indexer_fp4` (SM100-only path), DeepGEMM
   `fp8_fp4_paged_mqa_logits` / `fp8_fp4_mqa_logits`, `top_k_per_row_decode` /
   `top_k_per_row_prefill`, and `get_paged_mqa_logits_metadata`.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flashinfer
import flashinfer.mla as fi_mla
import flashinfer.mla._sparse_mla_sm120 as sm120_mla
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import downcast_to_mxfp
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm_ernie.models.ernie5_5.nvidia.attention import (
    _store_indexer_fp4,
    hadamard_rows,
    merge_swa_topk,
    sigmoid_mul_quant_fp8,
    vha_inv_rope_fp8_quant,
)

# ERNIE 5.5 attention constants
NUM_HCA_LAYERS = 37
NUM_DSA_LAYERS = 6
HCA_HEADS = 64
HCA_HEAD_DIM = 512
HCA_NOPE_DIM = 448
HCA_ROPE_DIM = 64
HCA_O_GROUPS = 8
HCA_O_LORA_RANK = 1024
HCA_WINDOW = 128
HCA_COMPRESS_RATIO = 128

DSA_HEADS = 64
DSA_KV_LORA_RANK = 512
DSA_QK_NOPE_DIM = 192
DSA_QK_ROPE_DIM = 64
DSA_V_DIM = 256
DSA_TOPK = 2048
DSA_WINDOW = 128
DSA_SPARSE_CAP = DSA_TOPK + DSA_WINDOW  # 2176
DSA_PAGE_SIZE = 64

IDX_HEADS = 64
IDX_HEAD_DIM = 128


def bench_cuda(fn: Callable[[], None], warmup: int = 5, iters: int = 30) -> float:
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


def bench_cg(fn: Callable[[], None], warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
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


def check_sm120_calibration():
    print("=== 0. FlashInfer SM120 Calibration & Config Inspection ===")
    cfgs = fi_mla.supported_sparse_mla_sm120_configs()
    for k in ("dsv4", "dsv3_2"):
        print(f"  {k}: {cfgs[k]}")
    try:
        import flashinfer.mla._sparse_mla_sm120_cpb as cpb
        cache_path = getattr(cpb, "_cache_path", None)
        if callable(cache_path):
            p = cache_path()
            print(f"  SM120 calibration cache path: {p} (exists={os.path.exists(p)})")
        else:
            print(f"  cpb attrs: {[a for a in dir(cpb) if 'cache' in a.lower() or 'cal' in a.lower() or 'load' in a.lower()]}")
    except Exception as ex:
        print(f"  cpb check err: {ex}")
    print()


def bench_hca_attention():
    print("=== 1. HCA Attention: FlashInfer SM120 `trtllm_batch_decode_sparse_mla_dsv4` (37 layers) ===")
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sinks = torch.zeros(HCA_HEADS, dtype=torch.float32, device="cuda")
    scale = HCA_HEAD_DIM ** -0.5

    # Allocate realistic large page pools so sparse indices are not trivially L2-hot
    # SWA pool: 1024 pages x 64 tokens x 584B = 38.3 MB (> 24 MiB L2)
    # DSv4 page-footer layout (448B NoPE FP8 + 128B RoPE BF16 + 8B scales); bytes 48..59 are finite in FP8/BF16/FP32
    num_swa_pages = 1024
    swa_cache = torch.randint(48, 60, (num_swa_pages, 64, 1, 584), dtype=torch.uint8, device="cuda")
    num_comp_pages = 1024
    comp_cache = torch.randint(48, 60, (num_comp_pages, 64, 1, 584), dtype=torch.uint8, device="cuda")

    print("  [Decode: B x kv_len -> SWA(128) + C128A(kv_len//128)]")
    print(f"  {'B':>4} | {'kv_len':>6} | {'comp_k':>6} | {'Eager(us)':>10} | {'Graph(us)':>10} | {'37L(ms)':>8}")
    print("  " + "-" * 58)
    for B in [1, 8, 16, 64]:
        q = torch.randn(B, HCA_HEADS, HCA_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
        out = torch.empty_like(q)
        for kv_len in [512, 2048, 8192, 16384, 32768]:
            comp_k = max(1, kv_len // HCA_COMPRESS_RATIO)
            # Pad extra_sparse_indices width to multiple of 64 if needed
            comp_w = ((comp_k + 63) // 64) * 64
            swa_idx = torch.randint(0, num_swa_pages * 64, (B, 1, HCA_WINDOW), dtype=torch.int32, device="cuda")
            swa_lens = torch.full((B,), min(kv_len, HCA_WINDOW), dtype=torch.int32, device="cuda")
            comp_idx = torch.randint(0, num_comp_pages * 64, (B, 1, comp_w), dtype=torch.int32, device="cuda")
            if comp_k < comp_w:
                comp_idx[:, :, comp_k:] = -1
            comp_lens = torch.full((B,), comp_k, dtype=torch.int32, device="cuda")

            def run():
                fi_mla.trtllm_batch_decode_sparse_mla_dsv4(
                    query=q,
                    swa_kv_cache=swa_cache,
                    workspace_buffer=workspace,
                    sparse_indices=swa_idx,
                    compressed_kv_cache=comp_cache,
                    out=out,
                    bmm1_scale=scale,
                    sinks=sinks,
                    kv_layout="NHD",
                    swa_topk_lens=swa_lens,
                    extra_sparse_indices=comp_idx,
                    extra_sparse_topk_lens=comp_lens,
                )

            try:
                te = bench_cuda(run, warmup=3, iters=30)
                tg = bench_cg(run, warmup=3, iters=40)
                print(f"  {B:4d} | {kv_len:6d} | {comp_k:6d} | {te:8.1f}us | {tg:8.1f}us | {tg*NUM_HCA_LAYERS/1000:6.2f}ms")
            except Exception as ex:
                print(f"  {B:4d} | {kv_len:6d} | {comp_k:6d} | ERR: {ex}")

    print("\n  [Prefill: seq_len -> SWA(128) + C128A(seq_len//128)]")
    print(f"  {'seq_len':>7} | {'comp_k':>6} | {'Eager(us)':>10} | {'37L(ms)':>8}")
    print("  " + "-" * 44)
    for seq_len in [512, 2048, 8192]:
        q = torch.randn(seq_len, HCA_HEADS, HCA_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
        out = torch.empty_like(q)
        comp_k = max(1, seq_len // HCA_COMPRESS_RATIO)
        comp_w = ((comp_k + 63) // 64) * 64
        swa_idx = torch.randint(0, num_swa_pages * 64, (seq_len, 1, HCA_WINDOW), dtype=torch.int32, device="cuda")
        swa_lens = torch.arange(1, seq_len + 1, dtype=torch.int32, device="cuda").clamp(max=HCA_WINDOW)
        comp_idx = torch.randint(0, num_comp_pages * 64, (seq_len, 1, comp_w), dtype=torch.int32, device="cuda")
        comp_lens = (torch.arange(seq_len, dtype=torch.int32, device="cuda") // HCA_COMPRESS_RATIO).clamp(max=comp_k)

        def run_pf():
            fi_mla.trtllm_batch_decode_sparse_mla_dsv4(
                query=q,
                swa_kv_cache=swa_cache,
                workspace_buffer=workspace,
                sparse_indices=swa_idx,
                compressed_kv_cache=comp_cache,
                out=out,
                bmm1_scale=scale,
                sinks=sinks,
                kv_layout="NHD",
                swa_topk_lens=swa_lens,
                extra_sparse_indices=comp_idx,
                extra_sparse_topk_lens=comp_lens,
            )

        try:
            te = bench_cuda(run_pf, warmup=3, iters=15)
            print(f"  {seq_len:7d} | {comp_k:6d} | {te:8.1f}us | {te*NUM_HCA_LAYERS/1000:6.2f}ms")
        except Exception as ex:
            print(f"  {seq_len:7d} | {comp_k:6d} | ERR: {ex}")
        del q, out, swa_idx, swa_lens, comp_idx, comp_lens
    del swa_cache, comp_cache, workspace
    torch.cuda.empty_cache()
    print()


def bench_hca_dsa_epilogue_ops():
    print("=== 2. HCA/DSA Post-Attention Ops (vha_inv_rope_fp8_quant, sigmoid_mul_quant_fp8, VHA bmm) ===")
    vha_m = (torch.eye(HCA_HEADS, dtype=torch.bfloat16, device="cuda") + 0.01 * torch.randn(HCA_HEADS, HCA_HEADS, dtype=torch.bfloat16, device="cuda"))
    cos_sin = torch.randn(16384, HCA_ROPE_DIM, dtype=torch.float32, device="cuda")

    print(f"  {'T':>5} | {'vha_inv_rope_q':>15} | {'sig_mul_q(8K)':>14} | {'sig_mul_q(16K)':>15} | {'vha_bmm(DSA)':>13}")
    print("  " + "-" * 74)
    for T in [1, 8, 16, 64, 512, 2048, 8192]:
        o_hca = torch.randn(T, HCA_HEADS, HCA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        pos = torch.arange(T, dtype=torch.int64, device="cuda")
        z8 = torch.randn(T, 8192, dtype=torch.bfloat16, device="cuda")
        g8 = torch.randn(T, 8192, dtype=torch.bfloat16, device="cuda")
        z16 = torch.randn(T, 16384, dtype=torch.bfloat16, device="cuda")
        g16 = torch.randn(T, 16384, dtype=torch.bfloat16, device="cuda")
        o_dsa = torch.randn(T, DSA_HEADS, DSA_V_DIM, dtype=torch.bfloat16, device="cuda")
        iters = 50 if T <= 64 else (20 if T <= 2048 else 10)
        fn_bench = bench_cg if T <= 64 else bench_cuda

        t_vha_q = fn_bench(lambda: vha_inv_rope_fp8_quant(o_hca, vha_m, pos, cos_sin, HCA_O_GROUPS, HCA_NOPE_DIM, HCA_ROPE_DIM), iters=iters)
        t_sig8 = fn_bench(lambda: sigmoid_mul_quant_fp8(z8, g8), iters=iters)
        t_sig16 = fn_bench(lambda: sigmoid_mul_quant_fp8(z16, g16), iters=iters)
        t_vha_dsa = fn_bench(lambda: torch.matmul(vha_m, o_dsa), iters=iters)
        print(f"  {T:5d} | {t_vha_q:13.1f}us | {t_sig8:12.1f}us | {t_sig16:13.1f}us | {t_vha_dsa:11.1f}us")
        del o_hca, pos, z8, g8, z16, g16, o_dsa
    torch.cuda.empty_cache()
    print()


def bench_dsa_and_indexer():
    print("=== 3. DSA Sparse MLA SM120 (`trtllm_batch_decode_with_kv_cache_mla`, topk=2176, sinks) (6 layers) ===")
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sinks = torch.zeros(DSA_HEADS, dtype=torch.float32, device="cuda")
    scale = (DSA_QK_NOPE_DIM + DSA_QK_ROPE_DIM) ** -0.5

    # DSA KV cache: 2048 pages x 64 tokens x 656B = 86 MB (> 24 MiB L2)
    num_pages = 2048
    kv_cache = torch.zeros((num_pages, 1, DSA_PAGE_SIZE, 656), dtype=torch.uint8, device="cuda")
    kv_cache[..., :512] = (
        (torch.randn(num_pages, 1, DSA_PAGE_SIZE, 512, dtype=torch.bfloat16, device="cuda") * 0.25)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    kv_cache[..., 512:528].view(torch.float32).fill_(1.0)
    kv_cache[..., 528:] = (
        (torch.randn(num_pages, 1, DSA_PAGE_SIZE, 64, dtype=torch.bfloat16, device="cuda") * 0.25)
        .view(torch.uint8)
    )

    print("  [DSA Sparse MLA Decode & Prefill (sparse_capacity=2176 = 2048 topk + 128 SWA)]")
    print(f"  {'T':>5} | {'active_k':>8} | {'Eager(us)':>10} | {'Graph(us)':>10} | {'6L(ms)':>8} | {'merge_swa':>10} | {'req2glob':>10}")
    print("  " + "-" * 78)
    for T, active_k in [(1, 512), (1, 2176), (8, 2176), (16, 2176), (64, 2176), (512, 2176), (2048, 2176), (8192, 2176)]:
        q = torch.randn(T, 1, DSA_HEADS, DSA_KV_LORA_RANK + DSA_QK_ROPE_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
        out = torch.empty(T, 1, DSA_HEADS, DSA_KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        # Physical token indices
        phys_idx = torch.randint(0, num_pages * DSA_PAGE_SIZE, (T, 1, DSA_SPARSE_CAP), dtype=torch.int32, device="cuda")
        if active_k < DSA_SPARSE_CAP:
            phys_idx[:, :, active_k:] = -1

        def run_dsa():
            fi_mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=DSA_QK_NOPE_DIM,
                kv_lora_rank=DSA_KV_LORA_RANK,
                qk_rope_head_dim=DSA_QK_ROPE_DIM,
                block_tables=phys_idx,
                seq_lens=None,
                max_seq_len=DSA_SPARSE_CAP,
                out=out,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                sparse_mla_top_k=DSA_SPARSE_CAP,
                kv_scale_format="pow2_fp32",
                sinks=sinks,
            )

        # Also benchmark merge_swa_topk and triton_convert_req_index_to_global_index
        topk_in = torch.randint(0, max(1, active_k - DSA_WINDOW), (T, DSA_TOPK), dtype=torch.int32, device="cuda")
        swa_out = torch.empty(T, DSA_SPARSE_CAP, dtype=torch.int32, device="cuda")
        pos = torch.full((T,), max(active_k, 4096) - 1, dtype=torch.int64, device="cuda")
        req_ids = torch.zeros(T, dtype=torch.int32, device="cuda")
        bt = torch.arange(min(num_pages, 512), dtype=torch.int32, device="cuda").unsqueeze(0)

        iters = 30 if T <= 64 else (10 if T <= 2048 else 5)
        try:
            te = bench_cuda(run_dsa, warmup=2, iters=iters)
            tg = bench_cg(run_dsa, warmup=2, iters=iters) if T <= 64 else te
            tm = (bench_cg if T <= 64 else bench_cuda)(lambda: merge_swa_topk(topk_in, swa_out, pos, DSA_WINDOW), iters=iters)
            tr = (bench_cg if T <= 64 else bench_cuda)(
                lambda: triton_convert_req_index_to_global_index(
                    req_ids, bt, swa_out, BLOCK_SIZE=DSA_PAGE_SIZE, NUM_TOPK_TOKENS=DSA_SPARSE_CAP
                ),
                iters=iters,
            )
            print(f"  {T:5d} | {active_k:8d} | {te:8.1f}us | {tg:8.1f}us | {tg*NUM_DSA_LAYERS/1000:6.2f}ms | {tm:8.1f}us | {tr:8.1f}us")
        except Exception as ex:
            print(f"  {T:5d} | {active_k:8d} | ERR: {ex}")
        del q, out, phys_idx, topk_in, swa_out, pos, req_ids, bt
    del kv_cache, workspace
    torch.cuda.empty_cache()

    print("\n=== 4. Lightning Indexer Ops (hadamard_rows, quant_fp8, mxfp4, paged_mqa_logits, topk) ===")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("  [4a. Prep ops: hadamard_rows(q+k), fp8_quant(q), mxfp4_quant(q+k)+store_fp4]")
    print(f"  {'T':>5} | {'hadamard(q+k)':>14} | {'fp8_quant(q)':>13} | {'mxfp4(q+k)+st':>14}")
    print("  " + "-" * 56)
    idx_fp4_cache = torch.empty(512, 64, 68, dtype=torch.uint8, device="cuda")
    for T in [1, 8, 16, 64, 512, 2048, 8192]:
        q = torch.randn(T * IDX_HEADS, IDX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(T, IDX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(T, dtype=torch.int64, device="cuda") % (512 * 64)
        iters = 40 if T <= 64 else 15
        fb = bench_cg if T <= 64 else bench_cuda

        t_had = fb(lambda: (hadamard_rows(q), hadamard_rows(k)), iters=iters)
        t_fp8 = fb(lambda: per_token_group_quant_fp8(q, 128, column_major_scales=False, use_ue8m0=True), iters=iters)
        def run_mxfp4():
            qv, qs, _ = downcast_to_mxfp(q, axis=-1, BLOCK_OUT_DIM=4 if q.shape[0] <= 1024 else 16, BLOCK_QUANT_DIM=IDX_HEAD_DIM)
            kv, ks, _ = downcast_to_mxfp(k, axis=-1, BLOCK_OUT_DIM=min(1 << (T - 1).bit_length(), 32), BLOCK_QUANT_DIM=IDX_HEAD_DIM)
            _store_indexer_fp4[(T,)](kv, ks, idx_fp4_cache, slots, kv.stride(0), ks.stride(0), idx_fp4_cache.stride(0), PAGE_SIZE=64, VALUE_BYTES=64, SCALE_BYTES=4, num_warps=1)
        try:
            t_fp4 = fb(run_mxfp4, iters=iters)
            fp4_s = f"{t_fp4:12.1f}us"
        except Exception as ex:
            fp4_s = f"ERR:{str(ex)[:9]}"
        print(f"  {T:5d} | {t_had:12.1f}us | {t_fp8:11.1f}us | {fp4_s:>14}")
        del q, k, slots
    del idx_fp4_cache
    torch.cuda.empty_cache()

    print("\n  [4b. DeepGEMM FP8 Paged MQA Logits + top_k_per_row_decode (SM12.1 active decode path)]")
    print(f"  {'B':>4} | {'kv_len':>6} | {'meta(us)':>9} | {'logits(us)':>11} | {'topk(us)':>9} | {'total(us)':>10} | {'6L(ms)':>8}")
    print("  " + "-" * 72)
    max_pages = 4096
    # DeepGEMM FP8 indexer page layout: [num_pages, block_size, 1, 132] uint8 (128 fp8 + 4 fp32 scale)
    idx_k_cache = torch.zeros((max_pages, DSA_PAGE_SIZE, 1, 132), dtype=torch.uint8, device="cuda")
    for B in [1, 8, 16, 64]:
        q_fp8 = torch.randn(B, 1, IDX_HEADS, IDX_HEAD_DIM, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        w = torch.ones(B, IDX_HEADS, dtype=torch.float32, device="cuda") * 0.01
        topk_out = torch.empty((B, DSA_TOPK), dtype=torch.int32, device="cuda")
        for kv_len in [512, 2048, 8192, 16384, 32768]:
            seq_lens_2d = torch.full((B, 1), kv_len, dtype=torch.int32, device="cuda")
            seq_lens_1d = seq_lens_2d.view(-1)
            pages_per_req = (kv_len + DSA_PAGE_SIZE - 1) // DSA_PAGE_SIZE
            bt = torch.randint(0, max_pages, (B, pages_per_req), dtype=torch.int32, device="cuda")
            try:
                sched = get_paged_mqa_logits_metadata(seq_lens_2d, DSA_PAGE_SIZE, num_sms)
                t_meta = bench_cuda(lambda: get_paged_mqa_logits_metadata(seq_lens_2d, DSA_PAGE_SIZE, num_sms), warmup=3, iters=30)
                logits = fp8_fp4_paged_mqa_logits((q_fp8, None), idx_k_cache, w, seq_lens_2d, bt, sched, max_model_len=kv_len, clean_logits=False)
                t_log = bench_cuda(lambda: fp8_fp4_paged_mqa_logits((q_fp8, None), idx_k_cache, w, seq_lens_2d, bt, sched, max_model_len=kv_len, clean_logits=False), warmup=3, iters=30)
                t_tk = bench_cuda(lambda: ops.top_k_per_row_decode(logits, 1, seq_lens_1d, topk_out, B, logits.stride(0), logits.stride(1), DSA_TOPK), warmup=3, iters=30)
                tot = t_meta + t_log + t_tk
                print(f"  {B:4d} | {kv_len:6d} | {t_meta:7.1f}us | {t_log:9.1f}us | {t_tk:7.1f}us | {tot:8.1f}us | {tot*NUM_DSA_LAYERS/1000:6.2f}ms")
                del logits, sched
            except Exception as ex:
                print(f"  {B:4d} | {kv_len:6d} | ERR: {ex}")
        del q_fp8, w, topk_out
    del idx_k_cache
    torch.cuda.empty_cache()

    print("\n  [4c. DeepGEMM FP8 Ragged Prefill MQA Logits + top_k_per_row_prefill]")
    print(f"  {'seq_len':>7} | {'logits(us)':>11} | {'topk(us)':>10} | {'total(us)':>10} | {'6L(ms)':>8}")
    print("  " + "-" * 58)
    for seq_len in [512, 2048, 8192]:
        q_fp8 = torch.randn(seq_len, IDX_HEADS, IDX_HEAD_DIM, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        k_fp8 = torch.randn(seq_len, IDX_HEAD_DIM, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        k_scale = torch.ones(seq_len, dtype=torch.float32, device="cuda")
        w = torch.ones(seq_len, IDX_HEADS, dtype=torch.float32, device="cuda") * 0.01
        ks = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
        ke = torch.arange(1, seq_len + 1, dtype=torch.int32, device="cuda").sub(DSA_WINDOW).clamp_min(0)
        topk_out = torch.empty((seq_len, DSA_TOPK), dtype=torch.int32, device="cuda")
        try:
            logits = fp8_fp4_mqa_logits((q_fp8, None), (k_fp8, k_scale), w, ks, ke, clean_logits=False)
            t_log = bench_cuda(lambda: fp8_fp4_mqa_logits((q_fp8, None), (k_fp8, k_scale), w, ks, ke, clean_logits=False), warmup=2, iters=15)
            t_tk = bench_cuda(lambda: ops.top_k_per_row_prefill(logits, ks, ke, topk_out, seq_len, logits.stride(0), logits.stride(1), DSA_TOPK), warmup=2, iters=15)
            tot = t_log + t_tk
            print(f"  {seq_len:7d} | {t_log:9.1f}us | {t_tk:8.1f}us | {tot:8.1f}us | {tot*NUM_DSA_LAYERS/1000:6.2f}ms")
            del logits
        except Exception as ex:
            print(f"  {seq_len:7d} | ERR: {ex}")
        del q_fp8, k_fp8, k_scale, w, ks, ke, topk_out
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "hca", "epi", "dsa"], default="all")
    args = parser.parse_args()
    check_sm120_calibration()
    if args.only in ("all", "hca"):
        bench_hca_attention()
    if args.only in ("all", "epi"):
        bench_hca_dsa_epilogue_ops()
    if args.only in ("all", "dsa"):
        bench_dsa_and_indexer()


if __name__ == "__main__":
    main()
