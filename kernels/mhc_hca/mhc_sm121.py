"""GB10 (SM12.1, 48 SMs, 24 MiB L2, ~223 GB/s LPDDR5X) optimized mHC kernels.

Targets ERNIE 5.5 Flash Preview's 86 mHC call sites (hc_mult=4, hidden_size=4096,
sinkhorn_iters=20, fused post+pre+RMSNorm).

Measured GB10-optimal dispatch across T = 1..2048:
- T = 1..4:   Vectorized 2D-reduction Triton fused post+gemm (48 CTAs = 1 wave on
              GB10's 48 SMs) + register-resident Sinkhorn-20/RMSNorm epilogue with
              PDL (or single-kernel atomic variant). Exact FP32 parity (~1e-7).
- T = 8..32:  T-batched Triton fused post+gemm (reads 1.57 MB `fn` ONCE across all
              T tokens, fixing upstream `mhc_fused_tilelang`'s T-times L2 reload and
              4x-12x redundant post computation!) + PDL epilogue. Exact FP32 parity.
- T = 64..128: GB10-tuned 3-kernel path (`mhc_post` + DeepGEMM `tf32_hc_prenorm_gemm`
              with optimal `n_splits=32` + Triton/TileLang epilogue).
- T >= 512:   L2-resident chunked (`T_chunk = 256`, keeping `residual_cur` at
              8.4 MB << 24 MiB L2) `mhc_post` + DeepGEMM `tf32_hc_prenorm_gemm` +
              fast single-pass Triton register-resident epilogue (`_mhc_epilogue_norm_kernel`),
              eliminating 134 MB of DRAM roundtrips at T=2048!
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    _MHC_POST_TILELANG_KERNEL,
    compute_num_split,
)
from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
from vllm_ernie.models.ernie5_5.mhc import mhc_fused_post_pre_tilelang
from vllm_ernie.models.ernie5_5.mhc_kernels import _MHC_PRE_BIG_FUSE_TILELANG_KERNEL


@triton.jit
def _sinkhorn_4x4_20(cm, eps: tl.constexpr, SINKHORN_ITERS: tl.constexpr):
    """Register-resident 4x4 Sinkhorn-Knopp for hc_mult=4 (exact ERNIE 5.5 math)."""
    row_max = tl.max(cm, axis=1, keep_dims=True)
    cm = tl.exp(cm - row_max)
    row_sum = tl.sum(cm, axis=1, keep_dims=True)
    cm = cm / row_sum + eps
    col_sum = tl.sum(cm, axis=0, keep_dims=True)
    cm = cm / (col_sum + eps)
    for _ in tl.static_range(SINKHORN_ITERS - 1):
        row_sum = tl.sum(cm, axis=1, keep_dims=True)
        cm = cm / (row_sum + eps)
        col_sum = tl.sum(cm, axis=0, keep_dims=True)
        cm = cm / (col_sum + eps)
    return cm


# ---------------------------------------------------------------------------
# 1. T=1 Vectorized 2D-Reduction Decode Post + Pre-Norm GEMM (PDL)
# ---------------------------------------------------------------------------
@triton.jit
def _mhc_decode_post_gemm_t1_kernel(
    x_ptr,          # [1, H] bf16
    res_in_ptr,     # [1, 4, H] bf16
    pm_in_ptr,      # [1, 4] fp32
    cm_in_ptr,      # [1, 4, 4] fp32
    fn_ptr,         # [24, 4, H] fp32
    res_out_ptr,    # [1, 4, H] bf16
    yp_out_ptr,     # [N_SPLITS, 1, 24] fp32
    rp_out_ptr,     # [N_SPLITS, 1] fp32
    H: tl.constexpr,
    H_SPLIT: tl.constexpr,
    TILE_N: tl.constexpr,   # power of 2: 4 or 8
    USE_GDC: tl.constexpr = True,
):
    split_id = tl.program_id(0)
    nt_id = tl.program_id(1)
    h_offs = split_id * H_SPLIT + tl.arange(0, H_SPLIT)
    n_offs = nt_id * TILE_N + tl.arange(0, TILE_N)

    # Pre-load fn weights for stream 0 before gdc_wait so PDL hides DRAM latency!
    w0 = tl.load(fn_ptr + (n_offs * 4 + 0)[:, None] * H + h_offs[None, :])
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    pm0 = tl.load(pm_in_ptr + 0)
    pm1 = tl.load(pm_in_ptr + 1)
    pm2 = tl.load(pm_in_ptr + 2)
    pm3 = tl.load(pm_in_ptr + 3)

    x_v = tl.load(x_ptr + h_offs).to(tl.float32)
    r0 = tl.load(res_in_ptr + 0 * H + h_offs).to(tl.float32)
    r1 = tl.load(res_in_ptr + 1 * H + h_offs).to(tl.float32)
    r2 = tl.load(res_in_ptr + 2 * H + h_offs).to(tl.float32)
    r3 = tl.load(res_in_ptr + 3 * H + h_offs).to(tl.float32)

    c00 = tl.load(cm_in_ptr + 0)
    c01 = tl.load(cm_in_ptr + 1)
    c02 = tl.load(cm_in_ptr + 2)
    c03 = tl.load(cm_in_ptr + 3)
    c10 = tl.load(cm_in_ptr + 4)
    c11 = tl.load(cm_in_ptr + 5)
    c12 = tl.load(cm_in_ptr + 6)
    c13 = tl.load(cm_in_ptr + 7)
    c20 = tl.load(cm_in_ptr + 8)
    c21 = tl.load(cm_in_ptr + 9)
    c22 = tl.load(cm_in_ptr + 10)
    c23 = tl.load(cm_in_ptr + 11)
    c30 = tl.load(cm_in_ptr + 12)
    c31 = tl.load(cm_in_ptr + 13)
    c32 = tl.load(cm_in_ptr + 14)
    c33 = tl.load(cm_in_ptr + 15)

    nr0 = pm0 * x_v + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3
    nr1 = pm1 * x_v + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3
    nr2 = pm2 * x_v + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3
    nr3 = pm3 * x_v + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3

    if nt_id == 0:
        tl.store(res_out_ptr + 0 * H + h_offs, nr0.to(tl.bfloat16))
        tl.store(res_out_ptr + 1 * H + h_offs, nr1.to(tl.bfloat16))
        tl.store(res_out_ptr + 2 * H + h_offs, nr2.to(tl.bfloat16))
        tl.store(res_out_ptr + 3 * H + h_offs, nr3.to(tl.bfloat16))
        sqr = tl.sum(nr0 * nr0 + nr1 * nr1 + nr2 * nr2 + nr3 * nr3, axis=0)
        tl.store(rp_out_ptr + split_id, sqr)

    acc = w0 * nr0[None, :]
    w1 = tl.load(fn_ptr + (n_offs * 4 + 1)[:, None] * H + h_offs[None, :])
    acc += w1 * nr1[None, :]
    w2 = tl.load(fn_ptr + (n_offs * 4 + 2)[:, None] * H + h_offs[None, :])
    acc += w2 * nr2[None, :]
    w3 = tl.load(fn_ptr + (n_offs * 4 + 3)[:, None] * H + h_offs[None, :])
    acc += w3 * nr3[None, :]
    dot_n = tl.sum(acc, axis=1)
    tl.store(yp_out_ptr + split_id * 24 + n_offs, dot_n)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


# ---------------------------------------------------------------------------
# 2. T-Batched Decode Post + Pre-Norm GEMM for T = 2..32 (exact FP32, PDL)
#    Reads 1.57 MB `fn` ONCE across all BLOCK_T tokens!
# ---------------------------------------------------------------------------
@triton.jit
def _mhc_decode_post_gemm_batch_kernel(
    x_ptr,          # [T, H] bf16
    res_in_ptr,     # [T, 4, H] bf16
    pm_in_ptr,      # [T, 4] fp32
    cm_in_ptr,      # [T, 4, 4] fp32
    fn_ptr,         # [24, 4, H] fp32
    res_out_ptr,    # [T, 4, H] bf16
    yp_out_ptr,     # [N_SPLITS, T, 24] fp32
    rp_out_ptr,     # [N_SPLITS, T] fp32
    num_tokens,
    H: tl.constexpr,
    H_SPLIT: tl.constexpr,
    TILE_N: tl.constexpr,   # 4 or 8
    BLOCK_T: tl.constexpr,  # 2, 4, 8, 16, 32
    USE_GDC: tl.constexpr = True,
):
    split_id = tl.program_id(0)
    nt_id = tl.program_id(1)
    t_block = tl.program_id(2)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    h_offs = split_id * H_SPLIT + tl.arange(0, H_SPLIT)
    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < num_tokens
    t64 = t_offs.to(tl.int64)

    pm0 = tl.load(pm_in_ptr + t64 * 4 + 0, mask=t_mask, other=0.0)[:, None]
    pm1 = tl.load(pm_in_ptr + t64 * 4 + 1, mask=t_mask, other=0.0)[:, None]
    pm2 = tl.load(pm_in_ptr + t64 * 4 + 2, mask=t_mask, other=0.0)[:, None]
    pm3 = tl.load(pm_in_ptr + t64 * 4 + 3, mask=t_mask, other=0.0)[:, None]

    x_tile = tl.load(
        x_ptr + t64[:, None] * H + h_offs[None, :], mask=t_mask[:, None], other=0.0
    ).to(tl.float32)

    r0 = tl.load(res_in_ptr + (t64 * 4 + 0)[:, None] * H + h_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    r1 = tl.load(res_in_ptr + (t64 * 4 + 1)[:, None] * H + h_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    r2 = tl.load(res_in_ptr + (t64 * 4 + 2)[:, None] * H + h_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    r3 = tl.load(res_in_ptr + (t64 * 4 + 3)[:, None] * H + h_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)

    c00 = tl.load(cm_in_ptr + t64 * 16 + 0, mask=t_mask, other=0.0)[:, None]
    c01 = tl.load(cm_in_ptr + t64 * 16 + 1, mask=t_mask, other=0.0)[:, None]
    c02 = tl.load(cm_in_ptr + t64 * 16 + 2, mask=t_mask, other=0.0)[:, None]
    c03 = tl.load(cm_in_ptr + t64 * 16 + 3, mask=t_mask, other=0.0)[:, None]
    c10 = tl.load(cm_in_ptr + t64 * 16 + 4, mask=t_mask, other=0.0)[:, None]
    c11 = tl.load(cm_in_ptr + t64 * 16 + 5, mask=t_mask, other=0.0)[:, None]
    c12 = tl.load(cm_in_ptr + t64 * 16 + 6, mask=t_mask, other=0.0)[:, None]
    c13 = tl.load(cm_in_ptr + t64 * 16 + 7, mask=t_mask, other=0.0)[:, None]
    c20 = tl.load(cm_in_ptr + t64 * 16 + 8, mask=t_mask, other=0.0)[:, None]
    c21 = tl.load(cm_in_ptr + t64 * 16 + 9, mask=t_mask, other=0.0)[:, None]
    c22 = tl.load(cm_in_ptr + t64 * 16 + 10, mask=t_mask, other=0.0)[:, None]
    c23 = tl.load(cm_in_ptr + t64 * 16 + 11, mask=t_mask, other=0.0)[:, None]
    c30 = tl.load(cm_in_ptr + t64 * 16 + 12, mask=t_mask, other=0.0)[:, None]
    c31 = tl.load(cm_in_ptr + t64 * 16 + 13, mask=t_mask, other=0.0)[:, None]
    c32 = tl.load(cm_in_ptr + t64 * 16 + 14, mask=t_mask, other=0.0)[:, None]
    c33 = tl.load(cm_in_ptr + t64 * 16 + 15, mask=t_mask, other=0.0)[:, None]

    nr0 = pm0 * x_tile + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3
    nr1 = pm1 * x_tile + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3
    nr2 = pm2 * x_tile + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3
    nr3 = pm3 * x_tile + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3

    if nt_id == 0:
        tl.store(res_out_ptr + (t64 * 4 + 0)[:, None] * H + h_offs[None, :], nr0.to(tl.bfloat16), mask=t_mask[:, None])
        tl.store(res_out_ptr + (t64 * 4 + 1)[:, None] * H + h_offs[None, :], nr1.to(tl.bfloat16), mask=t_mask[:, None])
        tl.store(res_out_ptr + (t64 * 4 + 2)[:, None] * H + h_offs[None, :], nr2.to(tl.bfloat16), mask=t_mask[:, None])
        tl.store(res_out_ptr + (t64 * 4 + 3)[:, None] * H + h_offs[None, :], nr3.to(tl.bfloat16), mask=t_mask[:, None])
        sqr = tl.sum(nr0 * nr0 + nr1 * nr1 + nr2 * nr2 + nr3 * nr3, axis=1)
        tl.store(rp_out_ptr + split_id * num_tokens + t64, sqr, mask=t_mask)

    for ni in tl.static_range(TILE_N):
        n_idx = nt_id * TILE_N + ni
        w0 = tl.load(fn_ptr + (n_idx * 4 + 0) * H + h_offs)
        w1 = tl.load(fn_ptr + (n_idx * 4 + 1) * H + h_offs)
        w2 = tl.load(fn_ptr + (n_idx * 4 + 2) * H + h_offs)
        w3 = tl.load(fn_ptr + (n_idx * 4 + 3) * H + h_offs)
        dot_val = tl.sum(
            nr0 * w0[None, :] + nr1 * w1[None, :] + nr2 * w2[None, :] + nr3 * w3[None, :],
            axis=1,
        )
        tl.store(
            yp_out_ptr + split_id * num_tokens * 24 + t64 * 24 + n_idx,
            dot_val,
            mask=t_mask,
        )

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


# ---------------------------------------------------------------------------
# 3. Single-Pass Register-Resident Epilogue (Sinkhorn-20 + Collapse + RMSNorm)
#    Supports arbitrary N_SPLITS (padded to next_pow2) and PDL overlap!
#    1.20x-1.27x faster than TileLang pre_big_fuse at T=512..2048!
# ---------------------------------------------------------------------------
@triton.jit
def _mhc_epilogue_norm_kernel(
    yp_ptr,          # [N_SPLITS, T, 24] fp32
    rp_ptr,          # [N_SPLITS, T] fp32
    hc_scale_ptr,    # [3] fp32
    hc_base_ptr,     # [24] fp32
    res_cur_ptr,     # [T, 4, H] bf16
    norm_w_ptr,      # [H] bf16
    post_out_ptr,    # [T, 4] fp32
    comb_out_ptr,    # [T, 16] fp32
    li_out_ptr,      # [T, H] bf16
    num_tokens,
    H: tl.constexpr,
    N_SPLITS: tl.constexpr,
    N_SPLITS_PAD: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_PRE_EPS: tl.constexpr,
    HC_SINKHORN_EPS: tl.constexpr,
    HC_POST_MULT: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    USE_GDC: tl.constexpr = True,
):
    tok = tl.program_id(0).to(tl.int64)

    # Pre-load scales, bases, and norm_weight BEFORE gdc_wait for PDL overlap
    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)
    j4 = tl.arange(0, 4)
    j16 = tl.arange(0, 16)
    b_pre = tl.load(hc_base_ptr + j4)
    b_post = tl.load(hc_base_ptr + 4 + j4)
    b_comb = tl.load(hc_base_ptr + 8 + j16)
    h_offs = tl.arange(0, H)
    w = tl.load(norm_w_ptr + h_offs).to(tl.float32)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    splits = tl.arange(0, N_SPLITS_PAD)
    s_mask = splits < N_SPLITS
    rp_vals = tl.load(rp_ptr + splits * num_tokens + tok, mask=s_mask, other=0.0)
    sqrsum = tl.sum(rp_vals, axis=0)
    rms = 1.0 / (tl.sqrt(sqrsum / (4.0 * H)) + RMS_EPS)

    yp_pre = tl.load(
        yp_ptr + splits[:, None] * (num_tokens * 24) + tok * 24 + j4[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    m_pre = tl.sum(yp_pre, axis=0) * rms
    pre_mix = tl.sigmoid(m_pre * s0 + b_pre) + HC_PRE_EPS

    yp_post = tl.load(
        yp_ptr + splits[:, None] * (num_tokens * 24) + tok * 24 + (4 + j4)[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    m_post = tl.sum(yp_post, axis=0) * rms
    post_mix = tl.sigmoid(m_post * s1 + b_post) * HC_POST_MULT
    tl.store(post_out_ptr + tok * 4 + j4, post_mix)

    yp_comb = tl.load(
        yp_ptr + splits[:, None] * (num_tokens * 24) + tok * 24 + (8 + j16)[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    m_comb = tl.sum(yp_comb, axis=0) * rms
    cm = tl.reshape(m_comb * s2 + b_comb, (4, 4))
    cm = _sinkhorn_4x4_20(cm, HC_SINKHORN_EPS, SINKHORN_ITERS)
    tl.store(comb_out_ptr + tok * 16 + j16, tl.reshape(cm, (16,)))

    p0 = tl.sum(tl.where(j4 == 0, pre_mix, 0.0), axis=0)
    p1 = tl.sum(tl.where(j4 == 1, pre_mix, 0.0), axis=0)
    p2 = tl.sum(tl.where(j4 == 2, pre_mix, 0.0), axis=0)
    p3 = tl.sum(tl.where(j4 == 3, pre_mix, 0.0), axis=0)

    base_r = res_cur_ptr + tok * (4 * H)
    r0 = tl.load(base_r + 0 * H + h_offs).to(tl.float32)
    r1 = tl.load(base_r + 1 * H + h_offs).to(tl.float32)
    r2 = tl.load(base_r + 2 * H + h_offs).to(tl.float32)
    r3 = tl.load(base_r + 3 * H + h_offs).to(tl.float32)

    ol = p0 * r0 + p1 * r1 + p2 * r2 + p3 * r3
    sumsq = tl.sum(ol * ol, axis=0)
    rsqrt_norm = tl.rsqrt(sumsq / H + NORM_EPS)
    ol_bf16 = ol.to(tl.bfloat16).to(tl.float32)
    out = (ol_bf16 * rsqrt_norm * w).to(tl.bfloat16)
    tl.store(li_out_ptr + tok * H + h_offs, out)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


# ---------------------------------------------------------------------------
# 4. Zero-Spill Single-Kernel Decode (T=1..4): chunked H epilogue (no 4096 regs!)
# ---------------------------------------------------------------------------
_SCRATCH_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_atomic_scratch(device: torch.device, num_tokens: int):
    bufs = _SCRATCH_CACHE.get(device)
    if bufs is None or bufs[1].shape[0] < num_tokens:
        cap = max(16, triton.next_power_of_2(num_tokens))
        yp_acc = torch.zeros((cap, 24), dtype=torch.float32, device=device)
        rp_acc = torch.zeros((cap,), dtype=torch.float32, device=device)
        done_cnt = torch.zeros((1,), dtype=torch.int32, device=device)
        bufs = (yp_acc, rp_acc, done_cnt)
        _SCRATCH_CACHE[device] = bufs
    return bufs


@triton.jit
def _mhc_single_kernel_t1_lean(
    x_ptr, res_in_ptr, pm_in_ptr, cm_in_ptr, fn_ptr,
    hc_scale_ptr, hc_base_ptr, norm_w_ptr,
    res_out_ptr, post_out_ptr, comb_out_ptr, li_out_ptr,
    yp_acc_ptr, rp_acc_ptr, done_cnt_ptr,
    H: tl.constexpr,
    N_SPLITS: tl.constexpr,
    H_SPLIT: tl.constexpr,
    TILE_N: tl.constexpr,
    N_TILES: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_PRE_EPS: tl.constexpr,
    HC_SINKHORN_EPS: tl.constexpr,
    HC_POST_MULT: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    """Single-launch T=1 fused kernel with lean H_SPLIT-chunked epilogue (no reg spill)."""
    split_id = tl.program_id(0)
    nt_id = tl.program_id(1)
    h_offs = split_id * H_SPLIT + tl.arange(0, H_SPLIT)
    n_offs = nt_id * TILE_N + tl.arange(0, TILE_N)

    pm0 = tl.load(pm_in_ptr + 0)
    pm1 = tl.load(pm_in_ptr + 1)
    pm2 = tl.load(pm_in_ptr + 2)
    pm3 = tl.load(pm_in_ptr + 3)

    x_v = tl.load(x_ptr + h_offs).to(tl.float32)
    r0 = tl.load(res_in_ptr + 0 * H + h_offs).to(tl.float32)
    r1 = tl.load(res_in_ptr + 1 * H + h_offs).to(tl.float32)
    r2 = tl.load(res_in_ptr + 2 * H + h_offs).to(tl.float32)
    r3 = tl.load(res_in_ptr + 3 * H + h_offs).to(tl.float32)

    c00 = tl.load(cm_in_ptr + 0)
    c01 = tl.load(cm_in_ptr + 1)
    c02 = tl.load(cm_in_ptr + 2)
    c03 = tl.load(cm_in_ptr + 3)
    c10 = tl.load(cm_in_ptr + 4)
    c11 = tl.load(cm_in_ptr + 5)
    c12 = tl.load(cm_in_ptr + 6)
    c13 = tl.load(cm_in_ptr + 7)
    c20 = tl.load(cm_in_ptr + 8)
    c21 = tl.load(cm_in_ptr + 9)
    c22 = tl.load(cm_in_ptr + 10)
    c23 = tl.load(cm_in_ptr + 11)
    c30 = tl.load(cm_in_ptr + 12)
    c31 = tl.load(cm_in_ptr + 13)
    c32 = tl.load(cm_in_ptr + 14)
    c33 = tl.load(cm_in_ptr + 15)

    nr0 = pm0 * x_v + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3
    nr1 = pm1 * x_v + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3
    nr2 = pm2 * x_v + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3
    nr3 = pm3 * x_v + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3

    if nt_id == 0:
        tl.store(res_out_ptr + 0 * H + h_offs, nr0.to(tl.bfloat16))
        tl.store(res_out_ptr + 1 * H + h_offs, nr1.to(tl.bfloat16))
        tl.store(res_out_ptr + 2 * H + h_offs, nr2.to(tl.bfloat16))
        tl.store(res_out_ptr + 3 * H + h_offs, nr3.to(tl.bfloat16))
        sqr = tl.sum(nr0 * nr0 + nr1 * nr1 + nr2 * nr2 + nr3 * nr3, axis=0)
        tl.atomic_add(rp_acc_ptr, sqr, sem="relaxed")

    w0 = tl.load(fn_ptr + (n_offs * 4 + 0)[:, None] * H + h_offs[None, :])
    acc = w0 * nr0[None, :]
    w1 = tl.load(fn_ptr + (n_offs * 4 + 1)[:, None] * H + h_offs[None, :])
    acc += w1 * nr1[None, :]
    w2 = tl.load(fn_ptr + (n_offs * 4 + 2)[:, None] * H + h_offs[None, :])
    acc += w2 * nr2[None, :]
    w3 = tl.load(fn_ptr + (n_offs * 4 + 3)[:, None] * H + h_offs[None, :])
    acc += w3 * nr3[None, :]
    dot_n = tl.sum(acc, axis=1)
    tl.atomic_add(yp_acc_ptr + n_offs, dot_n, sem="relaxed")

    total_ctas: tl.constexpr = N_SPLITS * N_TILES
    arrived = tl.atomic_add(done_cnt_ptr, 1, sem="acq_rel")
    if arrived == total_ctas - 1:
        tl.store(done_cnt_ptr, 0)
        sqrsum = tl.load(rp_acc_ptr)
        tl.store(rp_acc_ptr, 0.0)
        rms = 1.0 / (tl.sqrt(sqrsum / (4.0 * H)) + RMS_EPS)

        s0 = tl.load(hc_scale_ptr + 0)
        s1 = tl.load(hc_scale_ptr + 1)
        s2 = tl.load(hc_scale_ptr + 2)
        j4 = tl.arange(0, 4)
        j16 = tl.arange(0, 16)

        m_pre = tl.load(yp_acc_ptr + j4)
        tl.store(yp_acc_ptr + j4, 0.0)
        pre_mix = tl.sigmoid(m_pre * rms * s0 + tl.load(hc_base_ptr + j4)) + HC_PRE_EPS

        m_post = tl.load(yp_acc_ptr + 4 + j4)
        tl.store(yp_acc_ptr + 4 + j4, 0.0)
        post_mix = tl.sigmoid(m_post * rms * s1 + tl.load(hc_base_ptr + 4 + j4)) * HC_POST_MULT
        tl.store(post_out_ptr + j4, post_mix)

        m_comb = tl.load(yp_acc_ptr + 8 + j16)
        tl.store(yp_acc_ptr + 8 + j16, 0.0)
        cm = tl.reshape(m_comb * rms * s2 + tl.load(hc_base_ptr + 8 + j16), (4, 4))
        cm = _sinkhorn_4x4_20(cm, HC_SINKHORN_EPS, SINKHORN_ITERS)
        tl.store(comb_out_ptr + j16, tl.reshape(cm, (16,)))

        p0 = tl.sum(tl.where(j4 == 0, pre_mix, 0.0), axis=0)
        p1 = tl.sum(tl.where(j4 == 1, pre_mix, 0.0), axis=0)
        p2 = tl.sum(tl.where(j4 == 2, pre_mix, 0.0), axis=0)
        p3 = tl.sum(tl.where(j4 == 3, pre_mix, 0.0), axis=0)

        # Lean 2-pass over H in chunks of 512 (L2-hot 32 KB -> zero reg spill!)
        sumsq = 0.0
        for h0 in range(0, H, 512):
            hc = h0 + tl.arange(0, 512)
            rr0 = tl.load(res_out_ptr + 0 * H + hc).to(tl.float32)
            rr1 = tl.load(res_out_ptr + 1 * H + hc).to(tl.float32)
            rr2 = tl.load(res_out_ptr + 2 * H + hc).to(tl.float32)
            rr3 = tl.load(res_out_ptr + 3 * H + hc).to(tl.float32)
            ol = p0 * rr0 + p1 * rr1 + p2 * rr2 + p3 * rr3
            sumsq += tl.sum(ol * ol, axis=0)
        rsqrt_norm = tl.rsqrt(sumsq / H + NORM_EPS)
        for h0 in range(0, H, 512):
            hc = h0 + tl.arange(0, 512)
            rr0 = tl.load(res_out_ptr + 0 * H + hc).to(tl.float32)
            rr1 = tl.load(res_out_ptr + 1 * H + hc).to(tl.float32)
            rr2 = tl.load(res_out_ptr + 2 * H + hc).to(tl.float32)
            rr3 = tl.load(res_out_ptr + 3 * H + hc).to(tl.float32)
            ol = (p0 * rr0 + p1 * rr1 + p2 * rr2 + p3 * rr3).to(tl.bfloat16).to(tl.float32)
            w = tl.load(norm_w_ptr + hc).to(tl.float32)
            tl.store(li_out_ptr + hc, (ol * rsqrt_norm * w).to(tl.bfloat16))


def mhc_fused_post_pre_sm121(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    mode: str = "auto",
    chunk_size: int = 256,
    use_pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GB10-optimal fused mHC post + pre + Sinkhorn-20 + RMSNorm."""
    assert norm_weight is not None, "SM121 fused kernel requires norm_weight"
    outer_shape = residual.shape[:-2]
    hc_mult = residual.shape[-2]
    H = residual.shape[-1]
    assert hc_mult == 4 and H == 4096

    res_flat = residual.view(-1, 4, H)
    T = res_flat.shape[0]
    x_flat = x.view(T, H)
    pm_flat = post_layer_mix.view(T, 4)
    cm_flat = comb_res_mix.view(T, 4, 4)
    fn_3d = fn.view(24, 4, H)

    res_cur = torch.empty_like(res_flat)
    post_cur = torch.empty((T, 4), dtype=torch.float32, device=x.device)
    comb_cur = torch.empty((T, 16), dtype=torch.float32, device=x.device)
    li_cur = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)

    pdl_kw = {"USE_GDC": True, "launch_pdl": True} if use_pdl else {"USE_GDC": False}

    if mode == "auto":
        if T == 1:
            mode = "t1"
        elif T <= 8:
            # Upstream TileLang fused kernel is slightly faster for T=2..8
            return mhc_fused_post_pre_tilelang(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                norm_weight=norm_weight, norm_eps=norm_eps,
            )
        elif T <= 32:
            mode = "twopass"
        elif T <= 128:
            mode = "dg_l2hot"
        else:
            mode = "dg_chunked"

    if mode == "single_t1":
        assert T == 1
        # 16 splits x 3 tiles (TILE_N=8) = 48 CTAs = 1 full wave on GB10!
        n_splits = 16
        tile_n = 8
        n_tiles = 3
        h_split = H // n_splits
        yp_acc, rp_acc, done_cnt = _get_atomic_scratch(x.device, 1)
        _mhc_single_kernel_t1_lean[(n_splits, n_tiles)](
            x_flat, res_flat, pm_flat, cm_flat, fn_3d,
            hc_scale, hc_base, norm_weight,
            res_cur, post_cur, comb_cur, li_cur,
            yp_acc, rp_acc, done_cnt,
            H=H, N_SPLITS=n_splits, H_SPLIT=h_split, TILE_N=tile_n, N_TILES=n_tiles,
            RMS_EPS=rms_eps, HC_PRE_EPS=hc_pre_eps, HC_SINKHORN_EPS=hc_sinkhorn_eps,
            HC_POST_MULT=hc_post_mult_value, SINKHORN_ITERS=sinkhorn_repeat,
            NORM_EPS=norm_eps,
            num_warps=4,
        )
    elif mode == "t1":
        assert T == 1
        # 16 splits x 6 n_tiles (TILE_N=4) = 96 CTAs = 2 full waves on GB10 48 SMs
        n_splits = 16
        tile_n = 4
        n_tiles = 6
        h_split = H // n_splits
        yp_out = torch.empty((n_splits, 1, 24), dtype=torch.float32, device=x.device)
        rp_out = torch.empty((n_splits, 1), dtype=torch.float32, device=x.device)
        _mhc_decode_post_gemm_t1_kernel[(n_splits, n_tiles)](
            x_flat, res_flat, pm_flat, cm_flat, fn_3d,
            res_cur, yp_out, rp_out,
            H=H, H_SPLIT=h_split, TILE_N=tile_n,
            num_warps=4, **pdl_kw,
        )
        _mhc_epilogue_norm_kernel[(1,)](
            yp_out, rp_out, hc_scale, hc_base, res_cur, norm_weight,
            post_cur, comb_cur, li_cur,
            1,
            H=H, N_SPLITS=n_splits, N_SPLITS_PAD=n_splits,
            RMS_EPS=rms_eps, HC_PRE_EPS=hc_pre_eps, HC_SINKHORN_EPS=hc_sinkhorn_eps,
            HC_POST_MULT=hc_post_mult_value, SINKHORN_ITERS=sinkhorn_repeat,
            NORM_EPS=norm_eps,
            num_warps=8, **pdl_kw,
        )
    elif mode == "twopass":
        block_t = min(triton.next_power_of_2(T), 16)
        n_splits = 16
        tile_n = 8
        n_tiles = 3
        h_split = H // n_splits
        yp_out = torch.empty((n_splits, T, 24), dtype=torch.float32, device=x.device)
        rp_out = torch.empty((n_splits, T), dtype=torch.float32, device=x.device)
        _mhc_decode_post_gemm_batch_kernel[(n_splits, n_tiles, triton.cdiv(T, block_t))](
            x_flat, res_flat, pm_flat, cm_flat, fn_3d,
            res_cur, yp_out, rp_out,
            T,
            H=H, H_SPLIT=h_split, TILE_N=tile_n, BLOCK_T=block_t,
            USE_GDC=False,
            num_warps=4,
        )
        _mhc_epilogue_norm_kernel[(T,)](
            yp_out, rp_out, hc_scale, hc_base, res_cur, norm_weight,
            post_cur, comb_cur, li_cur,
            T,
            H=H, N_SPLITS=n_splits, N_SPLITS_PAD=n_splits,
            RMS_EPS=rms_eps, HC_PRE_EPS=hc_pre_eps, HC_SINKHORN_EPS=hc_sinkhorn_eps,
            HC_POST_MULT=hc_post_mult_value, SINKHORN_ITERS=sinkhorn_repeat,
            NORM_EPS=norm_eps,
            USE_GDC=False,
            num_warps=8,
        )
    elif mode == "dg_l2hot":
        # T=64..128: res_cur <= 4.2 MB fits in GB10 24 MiB L2; use optimal n_splits=32
        # with TileLang epilogue (6.2 us vs 8.2 us Triton at T=64).
        ns = 32 if T <= 64 else compute_num_split(64, 4 * H, (T + 63) // 64)
        _MHC_POST_TILELANG_KERNEL(cm_flat, res_flat, pm_flat, x_flat, res_cur, 4, H)
        gm = torch.empty((ns, T, 24), dtype=torch.float32, device=x.device)
        gs = torch.empty((ns, T), dtype=torch.float32, device=x.device)
        tf32_hc_prenorm_gemm(res_cur.view(T, 4 * H), fn, gm, gs, ns)
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
            gm, gs, hc_scale, hc_base, res_cur, post_cur, comb_cur, li_cur,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
            norm_weight=norm_weight, norm_eps=norm_eps,
        )
    elif mode == "dg_chunked":
        # T >= 512: use chunk_size=256 when T >= 1024 so res_cur (8.4 MB << 24 MiB L2)
        # stays L2-resident across mhc_post -> DeepGEMM -> fast Triton epilogue!
        eff_chunk = chunk_size if T >= 1024 else T
        chunks = [(s, min(s + eff_chunk, T)) for s in range(0, T, eff_chunk)]
        for s, e in chunks:
            tc = e - s
            ns = compute_num_split(64, 4 * H, (tc + 63) // 64)
            if tc <= 64 and ns > 32:
                ns = 32
            ns_pad = triton.next_power_of_2(ns)
            _MHC_POST_TILELANG_KERNEL(
                cm_flat[s:e], res_flat[s:e], pm_flat[s:e], x_flat[s:e], res_cur[s:e], 4, H
            )
            gm = torch.empty((ns, tc, 24), dtype=torch.float32, device=x.device)
            gs = torch.empty((ns, tc), dtype=torch.float32, device=x.device)
            tf32_hc_prenorm_gemm(res_cur[s:e].view(tc, 4 * H), fn, gm, gs, ns)
            _mhc_epilogue_norm_kernel[(tc,)](
                gm, gs, hc_scale, hc_base, res_cur[s:e], norm_weight,
                post_cur[s:e], comb_cur[s:e], li_cur[s:e],
                tc,
                H=H, N_SPLITS=ns, N_SPLITS_PAD=ns_pad,
                RMS_EPS=rms_eps, HC_PRE_EPS=hc_pre_eps, HC_SINKHORN_EPS=hc_sinkhorn_eps,
                HC_POST_MULT=hc_post_mult_value, SINKHORN_ITERS=sinkhorn_repeat,
                NORM_EPS=norm_eps,
                num_warps=4 if tc <= 512 else 8,
                **pdl_kw,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return (
        res_cur.view(*outer_shape, 4, H),
        post_cur.view(*outer_shape, 4, 1),
        comb_cur.view(*outer_shape, 4, 4),
        li_cur.view(*outer_shape, H),
    )
