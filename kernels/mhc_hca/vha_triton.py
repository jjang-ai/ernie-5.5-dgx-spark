"""Fused SM12.1 (GB10) Triton kernels for ERNIE 5.5 VHAPostMix.

Stock vLLM-ERNIE `VHAPostMix.forward` runs two `torch.bmm` calls with
expanded `[M, H, R]` weights and FP32 intermediates across all 43 layers,
which burns ~285 ms per 4096-token chunk on TP=4 (~437 ms on TP=2) and
adds ~1.6-2.5 ms/step during decode.

Because `U` ([H, R]) and `V` ([H, R]) are shared across all `M` tokens and
tiny (16x16 or 32x16 <= 1 KB), each CTA loads `U`/`V` once into registers,
accumulates in FP32 (`tl.dot(..., out_dtype=tl.float32)`), and writes BF16
directly (with in-place `o += V @ z` to avoid allocating a second `[M, H, D]`
tensor).

Live GB10 (SM12.1) microbenchmark vs stock `torch.bmm`:
  - TP=4 (H=16, R=16, D=512): 4.23x faster at M=4096 (284.5ms -> 67.2ms), 2.20-2.64x at M=8..64
  - TP=2 (H=32, R=16, D=512): 4.00x faster at M=4096 (437.1ms -> 109.3ms), 2.19-3.38x at M=8..64
  - DSA  (H=16, R=16, D=256): 4.23x faster at M=4096 (141.0ms -> 33.3ms), 2.18-2.21x at M=8..64
  - Parity: exact BF16 1-ULP vs FP32 reference (`max_abs <= 0.0156`).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vha_u_proj_kernel(
    o_ptr,
    ut_ptr,
    z_ptr,
    M,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute z[m, r, d] = sum_h U^T[r, h] * o[m, h, d] in FP32, store BF16."""
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, H)
    offs_r = tl.arange(0, R)

    ut = tl.load(ut_ptr + offs_r[:, None] * H + offs_h[None, :])
    o = tl.load(o_ptr + pid_m * (H * D) + offs_h[:, None] * D + offs_d[None, :])
    z = tl.dot(ut, o, out_dtype=tl.float32)
    tl.store(
        z_ptr + pid_m * (R * D) + offs_r[:, None] * D + offs_d[None, :],
        z.to(tl.bfloat16),
    )


@triton.jit
def _vha_v_add_inplace_kernel(
    z_ptr,
    v_ptr,
    o_ptr,
    M,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute o[m, h, d] += sum_r V[h, r] * z[m, r, d] in FP32, store BF16 in-place."""
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, H)
    offs_r = tl.arange(0, R)

    v = tl.load(v_ptr + offs_h[:, None] * R + offs_r[None, :])
    z = tl.load(z_ptr + pid_m * (R * D) + offs_r[:, None] * D + offs_d[None, :])
    o_offs = pid_m * (H * D) + offs_h[:, None] * D + offs_d[None, :]
    o = tl.load(o_ptr + o_offs).to(tl.float32)
    out = tl.dot(v, z, out_dtype=tl.float32) + o
    tl.store(o_ptr + o_offs, out.to(tl.bfloat16))


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def vha_u_proj(o: torch.Tensor, ut_bf16: torch.Tensor) -> torch.Tensor:
    """Fused z_local = U^T @ o -> [M, R, D] BF16 (CUDA-graph safe)."""
    M, H, D = o.shape
    R = ut_bf16.shape[0]
    block_d = 128 if D >= 128 and D % 128 == 0 else 64
    z = torch.empty((M, R, D), dtype=torch.bfloat16, device=o.device)
    grid = (M, D // block_d)
    _vha_u_proj_kernel[grid](
        o, ut_bf16, z, M, H=H, R=R, D=D, BLOCK_D=block_d, num_warps=4
    )
    return z


def vha_v_add_inplace(
    z: torch.Tensor, v_bf16: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    """Fused in-place o += V @ z -> [M, H, D] BF16 (CUDA-graph safe)."""
    M, H, D = o.shape
    R = v_bf16.shape[1]
    block_d = 128 if D >= 128 and D % 128 == 0 else 64
    grid = (M, D // block_d)
    _vha_v_add_inplace_kernel[grid](
        z, v_bf16, o, M, H=H, R=R, D=D, BLOCK_D=block_d, num_warps=4
    )
    return o


def supports_vha_triton(o: torch.Tensor, R: int) -> bool:
    if not o.is_cuda or o.dtype != torch.bfloat16 or not o.is_contiguous():
        return False
    if o.ndim != 3:
        return False
    _, H, D = o.shape
    return H >= 16 and R >= 16 and _is_pow2(H) and _is_pow2(R) and (D % 64 == 0)


@triton.jit
def _vha_v_add_inv_rope_fp8_quant_kernel(
    o_ptr,
    z_ptr,
    v_ptr,
    pos_ptr,
    cs_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    o_stride_tok,
    o_stride_h,
    z_stride_tok,
    z_stride_r,
    cs_stride_pos,
    out_stride_g,
    scale_stride_g,
    scale_stride_k,
    H: tl.constexpr,
    R: tl.constexpr,
    HPG: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Fused TP>1 VHA V-mix + residual add + inverse RoPE + per-128 UE8M0 FP8 quant."""
    tok = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    heads = tl.arange(0, H)
    ranks = tl.arange(0, R)
    group = heads // HPG
    hig = heads % HPG
    scale_idx = group * scale_stride_g + hig * scale_stride_k + tok
    scale_bytes = scale_ptr + scale_idx * 4 + chunk
    if tok >= num_tokens:
        tl.store(scale_bytes, tl.zeros((H,), dtype=tl.uint8))
        return
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(v_ptr + heads[:, None] * R + ranks[None, :])
    z = tl.load(z_ptr + tok * z_stride_tok + ranks[:, None] * z_stride_r + cols[None, :])
    o = tl.load(o_ptr + tok * o_stride_tok + heads[:, None] * o_stride_h + cols[None, :]).to(
        tl.float32
    )
    # Match BF16 rounding of o_mixed = (o + V @ z).to(bf16) before inverse RoPE + FP8 quant
    y = (tl.dot(v, z, out_dtype=tl.float32) + o).to(tl.bfloat16).to(tl.float32)
    if chunk * BLOCK + BLOCK > NOPE_DIM:
        rope = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2) - NOPE_DIM // 2
        is_rope = rope >= 0
        pos = tl.load(pos_ptr + tok)
        cs = cs_ptr + pos * cs_stride_pos + tl.maximum(rope, 0)
        cos = tl.load(cs, mask=is_rope, other=1.0)[None, :]
        sin = tl.load(cs + HALF_ROPE, mask=is_rope, other=0.0)[None, :]
        y_even, y_odd = tl.split(tl.reshape(y, (H, BLOCK // 2, 2)))
        y = tl.reshape(
            tl.join(y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
            (H, BLOCK),
        )
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10)
    scale = tl.math.exp2(tl.ceil(tl.log2(amax * (1.0 / FP8_MAX))))
    y_q = tl.clamp(y / scale[:, None], -FP8_MAX, FP8_MAX)
    out_rows = out_ptr + group * out_stride_g + tok * (HPG * HEAD_DIM) + hig * HEAD_DIM
    tl.store(out_rows[:, None] + cols[None, :], y_q.to(tl.float8e4nv))
    exp_byte = (scale.to(tl.int32, bitcast=True) >> 23) & 0xFF
    tl.store(scale_bytes, exp_byte.to(tl.uint8))


def vha_v_add_inv_rope_fp8_quant(
    o: torch.Tensor,
    z: torch.Tensor,
    v_bf16: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    tma_aligned_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused o_mixed=o+V@z -> inv_rope -> group-major FP8 + TMA-aligned UE8M0 scales."""
    T, H, D = o.shape
    R = v_bf16.shape[1]
    block = 128
    hpg = H // n_groups
    out = torch.empty((n_groups, T, hpg * D), dtype=torch.float8_e4m3fn, device=o.device)
    if tma_aligned_size is None:
        aligned = ((T + 3) // 4) * 4
    else:
        aligned = tma_aligned_size
    scale = torch.empty((n_groups, hpg, aligned), dtype=torch.int32, device=o.device)
    if aligned > 0:
        _vha_v_add_inv_rope_fp8_quant_kernel[(aligned, D // block)](
            o,
            z,
            v_bf16,
            positions,
            cos_sin_cache,
            out,
            scale.view(torch.uint8),
            T,
            o.stride(0),
            o.stride(1),
            z.stride(0),
            z.stride(1),
            cos_sin_cache.stride(0),
            out.stride(0),
            scale.stride(0),
            scale.stride(1),
            H=H,
            R=R,
            HPG=hpg,
            HEAD_DIM=D,
            NOPE_DIM=nope_dim,
            HALF_ROPE=rope_dim // 2,
            BLOCK=block,
            FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
            num_warps=4,
        )
    return out.transpose(0, 1), scale.permute(2, 0, 1)[:T]


@triton.jit
def _sigmoid_mul_inplace_kernel(z_ptr, g_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    z = tl.load(z_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    tl.store(z_ptr + offs, (z * s).to(tl.bfloat16), mask=m)


def sigmoid_mul_inplace(z: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """In-place `z *= sigmoid(gate)` in a single pass (zero temp allocation)."""
    if (
        z.is_cuda
        and gate.is_cuda
        and z.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and z.is_contiguous()
        and gate.is_contiguous()
        and z.numel() == gate.numel()
        and z.numel() > 0
    ):
        n = z.numel()
        _sigmoid_mul_inplace_kernel[(triton.cdiv(n, 1024),)](
            z, gate, n, BLOCK=1024, num_warps=4
        )
        return z
    return z * torch.sigmoid(gate)
