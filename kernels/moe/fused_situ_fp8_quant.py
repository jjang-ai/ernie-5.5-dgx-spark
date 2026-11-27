# SPDX-License-Identifier: Apache-2.0
"""Fused SiTU-and-Mul + DeepGEMM packed UE8M0 FP8 quantization kernel.

In vLLM 0.32.0, ``DeepGemmFP4Experts._act_mul_quant`` has a fused kernel for
SiLU, but for ERNIE 5.5's SiTU activation it falls back to:
  1. ``act_out = torch.empty((M_sum, N), dtype=bf16)``
  2. ``situ_and_mul(act_out, mm1_out, beta, linear_beta)``
  3. ``per_token_group_quant_fp8_packed_for_deepgemm(act_out, 128, use_ue8m0=True)``

Because ``M_sum`` is 128-row-padded per active expert, the unfused path writes
and re-reads tens of megabytes of BF16 intermediates per layer. This single-pass
Triton kernel computes ``situ_and_mul`` in FP32 registers and directly quantizes
each 128-element group to FP8 E4M3 + packed UE8M0 int32 scales (TMA-aligned
MN-major layout matching DeepGEMM's ``per_token_group_quant_fp8_packed_for_deepgemm``),
optionally skipping padding rows where ``expert_ids[row] < 0``.
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
def _fused_situ_mul_fp8_quant_packed_kernel(
    inp_ptr,          # [M_sum, 2 * N] bf16
    out_q_ptr,        # [M_sum, N] fp8_e4m3fn
    out_s_ptr,        # [M_sum_padded, N // 128 // 4] int32 (MN-major: stride_sm=1, stride_sk=M_pad)
    expert_ids_ptr,   # [M_sum] int32 (optional: <0 means padding row)
    M_sum,
    N: tl.constexpr,
    stride_in_m,
    stride_q_m,
    stride_s_m,
    stride_s_k,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    SKIP_NEG_EXPERT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 128
):
    row = tl.program_id(0)
    k_pack = tl.program_id(1)  # packs 4 groups of 128 (= 512 cols) into 1 int32 UE8M0

    if SKIP_NEG_EXPERT:
        eid = tl.load(expert_ids_ptr + row)
        if eid < 0:
            tl.store(out_s_ptr + row * stride_s_m + k_pack * stride_s_k, tl.zeros((), dtype=tl.int32))
            return

    inv_beta = 1.0 / BETA
    inv_lbeta = 1.0 / LINEAR_BETA if HAS_LINEAR_BETA else 0.0
    fp8_max = 448.0
    inv_fp8_max = 1.0 / 448.0
    eps = 1e-10

    offs_g = tl.arange(0, GROUP_SIZE)
    packed_sf = tl.zeros((), dtype=tl.int32)

    for sub in tl.static_range(4):
        g_idx = k_pack * 4 + sub
        col = g_idx * GROUP_SIZE + offs_g
        gate = tl.load(inp_ptr + row * stride_in_m + col).to(tl.float32)
        up = tl.load(inp_ptr + row * stride_in_m + N + col).to(tl.float32)

        sig = tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate))
        g_act = BETA * tldevice.tanh(gate * inv_beta) * sig
        if HAS_LINEAR_BETA:
            u_act = LINEAR_BETA * tldevice.tanh(up * inv_lbeta)
            val = g_act * u_act
        else:
            val = g_act * up

        # Round through BF16 to match unfused situ_and_mul -> quant path exactly
        val = val.to(tl.bfloat16).to(tl.float32)

        amax = tl.max(tl.abs(val), axis=0)
        scale_raw = tl.maximum(amax * inv_fp8_max, eps)
        exponent = tl.ceil(tl.log2(scale_raw))
        scale_f32 = tl.math.exp2(exponent)
        exp = tl.minimum(tl.maximum((exponent + 127.0).to(tl.int32), 0), 255)

        q = tl.minimum(tl.maximum(val / scale_f32, -fp8_max), fp8_max).to(tl.float8e4nv)
        tl.store(out_q_ptr + row * stride_q_m + col, q)
        packed_sf = packed_sf | ((exp & 0xFF) << (sub * 8))

    tl.store(out_s_ptr + row * stride_s_m + k_pack * stride_s_k, packed_sf)


def fused_situ_mul_fp8_quant_packed(
    inp: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
    out_q: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU-and-Mul + DeepGEMM packed UE8M0 FP8 quantization.

    Args:
        inp: ``[M_sum, 2 * N]`` BF16 GEMM1 output.
        beta: SiTU gate beta (4.0 for ERNIE 5.5).
        linear_beta: SiTU up soft-clip beta (25.0 for ERNIE 5.5).
        out_q: Optional preallocated ``[M_sum, N]`` float8_e4m3fn buffer.
        expert_ids: Optional ``[M_sum]`` int32 expert IDs; rows with ``< 0`` skipped.
    Returns:
        ``(out_q, out_s_packed)`` matching ``per_token_group_quant_fp8_packed_for_deepgemm``.
    """
    M_sum, two_N = inp.shape
    N = two_N // 2
    assert N % (group_size * 4) == 0, f"N={N} must be divisible by {group_size * 4}"

    if out_q is None:
        out_q = torch.empty((M_sum, N), dtype=torch.float8_e4m3fn, device=inp.device)

    # TMA-aligned MN-major packed int32 scales: [M_pad, N // 512] transposed stride
    m_pad = (M_sum + 3) // 4 * 4
    num_packs = N // (group_size * 4)
    # Allocate [num_packs, m_pad] contiguous, then transpose -> shape [M_sum, num_packs], stride (1, m_pad)
    s_storage = torch.empty((num_packs, m_pad), dtype=torch.int32, device=inp.device)
    out_s = s_storage.t()[:M_sum]

    if M_sum > 0:
        grid = (M_sum, num_packs)
        _fused_situ_mul_fp8_quant_packed_kernel[grid](
            inp,
            out_q,
            out_s,
            expert_ids if expert_ids is not None else inp,
            M_sum,
            N=N,
            stride_in_m=inp.stride(0),
            stride_q_m=out_q.stride(0),
            stride_s_m=out_s.stride(0),
            stride_s_k=out_s.stride(1),
            BETA=float(beta),
            LINEAR_BETA=float(linear_beta) if linear_beta is not None else 1.0,
            HAS_LINEAR_BETA=linear_beta is not None,
            SKIP_NEG_EXPERT=expert_ids is not None,
            GROUP_SIZE=group_size,
            num_warps=4,
        )
    return out_q, out_s
