#!/usr/bin/env python3
"""Unit & numerical parity tests for custom SM12.1 ERNIE 5.5 MoE/FP8 kernels.

Run on a CUDA SM12.x host (e.g. DGX Spark GB10):
    pytest -v kernels/moe/test_moe_kernels.py
or directly:
    python kernels/moe/test_moe_kernels.py
"""

from __future__ import annotations

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import fused_ernie55_router
import fused_situ_fp8_quant
import sm121_fp8_block_gemv
import sm121_mxfp4_moe_decode

FP4_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_mxfp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    lut = FP4_E2M1_LUT.to(w.device)
    lo = (w & 0x0F).long()
    hi = ((w >> 4) & 0x0F).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(w.shape[0], -1)
    scales = torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, dim=-1)
    return vals * scales


def test_fused_router_bit_exact():
    from vllm_ernie.models.ernie5_5.nvidia.moe import ernie5_5_routing, split_sigmoid_sum

    dev = torch.device("cuda")
    E, TOPK = 512, 10
    bias = torch.randn(E, dtype=torch.float32, device=dev) * 0.1
    scale = torch.rand(E, dtype=torch.float32, device=dev) * 2.0 + 1.0

    for M in [1, 7, 16, 64, 513]:
        logits = torch.randn((M, 2 * E), dtype=torch.float32, device=dev)
        dummy_h = torch.empty((M, 1), dtype=torch.bfloat16, device=dev)
        ref_w, ref_ids = ernie5_5_routing(
            dummy_h, split_sigmoid_sum(logits, E), TOPK, True, bias=bias, scale=scale
        )
        for fast in [False, True]:
            w, ids = fused_ernie55_router.fused_ernie5_5_routing(
                logits, TOPK, True, bias=bias, scale=scale, use_fast_argmax=fast
            )
            assert torch.equal(ids, ref_ids), f"ID mismatch at M={M}, fast={fast}"
            assert float((w - ref_w).abs().max().item()) == 0.0, f"Weight mismatch at M={M}"

    # Tie-breaking & padding test
    tie_logits = torch.zeros((8, 2 * E), dtype=torch.float32, device=dev)
    tie_bias = torch.zeros(E, dtype=torch.float32, device=dev)
    is_pad = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.bool, device=dev)
    w, ids = fused_ernie55_router.fused_ernie5_5_routing(
        tie_logits, TOPK, True, bias=tie_bias, scale=scale, is_padding=is_pad, use_fast_argmax=False
    )
    assert (ids[is_pad] == -1).all()
    assert (w[is_pad] == 0.0).all()
    assert torch.equal(ids[0], torch.arange(TOPK, dtype=torch.int32, device=dev))
    print("PASS: test_fused_router_bit_exact")


def test_fused_situ_fp8_quant_parity():
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )

    dev = torch.device("cuda")
    for M_sum in [128, 640, 2048]:
        x = torch.randn((M_sum, 4096), dtype=torch.bfloat16, device=dev) * 0.5
        act_ref = torch.empty((M_sum, 2048), dtype=torch.bfloat16, device=dev)
        torch.ops._C.situ_and_mul(act_ref, x, 4.0, 25.0)
        q_ref, s_ref = per_token_group_quant_fp8_packed_for_deepgemm(
            act_ref, 128, use_ue8m0=True
        )
        q_fus, s_fus = fused_situ_fp8_quant.fused_situ_mul_fp8_quant_packed(
            x, beta=4.0, linear_beta=25.0
        )
        assert torch.equal(s_ref, s_fus), f"Packed UE8M0 scale mismatch at M_sum={M_sum}"
        rel_l2 = float(
            torch.linalg.norm(q_ref.float() - q_fus.float())
            / (torch.linalg.norm(q_ref.float()) + 1e-12)
        )
        assert rel_l2 < 1e-3, f"FP8 quant rel_l2={rel_l2} too high!"
    print("PASS: test_fused_situ_fp8_quant_parity")


def test_sm121_mxfp4_moe_decode_parity():
    dev = torch.device("cuda")
    E_local, H, N, TOPK = 32, 2048, 2048, 10
    w13 = torch.randint(0, 256, (E_local, 2 * N, H // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E_local, H, N // 2), dtype=torch.uint8, device=dev)
    w13_s = torch.randint(118, 123, (E_local, 2 * N, H // 32), dtype=torch.uint8, device=dev)
    w2_s = torch.randint(118, 123, (E_local, H, N // 32), dtype=torch.uint8, device=dev)
    emap = torch.full((512,), -1, dtype=torch.int32, device=dev)
    emap[:E_local] = torch.arange(E_local, dtype=torch.int32, device=dev)

    for M in [1, 2, 4]:
        x = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.5
        tids = torch.randint(0, 64, (M, TOPK), dtype=torch.int32, device=dev)
        tw = torch.rand((M, TOPK), dtype=torch.float32, device=dev)

        # Exact FP32 reference
        ref = torch.zeros((M, H), dtype=torch.float32, device=dev)
        for m in range(M):
            for k in range(TOPK):
                le = int(emap[tids[m, k]].item())
                if le < 0:
                    continue
                W13 = _dequant_mxfp4(w13[le], w13_s[le])
                W2 = _dequant_mxfp4(w2[le], w2_s[le])
                h1 = x[m : m + 1].float() @ W13.t()
                g, u = h1[:, :N], h1[:, N:]
                a = 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))
                ref[m] += tw[m, k] * (a @ W2.t()).squeeze(0)
        ref = ref.to(torch.bfloat16)

        for atom in [False, True]:
            out = sm121_mxfp4_moe_decode.mxfp4_moe_decode_sm121(
                x, w13, w13_s, w2, w2_s, tids, tw, emap,
                beta=4.0, linear_beta=25.0, use_atomic_w2=atom,
            )
            rel_l2 = float(
                torch.linalg.norm(out.float() - ref.float())
                / (torch.linalg.norm(ref.float()) + 1e-12)
            )
            assert rel_l2 < 5e-3, f"MXFP4 MoE decode rel_l2={rel_l2} at M={M}, atom={atom}"
    print("PASS: test_sm121_mxfp4_moe_decode_parity")


def test_sm121_fp8_block_gemv_parity():
    dev = torch.device("cuda")
    for N, K in [(4096, 4096), (2048, 4096), (4096, 2048), (4096, 1024)]:
        wf = torch.randn((N, K), dtype=torch.float32, device=dev) * 0.05
        amax = wf.view(N // 128, 128, K // 128, 128).abs().amax(dim=(1, 3)).clamp(min=1e-6)
        s_inv = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0))).float()
        s_exp = s_inv.repeat_interleave(128, 0).repeat_interleave(128, 1)
        w_fp8 = (wf / s_exp).clamp(-448, 448).to(torch.float8_e4m3fn)
        w_deq = w_fp8.float() * s_exp
        for M in [1, 2, 4, 8]:
            x = torch.randn((M, K), dtype=torch.bfloat16, device=dev) * 0.5
            ref = (x.float() @ w_deq.t()).to(torch.bfloat16)
            out = sm121_fp8_block_gemv.fp8_block128_gemv_sm121(x, w_fp8, s_inv)
            rel_l2 = float(
                torch.linalg.norm(out.float() - ref.float())
                / (torch.linalg.norm(ref.float()) + 1e-12)
            )
            assert rel_l2 < 3e-3, f"FP8 GEMV rel_l2={rel_l2} at ({N},{K}) M={M}"
    print("PASS: test_sm121_fp8_block_gemv_parity")


if __name__ == "__main__":
    test_fused_router_bit_exact()
    test_fused_situ_fp8_quant_parity()
    test_sm121_mxfp4_moe_decode_parity()
    test_sm121_fp8_block_gemv_parity()
    print("ALL TESTS PASSED!")
