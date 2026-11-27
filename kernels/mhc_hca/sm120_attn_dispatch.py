"""GB10 (SM12.1) calibrated FlashInfer Sparse-MLA dispatch for ERNIE 5.5 HCA & DSA.

Root causes discovered on DGX Spark GB10 (48 SMs, SM12.1):
1. HCA (37 layers, `dsv4`, H=64, SWA=128 + C128A compressed KV):
   FlashInfer's SM120 calibration cache (`sparse_mla_sm120_cpb.json`) sets
   `"dsv4|64|128": 0` (keyed on SWA width=128 alone, ignoring extra C128A tokens),
   forcing even B=1..4 decode onto `PREFILL_MG_DUAL` (24.7 us at 8K, 41.0 us at 32K).
   Routing B <= 4 to `sparse_mla_sm120_decode_dsv4` with GB10-optimal `cpb` runs
   in 14.2-14.3 us (1.74x-2.87x faster, saving 0.39-0.99 ms/token across 37 layers)
   with 0.0e+00 bit-exact parity!
2. DSA (6 layers, `dsv3_2`, H=64, topk=2176 = 2048 + 128 SWA):
   `dsv3_2` is completely absent from GB10's `sparse_mla_sm120_cpb.json`, so
   FlashInfer uses its uncalibrated decode-first policy up to T=64.
   Routing T >= 16 to `PREFILL_SWAPAB` (and T=8 to `cpb=8` decode) cuts latency
   from 374 us -> 111 us at B=16 (3.38x, +1.58 ms/6L), 702 us -> 213 us at B=32
   (3.29x, +2.93 ms/6L), and 2523 us -> 428 us at B=64 (5.89x, +12.57 ms/6L)
   with 0.0e+00 bit-exact parity!
"""

from __future__ import annotations

import functools
import torch
import flashinfer.mla._sparse_mla_sm120 as sm120

_SCRATCH_POOL: dict[tuple[torch.device, str], tuple[torch.Tensor, torch.Tensor]] = {}


@functools.lru_cache(maxsize=1)
def _get_cpp_module():
    return sm120.gen_sparse_mla_sm120_module().build_and_load()


def _get_decode_scratch(
    device: torch.device,
    key: str,
    num_tokens: int,
    num_heads: int,
    num_splits: int,
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    buf = _SCRATCH_POOL.get((device, key))
    need_out = num_tokens * num_heads * num_splits * d_v
    need_lse = num_tokens * num_heads * num_splits
    if buf is None or buf[0].numel() < need_out or buf[1].numel() < need_lse:
        mid_out = torch.zeros((need_out,), dtype=torch.bfloat16, device=device)
        mid_lse = torch.full((need_lse,), float("-inf"), dtype=torch.float32, device=device)
        buf = (mid_out, mid_lse)
        _SCRATCH_POOL[(device, key)] = buf
    return (
        buf[0][:need_out].view(num_tokens, num_heads, num_splits, d_v),
        buf[1][:need_lse].view(num_tokens, num_heads, num_splits),
    )


def hca_sparse_mla_sm121(
    q: torch.Tensor,                      # [T, 64, 512] bf16
    swa_kv_cache: torch.Tensor,           # [pages, 64, 584] or [pages, 64, 1, 584] uint8
    swa_indices: torch.Tensor,            # [T, 128] or [T, 1, 128] int32
    swa_lens: torch.Tensor,               # [T] int32
    compressed_kv_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,   # [T, comp_w] or [T, 1, comp_w] int32
    extra_lens: torch.Tensor | None,      # [T] int32
    sinks: torch.Tensor,                  # [64] fp32
    sm_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GB10-calibrated HCA SM120 sparse MLA (dsv4, d_qk=512, fp8_ds_mla + sinks)."""
    T, H, D = q.shape
    if out is None:
        out = torch.empty((T, H, 512), dtype=q.dtype, device=q.device)
    out_lse = torch.empty((T, H), dtype=torch.float32, device=q.device)

    swa_3d = swa_kv_cache.squeeze(2) if swa_kv_cache.dim() == 4 else swa_kv_cache
    comp_3d = (
        compressed_kv_cache.squeeze(2)
        if compressed_kv_cache is not None and compressed_kv_cache.dim() == 4
        else compressed_kv_cache
    )
    swa_idx_2d = swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices
    extra_idx_2d = (
        extra_indices.squeeze(1)
        if extra_indices is not None and extra_indices.dim() == 3
        else extra_indices
    )
    swa_w = swa_idx_2d.shape[-1]
    extra_w = extra_idx_2d.shape[-1] if extra_idx_2d is not None else 0

    # GB10 (48 SMs): DECODE_SPLITK wins strongly for T <= 4; PREFILL_MG_DUAL wins for T >= 8!
    if T <= 4:
        num_splits = sm120._decode_dsv4_num_splits(swa_w, extra_w, sm120._MODEL_TYPE_DSV4)
        mid_out, mid_lse = _get_decode_scratch(q.device, "hca", T, H, num_splits, 512)
        cpb = min(4 if extra_w <= 128 else 8, num_splits)
        sm120.sparse_mla_sm120_decode_dsv4(
            q,
            swa_3d,
            swa_idx_2d,
            mid_out,
            mid_lse,
            out,
            out_lse,
            sm_scale,
            topk_length=swa_lens,
            attn_sink=sinks,
            extra_kv_cache=comp_3d,
            extra_indices=extra_idx_2d,
            extra_topk_length=extra_lens,
            chunks_per_block=cpb,
        )
    else:
        cpp_mod = _get_cpp_module()
        cpp_mod.sparse_mla_sm120_paged_attention(
            q,
            swa_3d,
            swa_idx_2d,
            out,
            out_lse,
            sm_scale,
            sm120._MODEL_TYPE_DSV4,
            int(sm120.KernelVariant.PREFILL_MG_DUAL),
            swa_lens,
            sinks,
            comp_3d,
            extra_idx_2d,
            extra_lens,
        )
    return out


def dsa_sparse_mla_sm121(
    q: torch.Tensor,              # [T, 64, 576] bf16
    kv_cache: torch.Tensor,       # [pages, 64, 656] or [pages, 1, 64, 656] uint8
    indices: torch.Tensor,        # [T, 2176] or [T, 1, 2176] int32
    sinks: torch.Tensor,          # [64] fp32
    sm_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GB10-calibrated DSA SM120 sparse MLA (dsv3_2, d_qk=576, topk=2176, sinks)."""
    T, H, D = q.shape
    if out is None:
        out = torch.empty((T, H, 512), dtype=q.dtype, device=q.device)
    out_lse = torch.empty((T, H), dtype=torch.float32, device=q.device)

    kv_3d = kv_cache.squeeze(1) if kv_cache.dim() == 4 else kv_cache
    idx_2d = indices.squeeze(1) if indices.dim() == 3 else indices
    topk = idx_2d.shape[-1]

    # GB10 (48 SMs, H=64, topk=2176):
    #   T <= 8:  DECODE_SPLITK (with GB10-optimal cpb)
    #   T >= 16: PREFILL_SWAPAB (3.3x-5.9x faster than uncalibrated decode at T=16..64!)
    if T <= 8:
        num_splits = (topk + sm120._BI - 1) // sm120._BI
        mid_out, mid_lse = _get_decode_scratch(q.device, "dsa", T, H, num_splits, 512)
        cpb = None if T <= 2 else (4 if T <= 4 else 8)
        sm120.sparse_mla_sm120_decode_dsv3_2(
            q,
            kv_3d,
            idx_2d,
            mid_out,
            mid_lse,
            out,
            out_lse,
            sm_scale,
            attn_sink=sinks,
            model_type=sm120._MODEL_TYPE_DSV3_2,
            chunks_per_block=cpb,
        )
    else:
        cpp_mod = _get_cpp_module()
        cpp_mod.sparse_mla_sm120_paged_attention(
            q,
            kv_3d,
            idx_2d,
            out,
            out_lse,
            sm_scale,
            sm120._MODEL_TYPE_DSV3_2,
            int(sm120.KernelVariant.PREFILL_SWAPAB),
            None,
            sinks,
            None,
            None,
            None,
        )
    return out
