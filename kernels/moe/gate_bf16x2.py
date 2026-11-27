"""GB10 (SM12.1) cuBLAS Tensor-Core BF16x2 MoE Router Gate GEMM.

ERNIE 5.5 uses FP32 router weights (`GateLinear(4096 -> 1024, params_dtype=fp32,
out_dtype=fp32)`). With PyTorch's default `allow_tf32=False` and `bf16x3_router_gemm`
failing NVVM on `sm_121a`, stock vLLM casts `[T, 4096]` BF16 inputs to FP32 and runs
non-Tensor-Core FP32 SIMT `cublasSgemm`, burning ~246 ms per 8K prefill chunk across
42 MoE layers (~126 ms at 4K, ~1.7-2.1 ms at T=8 decode).

Decomposing the static FP32 router weight into `w_hi = w.bfloat16()` and
`w_lo = (w - w_hi.float()).bfloat16()` gives ~15 bits of mantissa precision on
cuBLAS BF16->FP32 Tensor Cores (measured on GB10 SM12.1 across 42 layers):
- Decode (`T <= 32`): single-term `torch.mm(x, w_hi.t(), out_dtype=fp32)`:
  `1.7-2.1 ms -> 0.8 ms` (`2.2x-2.6x` faster, `100.00%` top-10 routing agreement).
- Prefill (`T > 32`): exact two-term `w_hi + w_lo`:
  - `T = 4096`: `125.5 ms -> 44.8 ms` (`2.80x` faster, `-80.7 ms`, `99.98%` top-10)
  - `T = 8192`: `246.2 ms -> 85.1 ms` (`2.89x` faster, `-161.1 ms`, `99.99%` top-10)
  Optional `SPARK_GATE_BF16_SINGLE=1` uses single-term BF16->FP32 (`31.5 ms` at 8K,
  `7.8x` faster, `-214.7 ms`, `99.0%` top-10).
"""
from __future__ import annotations

import os
import torch


def split_fp32_to_bf16x2(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an FP32 weight [N, K] into transposed contiguous BF16 pair (w_hi_t, w_lo_t)."""
    w_hi = weight.detach().to(torch.bfloat16)
    w_lo = (weight.detach() - w_hi.float()).to(torch.bfloat16)
    # Keep [N, K] contiguous so `.t()` passes cuBLAS TN (column-major B) fast path
    return w_hi.contiguous(), w_lo.contiguous()


def bf16x2_router_gemm(
    x: torch.Tensor,
    w_hi: torch.Tensor,
    w_lo: torch.Tensor,
    *,
    single_term_max_m: int = 32,
) -> torch.Tensor:
    """Compute x @ (w_hi + w_lo).T -> FP32 via cuBLAS BF16->FP32 Tensor Cores."""
    M = x.shape[0]
    out = torch.mm(x, w_hi.t(), out_dtype=torch.float32)
    if M <= single_term_max_m or os.environ.get("SPARK_GATE_BF16_SINGLE", "0") == "1":
        return out
    out.add_(torch.mm(x, w_lo.t(), out_dtype=torch.float32))
    return out
