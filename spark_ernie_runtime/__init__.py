"""DealignAI DGX Spark (GB10, SM12.1) runtime optimizations for ERNIE 5.5 Flash Preview.

Non-destructive overlay for stock `vllm==0.32.0` + `vllm-ernie` that enables:
1. GB10-calibrated FlashInfer SM120 HCA (dsv4) & DSA (dsv3_2) sparse-MLA dispatch.
2. Bit-exact single-kernel fused ERNIE 5.5 split-feature sigmoid + top-k router.
3. GB10 L2-aware mHC fused post+pre+RMSNorm Triton kernels (T=16..32 & T>=512 chunked).
4. Optional `shared_experts` replication (`SPARK_REPLICATE_SHARED_EXPERTS=1`) eliminating 42 ARs/step.
5. DSpark `apply_markov_bias_gathered()` unlocking `dspark_draft_topk=64` (cuts 99% of Markov GEMMs).
"""
from .patch import install

__all__ = ["install"]
