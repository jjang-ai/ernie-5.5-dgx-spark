# ERNIE 5.5 Flash Preview MXFP4 on DGX Spark (GB10, SM12.1): Runtime, Compilation & Parallelism Audit

**Target Hardware:** 2× and 4× NVIDIA DGX Spark (GB10, `sm_121`, 121.7 GiB unified memory, ~273 GB/s memory bandwidth per node, 200 GbE RoCEv2 ConnectX-7)  
**Target Checkpoint:** `ERNIE-5.5-Flash-Preview` MXFP4 (`Ernie5_5_MoeForCausalLM`, 154.49 GiB total on disk: 43 layers = 37 HCA + 6 DSA; 1 dense MLP + 42 MoE layers with 512 routed experts + 1 shared expert, top-10 routing, latent dim 2048; bundled 3-layer DSpark draft)  
**Audited Software Stack (`spark-node-0` `$VLLM_VENV`):**  
`vllm==0.32.0`, `vllm-ernie==0.1.0`, `torch==2.13.0`, `flashinfer-python==0.7.0.post1`, `humming-kernels==0.1.17`, `tilelang==0.1.12`, `triton==3.7.1`, `tokenspeed-triton==3.9.10.post20261028`, `nvidia-nccl-cu13==2.29.7`, `nvidia-nvshmem-cu13==3.4.5`, `nvidia-cutlass-dsl==4.8.0`.  
*(Not installed in venv: `deep_ep`, `deep_gemm`, `pplx_kernels`, `mori`, `nixl`, `moonep`, `halops`.)*

---

## Executive Summary — Highest-Impact Findings

1. **Why `CompilationMode.NONE` (`mode=0`) ran:** It is **intentional by design**, triggered by `vllm_ernie/models/ernie5_5/patches.py:19-45` adding `Ernie5_5_MoeForCausalLM` to `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`, which causes `vllm/config/vllm.py:812-840` to set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and **unconditionally force `compilation_config.mode = CompilationMode.NONE`**. Passing `-O2`/`-O3` or `--compilation-config '{"mode": 3}'` **will not** enable `torch.compile`/Inductor because:
   - Neither `Ernie5_5ForCausalLM` nor `Ernie5_5Model` carries `@support_torch_compile` (`model.py:337, 775`).
   - Both HCA (`DeepseekV4Attention`, `attention.py:538, 733`) and DSA (`Ernie5_5DSAAttention._indexer_and_attn`, `attention.py:1295`) rely on `@eager_break_during_capture` for piecewise CUDA graphs without Dynamo FX splitting.
   - All 5 audited `PassConfig` fusions (`fuse_allreduce_rms`, `enable_qk_norm_rope_fusion`, `fuse_rope_kvcache_cat_mla`, `fuse_act_padding`, `fuse_qk_norm_rope_kvcache`) are either ROCm-only, Hopper/SM100-only, or dead under `mode=0`. **Do NOT fight breakable CUDA graphs — keep `mode=0` with `FULL_AND_PIECEWISE` breakable graphs, and put effort into custom kernels / parallelism instead.**
2. **Exact SM12.1 Patch Inventory (SHA256 `RECORD` audit):** Across the entire `vllm` + `vllm_ernie` venv, **only 2 serving files** differ from upstream wheels:
   - `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` (+2 lines: pop & pass `sinks` to FlashInfer SM120 decode kernel)
   - `vllm_ernie/models/ernie5_5/nvidia/attention.py` (+24 lines: select `Ernie5_5HCAFlashInferSM120Attention` and `FlashInferMLASparseSM120Backend` when `capability.major == 12`)
3. **4-Spark Scaling (`TP=4+EP=4` vs `DP=4+EP=4` vs `2×(TP=2+EP=2)`):**
   - **ERNIE 5.5 divides cleanly by 4 on every dimension** (`o_groups=8`, `num_attention_heads=64`, `hybrid_mla_num_attention_heads=64`, `n_routed_experts=512`, and `flashinfer-python 0.7.0.post1`'s `_DECODE_DSV4_DISPATCH` natively supports `num_heads ∈ {8, 16, 32, 64, 128}` for all `topk ≥ 1`).
   - **`TP=4 + EP=4` (`DP=1`)** is the **single-stream latency ceiling**: ~38.7 GiB/Spark weights (~41.8 GiB with DSpark), 4× aggregate memory bandwidth (`1,092 GB/s`), ~80 GiB/Spark free for KV cache.
   - **`DP=4 + EP=4` (`TP=1`)** is the **high-concurrency throughput king**: ~43.8 GiB/Spark weights, **0 attention/VHA all-reduces** (down from 86/step; `VHAPostMix` folds locally into `m = I + V@U^T`), and only 42 MoE `all_gatherv + reduce_scatter` pairs in the 2048-dim latent space over RoCEv2.
   - **`PP=2` (`TP=1, EP=1`)** is supported by `Ernie5_5ForCausalLM(SupportsPP)` for batched AR evals (only 1 P2P transfer of `[T, 4, 4096]` BF16 per microbatch, 0 per-layer collectives), but is **incompatible with DSpark** and ~2× slower at `c=1`.
4. **Why DSpark printed `max_num_scheduled_tokens is set to 2048` and loses at `concurrency >= 4`:**
   - `vllm/engine/arg_utils.py:2923-2948` defaults `max_num_batched_tokens=2048` on GB10 (because `device_memory` check or non-H100/B200 fallback sets `OPENAI_API_SERVER: 2048`), triggering the `< 8192` warning in `vllm/config/vllm.py:2275-2283`. Setting `--max-num-batched-tokens 8192` (or `16384`) removes this cap immediately.
   - At `c >= 4`, DSpark with `N=7` verifies `8c` target tokens/step (`32` tokens at `c=4`, expanding unique routed experts/layer from ~38 to ~220 out of 512 on bandwidth-limited 273 GB/s memory), **plus** runs a **3-layer 512-expert MoE draft model (`10.27 GiB`)** for `7c` tokens, **plus** 7 sequential full-vocab (`201,216`) Markov projections + `multimax` passes.
   - **Hidden gem discovered:** The ERNIE 5.5 checkpoint **already includes a trained confidence head** (`model.mtp.confidence_head.projection.weight`), which enables `enable_adaptive_verification=true` once we add a 3-line varlen-CG declaration to `FlashInferMLASparseSM120Backend` (currently blocked only by `AttentionCGSupport.UNIFORM_BATCH` in `flashinfer_mla_sparse.py:258`).
5. **SGLang vs Custom vLLM SM12.1 Runtime Repo:** SGLang has upstream ERNIE 5.5 + DSpark + Mega-mHC recipes targeting SM100/GB300, but lacks out-of-the-box SM12.1 GB10 sparse-MLA + MXFP4 Humming/Marlin wiring. **A clean, versioned `vllm` + `vllm-ernie` SM12.1 overlay/plugin repository is the highest-leverage path** (existing patch delta is <30 LOC).

---

## 1. Audit: `CompilationMode.NONE`, Breakable CUDA Graphs & Triton JIT Warnings

### 1.1 Exact Causal Chain for `CompilationMode.NONE` (`mode=0`)

1. `vllm_ernie/models/ernie5_5/patches.py:19-22, 38-45` registers ERNIE 5.5 into vLLM's breakable CUDA graph allowlist during plugin load:
   ```python
   ERNIE5_BREAKABLE_CUDAGRAPH_ARCHITECTURES = (
       "Ernie5_5MoeMTPModel",
       "Ernie5_5_MoeForCausalLM",
   )
   ```
   and updates `vllm.config.vllm.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` + clears `default_breakable_cudagraph_architectures.cache_clear()`.
2. During `VllmConfig.__post_init__`, `vllm/config/vllm.py:1718` calls `self._maybe_enable_breakable_cudagraph()` (`vllm/config/vllm.py:812-840`):
   ```python
   if ("VLLM_USE_BREAKABLE_CUDAGRAPH" not in os.environ
       and self._uses_breakable_cudagraph_by_default()):
       os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
   ...
   enabled = is_breakable_cudagraph_enabled()
   if enabled:
       self.compilation_config.mode = CompilationMode.NONE
   return enabled
   ```
3. Notice this runs **regardless** of `-O2`, `-O3`, or `--compilation-config '{"mode": 3}'` unless `VLLM_USE_BREAKABLE_CUDAGRAPH=0` is explicitly set in the environment!

### 1.2 Can `-O2` / `-O3` / `mode=3` (`VLLM_COMPILE`) Work on `Ernie5_5_MoeForCausalLM`?

**No — and forcing `VLLM_USE_BREAKABLE_CUDAGRAPH=0` with `mode=3` degrades or breaks ERNIE 5.5.** Specific blockers verified in source:

| Blocker | Exact File & Line | Effect if `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + `mode=3` |
|---|---|---|
| **No `@support_torch_compile` decorator** | `vllm_ernie/models/ernie5_5/nvidia/model.py:337, 775` (`Ernie5_5Model`, `Ernie5_5ForCausalLM`) | Neither class wraps `forward` with `VllmBackend`; Dynamo never captures the model body. (Same as upstream `DeepseekV4ForCausalLM`, which only uses `@support_torch_compile` on CPU `deepseek_v4/cpu/model.py:452`.) |
| **`@eager_break_during_capture` segments** | `vllm_ernie/.../nvidia/attention.py:1295` (`Ernie5_5DSAAttention._indexer_and_attn`), `vllm/models/deepseek_v4/attention.py:538, 733` | Without breakable CUDA graphs, `@eager_break_during_capture` becomes a no-op (`breakable_cudagraph.py:98-99`), and `Ernie5_5Compressor.forward` (`attention.py:422`) + DSA indexer/sparse-MLA fail piecewise stream capture or drop to `CUDAGraphMode.NONE` (`vllm.py:1077-1097`). |
| **TileLang mHC & custom Triton ops** | `vllm_ernie/models/ernie5_5/mhc.py:490-501`, `attention.py:186, 308, 613, 881`, `moe.py:79, 116` | Already hand-fused into TileLang (`mhc_fused_post_pre_tilelang`, `mhc_pre_big_fuse_with_norm_tilelang`) and custom Triton kernels (`_split_sigmoid_sum_kernel`, `_ernie5_5_routing_kernel`, `_merge_swa_topk_kernel`, `_multimax_kernel`), leaving almost zero unfused PyTorch eager ops for Inductor to fuse. |
| **V2 Model Runner & DSpark** | `vllm/config/vllm.py:2976-2981` | DSpark **requires** V2 Model Runner (`MRV1` explicitly lists `"dspark speculative decoding"` and `"adaptive draft verification"` as unsupported). V2 runner pairs natively with breakable CUDA graphs (target: `FULL_AND_PIECEWISE` via `BreakableCUDAGraphCapture`; DSpark draft: `FULL_DECODE_ONLY` via `DFlashCudaGraphManager`, `dflash/speculator.py:141`). |

### 1.3 Audit of `PassConfig` Graph Fusion Passes (`vllm/config/compilation.py`)

All five passes only execute inside `VllmBackend` / Inductor (`mode == CompilationMode.VLLM_COMPILE`). Even if `mode=3` were forced, none of them apply to ERNIE 5.5 on SM12.1:

| `PassConfig` Flag | Gate in `vllm/config/compilation.py` & `vllm/config/vllm.py` | Applies to ERNIE 5.5 on SM12.1? |
|---|---|---|
| `fuse_allreduce_rms` | `vllm.py:171-195`: requires `TP > 1`, `has_flashinfer()`, and `capability == 90 or family(100)` | **No** — gated to SM90/SM100 (excludes SM12.1); also ERNIE 5.5 does not use standard `AllReduce -> RMSNorm` (it uses `mhc_fused_post_pre_tilelang` which fuses residual mixing + Sinkhorn + RMSNorm inside TileLang). |
| `enable_qk_norm_rope_fusion` | `compilation.py:262-269`: CUDA/ROCm/XPU, matches standard `RMSNorm -> RoPE` custom op pattern | **No** — HCA uses custom `fused_q_kv_rmsnorm` (`deepseek_v4/attention.py:25`) + fused inverse-RoPE o_proj; DSA applies RoPE only to the 64-dim rope tail of `q`/`k_pe`. |
| `fuse_rope_kvcache_cat_mla` | `compilation.py:294-299`: CUDA/ROCm, matches standard dense MLA `concat_and_cache_mla` | **No** — ERNIE 5.5 DSA uses `Ernie5_5Indexer` + `fp8_ds_mla` sparse cache, and HCA uses `DeepseekV4SWACache` + `Ernie5_5Compressor`. |
| `fuse_act_padding` | `compilation.py:270-275`: `if not current_platform.is_rocm(): False` | **No** — ROCm-only (and gpt-oss hidden=2880 specific, `vllm.py:222-228`). |
| `fuse_qk_norm_rope_kvcache` | `compilation.py:288-293`: `if not current_platform.is_rocm(): False` | **No** — ROCm/AITER-only (`vllm.py:238-240`). |

**Conclusion on Compilation:** `-O2` (vLLM default) + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (`mode=0`, `cudagraph_mode=FULL_AND_PIECEWISE`) is the **correct and optimal** compilation configuration for ERNIE 5.5 on vLLM 0.32.0. Any further kernel-level fusion must be written directly in Triton/CuTeDSL/TileLang (e.g., fusing `VHAPostMix` + `o_proj` for `TP > 1`, or fusing `Ernie5_5MLP` `SituAndMul` + FP8 quant).

### 1.4 Runtime Triton JIT Compilation Warnings (`_count_expert_num_tokens` & `_topk_log_softmax_kernel`)

vLLM 0.32.0 activates `vllm/utils/jit_monitor.py:52-85` after startup warmup to flag any Triton kernel compiled during live inference:

1. **`_count_expert_num_tokens`**
   - **Location:** `vllm/model_executor/layers/fused_moe/utils.py:168-235` (`count_expert_num_tokens`).
   - **Caller:** `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:538` inside `deepgemm_moe_permute()` (called by `deep_gemm_moe.py:374, 637` when `expert_tokens_meta.expert_num_tokens is None`).
   - **Why it JITs at runtime:** `BLOCK_SIZE = triton.next_power_of_2(min(topk_ids.numel(), 1024))` (`utils.py:200-201`) is a `tl.constexpr`, so every unseen power-of-2 bucket of `num_tokens * top_k` (`16, 32, 64, 128, 256, 512, 1024`) AND `HAS_EXPERT_MAP ∈ {False, True}` triggers a fresh Triton specialization!
   - **Fix (warmup registration):** Wrap `_count_expert_num_tokens` with `VllmTritonJitKernel` (following `vllm/model_executor/warmup/jit_warmup_triton_helper.py` / `vllm/v1/attention/backends/mla/compressor_utils.py:30-45`) or invoke a lightweight warmup loop in `vllm/model_executor/warmup/kernel_warmup.py` (around line 206) over `BLOCK_SIZE ∈ (16, 32, 64, 128, 256, 512, 1024)` × `HAS_EXPERT_MAP ∈ (ep_size > 1,)` with dummy `int32` `topk_ids` and `num_local_experts`.
2. **`_topk_log_softmax_kernel`**
   - **Location:** `vllm/v1/worker/gpu/sample/logprob.py:17-59` (launched by `compute_token_logprobs` at `logprob.py:84-108`).
   - **Callers:** `vllm/v1/worker/gpu/sample/logprob.py:133, 170` (`compute_topk_scores`) and `vllm/v1/worker/gpu/sample/prompt_logprob.py:112` — triggered **the first time any client requests `logprobs` or `prompt_logprobs`** (startup dummy runs don't request logprobs!).
   - **Why it JITs at runtime:** `TOPK_BLOCK_SIZE = min(triton.next_power_of_2(num_logprobs), 1024)` (`logprob.py:96`) is a `tl.constexpr`, plus `logits.dtype` (`float32` for ERNIE 5.5 since `config.py:130` sets `head_dtype="float32"`).
   - **Fix (warmup registration):** Register `_topk_log_softmax_kernel` in `vllm/model_executor/warmup/kernel_warmup.py` (or a `register_logprob_warmups()` helper called alongside `register_top_k_top_p_warmups()` in `topk_topp_sampler.py:27`) for `logits.dtype ∈ (torch.float32, torch.bfloat16)` and common `TOPK_BLOCK_SIZE ∈ (1, 2, 4, 8, 16, 32, 64)` with `BLOCK_SIZE=1024`.

---

## 2. Parallelism, Memory Footprint & Serving Flags Audit (2× and 4× DGX Spark)

### 2.1 Exact Checkpoint Weight Breakdown (`ernie55-preview-mxfp4-upstream-release`)

Measured directly from safetensors shard headers (128 shards, 140,457 tensors, 154.49 GiB total):

| Component | Sharded Under TP? | Sharded Under EP? | Sharded Under PP? | Total Size (GiB) |
|---|---|---|---|---:|
| **Routed Experts** (42 layers × 512 MXFP4 experts + e8m0 scales) | By `intermediate_size` (if no EP) | **Yes** (`512 / EP` experts/rank) | **Yes** (`~21 / 42` layers/stage) | **133.88** |
| **Attention** (37 HCA + 6 DSA, FP8 + BF16 indexers) | Mostly (`wq_b`, `q_b`, `kv_b`, `wo_a`, `wo_b`, `o_gate`); `fused_wqa_wkv` & `indexer` replicated | No | **Yes** | **5.01** |
| **Embeddings + LM Head** (`vocab_size=201216`, `H=4096`, untied) | **Yes** (`VocabParallelEmbedding`, `ParallelLMHead`) | No | Split across first/last stage | **3.07** |
| **Shared Experts + Dense Layer 0 + Latent Proj** | `shared_experts` + dense MLP sharded; `fc1/fc2_latent_proj` replicated | No | **Yes** | **1.81** |
| **mHC Mixers + Norms + Gate Routers** (FP32/BF16) | Replicated | No | **Yes** | **0.46** |
| **Target Subtotal (Autoregressive / No DSpark)** | — | — | — | **144.23** |
| **Bundled DSpark Draft** (`model.mtp.*`: 3 MoE layers × 512 experts + `markov_head` + `confidence_head`) | Experts by EP; `markov_head` / `main_proj` replicated | **Yes** (3 × 512 experts ≈ 9.56 GiB) | **No** (`PP=1` forced in `dspark/utils.py:54`) | **10.27** |
| **Full Total (Target + DSpark)** | — | — | — | **154.49** |

### 2.2 Attention Head & Expert Divisibility Audit for `TP=4`

Checked against `Ernie5_5Config` (`ernie55_config.json`) and `_check_attn_tp` (`attention.py:130-137`):

| Parameter | Config Value | `TP=1` | `TP=2` | `TP=4` | `TP=8` | Notes |
|---|---:|---:|---:|---:|---:|---|
| `o_groups` | `8` | 8 | 4 | **2** | 1 | Divisible by 4 (`8 % 4 == 0`) |
| `num_attention_heads` (HCA) | `64` | 64 | 32 | **16** | 8 | Divisible by 4 (`64 % 4 == 0`) |
| `hybrid_mla_num_attention_heads` (DSA) | `64` | 64 | 32 | **16** | 8 | Divisible by 4 (`64 % 4 == 0`) |
| `n_routed_experts` | `512` | 512 | 256 | **128** | 64 | Divisible by 4 (`512 % 4 == 0`) |
| `FlashInfer SM120` `_DECODE_DSV4_DISPATCH` | `num_heads <= 128, topk >= 1` | **OK** (`h=64`) | **OK** (`h=32`) | **OK** (`h=16`) | **OK** (`h=8`) | Verified in `flashinfer.mla._sparse_mla_sm120` for all `topk ∈ {64, 128, 132, 136, 144, 192, 256, 2048, 2176}` |
| `Ernie5_5HCAAttentionMixin._fuse_vha_quant` | `tp_size == 1` (`attention.py:462`) | Active (SM100 only) | Off | Off | Off | Uses `VHAPostMix` all-reduce when `TP > 1` |

**Result:** `TP=4 + EP=4` passes every architectural divisibility and FlashInfer SM120 kernel dispatch constraint with zero padding overhead (`16` is a native FlashInfer SM120 head count).

### 2.3 All-to-All Backend Matrix on SM12.1 + 200 GbE RoCEv2

Audited in `vllm/config/parallel.py:44-59, 533-539` and `vllm/distributed/device_communicators/all2all.py`:

| `--all2all-backend` | Status in vLLM 0.32.0 | Installed on Spark? | Works on SM12.1 + RoCEv2? | Details / Recommendation |
|---|---|---|---|---|
| **`allgather_reducescatter`** *(default)* | Supported (`AgRsAll2AllManager`, `all2all.py:67`) | **Built-in (NCCL)** | **YES — Recommended now** | Uses standard NCCL `all_gatherv` + `reduce_scatter` over RoCEv2 (200 GbE). Zero extra deps. Operates on 2048-dim latent MoE vectors (`moe_latent_size=2048`, half of `hidden_size=4096`). |
| `naive`, `pplx` | **Removed** (`parallel.py:533-539`) | N/A | **No** | Auto-falls back to `allgather_reducescatter` with warning. |
| `deepep_low_latency` / `deepep_high_throughput` / `deepep_v2` | Supported (`all2all.py:215-425`) | Not installed (`deep_ep` missing) | **Experimental / High Risk** | Requires building `deep_ep` + NVSHMEM IBGDA over ConnectX-7 RoCEv2 (`nvidia-nvshmem-cu13==3.4.5` is present, but DeepEP kernels target SM90/SM100 and IBGDA on GB10 requires kernel-module/rdma tuning). Worth a Phase-3 R&D benchmark for `DP=4+EP=4` decode latency, NOT blocking Phase 1–2. |
| `nixl_ep` | Supported (`all2all.py:436-655`) | Not installed (`nixl` missing) | Untested | RDMA EP via NIXL; heavier setup than NCCL. |
| `flashinfer_all2allv` / `flashinfer_nvlink_two_sided` / `flashinfer_nvlink_one_sided` | Supported (`all2all.py:662-990`) | FlashInfer installed | **NO (MNNVL only)** | Explicitly requires NVLink / MNNVL (`flashinfer.comm.mnnvl.MnnvlConfig`); DGX Sparks have no NVLink between nodes (only 200 GbE RoCEv2). |
| `mori_high_throughput` / `mori_low_latency` | Supported (`all2all.py:995-1100`) | Not installed | **NO (AMD ROCm only)** | Asserts `on_gfx942() or on_gfx950()` (`all2all.py:1040`). |
| `moonep` | Supported | Not installed | **NO (NVLink only)** | NVLink-only balanced EP (`parallel.py:227`). |

### 2.4 Comprehensive Parallelism Mode Comparison (2× and 4× DGX Spark, 121.7 GiB/Spark)

| Topology | Nodes | Weights / Spark (AR / +DSpark) | Free RAM / Spark (@ `util=0.85`, ~103.4 GiB budget) | Per-Step Collectives (43 Layers) | Aggregate Mem BW | Best Workload | Compatibility / Caveats |
|---|---:|---:|---:|---|---:|---|---|
| **`TP=2 + EP=2`** *(1 replica)* | 2 | **73.6 / 79.1 GiB** | **~29.8 / 24.3 GiB** | **129 all-reduces** (43 VHA + 43 `wo_b` + 43 FFN/MoE) | 546 GB/s | 2-Spark interactive / moderate batch | Works today with AR & DSpark. Baseline. |
| **`2× (TP=2 + EP=2)`** *(2 independent replicas)* | 4 | **73.6 / 79.1 GiB** | **~29.8 / 24.3 GiB** | 129 AR/replica (no cross-pair traffic) | 1,092 GB/s (2×546) | Mixed interactive + concurrent (`c >= 2`) | Zero 4-node collective overhead; fault-isolated; easy load-balancer scaling. |
| **`TP=4 + EP=4`** *(1 replica)* | 4 | **38.7 / 41.8 GiB** | **~64.7 / 61.6 GiB** | **129 4-rank all-reduces** over RoCEv2 | **1,092 GB/s** (1 stream!) | **Lowest single-stream (`c=1`) latency** + huge KV cache (~1M ctx) | **Fully supported** (`o_groups=8`, `heads=64`, `experts=512` all ÷ 4; SM120 FlashInfer supports `h=16`). Watch 4-rank RoCEv2 all-reduce latency (~129 × ~15–25 µs = ~2–3 ms/step) vs ~50% weight-read savings (~5–8 ms/step saved!). |
| **`DP=2 + EP=2`** (`TP=1`) | 2 | **77.3 / 83.0 GiB** | **~26.1 / 20.4 GiB** | **84 latent-2048 AG+RS** (**0** attn/VHA collectives!) | 546 GB/s | 2-Spark batched throughput (`c >= 4`) | **Supported in `vllm_ernie`!** (`moe.py:195` explicitly handles DP-gathered mask; `VHAPostMix` folds locally with 0 comm; `needs_dp_coordinator=True`.) Needs `--enable-expert-parallel -dp 2 -tp 1`. At `c=1`, 1 rank idles. |
| **`DP=4 + EP=4`** (`TP=1`) | 4 | **43.8 / 47.5 GiB** | **~59.6 / 55.9 GiB** | **84 latent-2048 AG+RS** (**0** attn/VHA collectives!) | **1,092 GB/s** | **Maximum 4-Spark batch throughput (`c >= 8–256`)** | Eliminates all 86 attention/VHA all-reduces per step! Each DP rank holds independent KV cache (4× KV capacity). Best mode for high-concurrency evals/serving. Requires external LB or vLLM DP coordinator. |
| **`PP=2`** (`TP=1, EP=1`) | 2 | **~72.1 GiB** *(AR only)* | **~31.3 GiB** | **1 P2P send/recv per microbatch** (`[T,4,4096]` BF16) | 273 GB/s (`c=1`), 546 GB/s (pipelined) | High-concurrency AR batch evals (`c >= 16`) | `Ernie5_5ForCausalLM` inherits `SupportsPP` (`model.py:776`). **Incompatible with DSpark** (`dspark/utils.py:54`). Requires `VLLM_USE_V2_MODEL_RUNNER=1` for async scheduling (`vllm.py:1635, 2970`). Poor `c=1` latency. |

> **Key Architectural Insight on `DP + EP` vs `TP + EP`:**  
> ERNIE 5.5's latent MoE projects `hidden_size=4096 -> moe_latent_size=2048` (`fc1_latent_proj`) **before** expert dispatch, and `2048 -> 4096` (`fc2_latent_proj`) **after** expert combine (`moe.py:335-375`). Under `DP=4 + EP=4 (TP=1)`:
> 1. All 43 `VHAPostMix` layers fold `m = I + V @ U^T` locally (`attention.py:159-170`) — eliminating 43 FP32 all-reduces.
> 2. All 43 `wo_b` projections run locally — eliminating 43 BF16 `[T, 4096]` all-reduces.
> 3. The 42 MoE layers only communicate `[T, 2048]` BF16 tensors (half the byte volume of `[T, 4096]`).
> 4. Note: `vllm/config/vllm.py:1058-1075` disables dynamic speculative decoding (`num_speculative_tokens_per_batch_size`) under `DP > 1`, falling back cleanly to static `num_speculative_tokens`.

### 2.5 Serving Flag Recommendations (`--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`)

- **`--max-num-batched-tokens`**: **Always pass explicitly (`8192` or `16384`)**. Without it, `arg_utils.py:2945-2948` defaults to `2048` for `OPENAI_API_SERVER` on non-H100/B200 GPUs (which also triggers the speculative decoding warning in `vllm.py:2275-2283` and caps chunked prefill speed). Note that `Ernie5_5Model.__init__` (`model.py:349-358`) pre-allocates `indexer_topk_buffer` (`[T, 2048]` int32) and `swa_topk_buffer` (`[T, 2176]` int32) sized to `max_num_batched_tokens`: at `8192` tokens that's only **132 MiB** (at `16384`: 264 MiB), negligible on 121.7 GiB Sparks.
- **`--max-num-seqs`**:
  - Low-latency / DSpark profile: `32` or `64` (keeps CUDA graph capture sizes compact and memory pool lean).
  - High-throughput batch profile: `128` (or `256` on `TP=4`/`DP=4` where 60+ GiB/Spark is free for KV cache).
- **`--gpu-memory-utilization`**:
  - On 2-Spark (`TP=2` or `DP=2`, ~74–83 GiB weights): **`0.85`–`0.88`** (~103–107 GiB budget; unified memory leaves ~14–18 GiB for Linux OS + RoCEv2 RDMA buffers + NCCL). Avoid `>0.90` on GB10 unified memory to prevent OS swap/OOM during CUDA graph capture.
  - On 4-Spark (`TP=4` or `DP=4`, ~39–48 GiB weights): **`0.82`–`0.85`** already gives 55–65 GiB KV cache per Spark.
- **`--kv-cache-dtype fp8`** (canonicalized automatically to `fp8_ds_mla` by `DeepseekV4FlashInferSM120Attention`, `deepseek_v4/attention.py:106-120`) — required by `FlashInferMLASparseSM120Impl` (`flashinfer_mla_sparse_sm120.py:64-68`).

---

## 3. Deep Audit: DSpark Speculative Decoding

### 3.1 Why Startup Printed `max_num_scheduled_tokens is set to 2048`

Exact code path:
1. `vllm/engine/arg_utils.py:2923-2948`: Because GB10's `get_device_total_memory()` is `121.7 GiB` (`< 160 GiB`) — or if the launcher passed `--max-num-batched-tokens 2048` — `OPENAI_API_SERVER` defaults/sets `max_num_batched_tokens = 2048`. *(Wait: if `device_memory >= 70 GiB`, line 2937 gives `8192` unless `max_num_batched_tokens=2048` was explicitly passed or `get_device_total_memory()` fell into `except:` -> `0` on Ray/head initialization!)*
2. `vllm/config/vllm.py:2254-2283` (`_set_max_num_scheduled_tokens`): Copies `max_num_batched_tokens` into `max_num_scheduled_tokens`, then checks:
   ```python
   if self.scheduler_config.max_num_scheduled_tokens < 8192:
       logger.warning_once(
           "max_num_scheduled_tokens is set to %d based on the speculative decoding settings..."
       )
   ```
**Fix:** Pass `--max-num-batched-tokens 8192` (or `16384` for long-context prefill).

### 3.2 Root Cause: Why DSpark Loses to AR at `concurrency >= 4` on DGX Spark

Audited across `vllm_ernie/models/ernie5_5/nvidia/dspark.py` and `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`:

1. **MoE Weight-Bandwidth Explosion During Target Verification (Primary Bottleneck on 273 GB/s GB10):**
   - With `dspark_block_size = 7` (`ernie55_config.json:57`), every target verification step runs **`(1 + 7) × c = 8c` tokens** instead of `c` tokens.
   - ERNIE 5.5 has `512` routed experts with `top_k = 10` (`2048×2048×3` MXFP4 ≈ 6.3 MiB/expert). By coupon-collector math (`512 * (1 - (502/512)^T)`):
     - `c=1`: AR (`T=1`) reads **10 experts/layer**; DSpark verify (`T=8`) reads **~74 experts/layer** — still well inside the latency floor where collective + launch overhead dominates, so ~4–5 accepted tokens yields a big net speedup!
     - `c=4`: AR (`T=4`) reads **~39 experts/layer** (~246 MiB/layer); DSpark verify (`T=32`) reads **~240 experts/layer** (~1.51 GiB/layer — **6.1× more DRAM bytes read!**). On a 273 GB/s memory bus, target verification alone becomes ~4–5× slower than a single AR decode step, completely erasing a ~3.5–4.5× mean acceptance length!
     - `c=8`: AR (`T=8`) reads ~74 experts/layer; DSpark verify (`T=64`) reads **~367 experts/layer** (72% of the entire 134 GiB MoE!).
2. **Heavy 3-Layer 512-Expert MoE Draft Model (`10.27 GiB`):**
   - Unlike lightweight dense EAGLE/MTP heads, ERNIE 5.5's bundled DSpark draft (`dspark.py:114-125`, `config.py:141-155`) runs **3 full `Ernie5_5DecoderLayer` MoE blocks** (`first_k_dense_replace = 0`, 512 MXFP4 routed experts per layer!) over `7c` query tokens, **plus** `context_wkv_proj` over verified tokens. At `c=4` (`28` draft tokens), the draft reads ~217 unique experts × 3 layers ≈ **4.1 GiB** every step!
3. **7 Sequential Full-Vocab Markov Projections (`vocab_size=201,216`):**
   - `DSparkSpeculator._sample_sequential` (`dspark/speculator.py:162-199`) computes `base_logits` (`[c*7, 201216]` FP32 via `lm_head`), then loops `for i in range(7):` running `markov_embed` (`[c, 256]`) -> `markov_bias` (`[c, 256] @ [256, 201216]` FP32, `~206 MiB` weight read **7 times sequentially** = **1.44 GiB/step** on TP=1, 720 MiB/step on TP=2!) -> `postprocess_draft_logits` (`_multimax_kernel` over 201,216 FP32 elements × 7) -> sampling.
   - Note: `dspark_draft_topk` (`dspark/speculator.py:201-249`) would restrict the Markov projection to only top-K candidates (eliminating 99% of the `[256, 201216]` bandwidth!), **but `Ernie5_5DSparkForCausalLM` (`dspark.py`) does NOT yet implement `apply_markov_bias_gathered()`** (only `qwen3_dspark` does). Implementing `apply_markov_bias_gathered` on `Ernie5_5DSparkForCausalLM` is a high-leverage optimization!
4. **Adaptive Verification Is Disabled by Default Despite Trained Confidence Head in Checkpoint:**
   - Verified in checkpoint index: `model.mtp.confidence_head.projection.weight` **exists** in `ernie55-preview-mxfp4-upstream-release`!
   - `dspark.py:157-162` loads it, and `dspark/speculator.py:112-119, 192-199` wires it directly to `draft_token_confidence_probs` when `enable_adaptive_verification=true`.
   - **Blocker on SM12.1 (trivial 3-line fix):** `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:481-502` rejects `FlashInferMLASparseSM120Backend` because its inherited `FlashInferMLASparseMetadataBuilder` declares `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH` (`flashinfer_mla_sparse.py:258`) and lacks `get_varlen_cudagraph_max_query_len()` / `supports_query_lens_mismatch`, even though `FlashInferMLASparseSM120Impl._run_mqa_kernel` (`flashinfer_mla_sparse_sm120.py:234`) executes token-by-token (`query.unsqueeze(1)`, `seq_lens=None`) and is completely agnostic to per-request query lengths!

### 3.3 Concrete DSpark Tuning Knobs

1. **`num_speculative_tokens` tuning (`3`, `5`, `7`):**
   - Verified in `vllm/config/speculative.py:1481-1488` and `vllm_ernie/models/ernie5_5/patches.py:113-125`: `num_speculative_tokens` defaults to `dspark_block_size = 7`, **but can be explicitly overridden to `3` or `5`** (unlike Qwen3-Omni which enforces equality at `speculative.py:178`; ERNIE 5.5 has no such hard check).
   - FlashInfer SM120 check: `get_dspark_swa_index_width(128, N)` rounds `128 + N` up to a multiple of `64` (`compressor_utils.py:21-27`) -> `192` for all `N ∈ {1..64}`, which is **natively supported** in `_DECODE_DSV4_DISPATCH` for all `h ∈ {8, 16, 32, 64}`.
   - **Expected sweet spots:** `N=7` for `c=1` (`TP=4` or `TP=2`); `N=3` or `N=4` for `c=2..4` (cuts target MoE verification bandwidth by ~40% and Markov loop by 57%); **disable DSpark (pure AR)** for `c >= 8` (or enable adaptive verification once patched).
2. **CUDA Graph Capture for DSpark:**
   - Already captured as `CUDAGraphMode.FULL_DECODE_ONLY` (`dflash/speculator.py:140-141, 462-516`) covering both the 3-layer parallel backbone forward AND the sequential Markov sampling loop (`dspark/speculator.py:259-268`), **provided** `cudagraph_mode` has FULL decode graphs enabled (default under `-O2` `FULL_AND_PIECEWISE`).
   - Do **not** set `cudagraph_mode=PIECEWISE` (forces DSpark draft to `CUDAGraphMode.NONE` eager mode, `dflash/speculator.py:139-143`!).
3. **Implement `apply_markov_bias_gathered` + `--speculative-config '{"dspark_draft_topk": 64}'` (Phase 2 runtime patch):**
   - Eliminates `7 × [256, 201216]` full-vocab Markov GEMMs and reduces `_multimax_kernel` work during sequential drafting.
4. **Unlock `enable_adaptive_verification=true` on SM12.1 (Phase 2 runtime patch):**
   - Subclass `FlashInferMLASparseSM120Backend` with varlen-CG support so the checkpoint's trained `confidence_head` dynamically trims verification tokens at `c >= 4` instead of always verifying all `7c` drafts.

---

## 4. Existing Ad-Hoc SM12.1 Patch Inventory vs Upstream + SGLang Feasibility

### 4.1 Complete Diff Inventory on `spark-node-0` (Verified via `RECORD` SHA256)

Exactly **3 files** were modified in `$VLLM_VENV/lib/python3.12/site-packages/`:

| # | File | Delta | Purpose | Production Overlay Verdict |
|---|---|---|---|---|
| **1** | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | **+2 lines** (`L71`: `self.sinks = mla_args.pop("sinks", None)`, `L249`: `sinks=self.sinks`) | Forward ERNIE 5.5's per-head `attn_sink` into `flashinfer_trtllm_batch_decode_with_kv_cache_mla` on SM120/SM121. | **Keep & Clean Up** — also override `supports_sink() -> True` on the SM120 backend class so `attention.py` doesn't need to lie with `has_sink=False`! |
| **2** | `vllm_ernie/models/ernie5_5/nvidia/attention.py` | **+24 lines** (`L56`, `L76-78`, `L573-586`, `L1120-1131`, `L1252-1256`) | Add `Ernie5_5HCAFlashInferSM120Attention` (combining `Ernie5_5HCAAttentionMixin` + `DeepseekV4FlashInferSM120Attention`) and select `FlashInferMLASparseSM120Backend` for DSA when `capability.major == 12`. | **Keep & Clean Up** — replace the `has_sink=capability.major != 12` hack at `L1131` with a proper `Ernie5_5FlashInferMLASparseSM120Backend` subclass that sets `supports_sink() -> True` (and varlen CG support for adaptive verification). |

### 4.2 SGLang Feasibility Assessment

- **What SGLang has:** Upstream SGLang supports `Ernie5_5_MoeForCausalLM`, `DSPARK` speculative decoding (`--speculative-algorithm DSPARK`), `ernie55` reasoning/tool parsers, `--attn-dp-size`, and Mega-mHC kernels — primarily validated on **SM100/GB300** (and Hopper) with NVLink/FP8/DeepEP.
- **Why SGLang is NOT the highest-leverage path on DGX Spark (SM12.1) right now:**
  1. SGLang's SM121 (GB10) support lags vLLM 0.32.0 + `flashinfer-python 0.7.0.post1` on three simultaneous fronts required by ERNIE 5.5: (a) SM120/121 `fp8_ds_mla` sparse MLA with attention sinks for both HCA and DSA, (b) MXFP4 routed experts (`humming-kernels` / Marlin W4A16/W4A8 on SM121), and (c) multi-node RoCEv2 without NVLink.
  2. Our current vLLM 0.32.0 + `vllm-ernie 0.1.0` stack **already boots, passes coherence, and serves ERNIE 5.5 MXFP4 + DSpark on SM12.1** with only **26 lines of SM12.1 glue**!
  3. All biggest remaining speedups on DGX Spark — `TP=4+EP=4`, `DP=4+EP=4`, custom SM12.1 MoE/mHC/VHA kernels, DSpark top-k Markov + adaptive verification, and RoCEv2 collective tuning — can be shipped directly inside our clean `dgx-spark-ernie-runtime` overlay/plugin repo on top of `vllm==0.32.0` + `vllm-ernie==0.1.0` with zero framework migration risk.

---

## 5. Recommended vLLM Launch Matrices (DGX Spark)

All commands assume the shared RoCEv2 NCCL env vars (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA` / `UCX_NET_DEVICES` for ConnectX-7) and `VLLM_USE_V2_MODEL_RUNNER=1` (auto-selected when DSpark is enabled; explicitly set for PP+async).

### Profile A: 4-Spark Single-Stream Speed Demon (`TP=4 + EP=4`, DSpark `N=7`)
*Goal: Lowest `c=1..2` latency (~1,092 GB/s aggregate BW, ~41.8 GiB/Spark weights, ~61 GiB/Spark KV).*
```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve $HOME/models/ernie55-preview-mxfp4-upstream-release \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7}' \
  --reasoning-parser ernie5 --enable-auto-tool-choice --tool-call-parser ernie5
```
*(For `c=2..4` interactive users, test `"num_speculative_tokens": 3` or `5` to avoid MoE verification bandwidth saturation.)*

### Profile B: 4-Spark Maximum Batch Throughput (`DP=4 + EP=4`, `TP=1`, Pure AR)
*Goal: Highest aggregate tok/s at `c=8..256` (0 attention/VHA all-reduces, ~43.8 GiB/Spark weights, ~59 GiB/Spark KV).*
```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve $HOME/models/ernie55-preview-mxfp4-upstream-release \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --reasoning-parser ernie5 --enable-auto-tool-choice --tool-call-parser ernie5
```
*(Multi-node DP flags: add `--data-parallel-size-local 1 --data-parallel-address <head_ip> --data-parallel-rpc-port 13345` per Spark node, or `--headless` on worker nodes 1–3.)*

### Profile C: 2-Spark Balanced Interactive (`TP=2 + EP=2`, DSpark `N=5` or `N=7`) / 2× Replicas on 4 Sparks
*Goal: Best 2-Spark interactive latency (~79.1 GiB/Spark weights, ~24 GiB/Spark KV).*
```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve $HOME/models/ernie55-preview-mxfp4-upstream-release \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.86 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}' \
  --reasoning-parser ernie5 --enable-auto-tool-choice --tool-call-parser ernie5
```

### Profile D: 2-Spark High-Concurrency Batch (`DP=2 + EP=2` or `PP=2`, Pure AR)
*Goal: Eliminate attention all-reduces on 2 Sparks for `c >= 8` batched evals (`DP=2+EP=2`: ~77.3 GiB/Spark; `PP=2`: ~72.1 GiB/Spark, 1 P2P transfer/microbatch).*
```bash
# Option D1: DP=2 + EP=2 (preferred for mixed prefill+decode)
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve $HOME/models/ernie55-preview-mxfp4-upstream-release \
  --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.86 --max-num-seqs 128 --max-num-batched-tokens 16384

# Option D2: PP=2 (zero per-layer collectives; best for uniform batched evals, NO DSpark)
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve $HOME/models/ernie55-preview-mxfp4-upstream-release \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.86 \
  --max-num-seqs 128 --max-num-batched-tokens 16384
```

---

## 6. Prioritized Implementation Roadmap for `dgx-spark-ernie-runtime`

| Priority | Workstream | Concrete Changes | Expected Impact |
|---|---|---|---|
| **P0** | **Clean Reproducible SM12.1 Overlay Package** | Version the 2 real patches (`flashinfer_mla_sparse_sm120.py` + `vllm_ernie/.../attention.py`) cleanly as a pip-installable plugin/patcher; add `Ernie5_5FlashInferMLASparseSM120Backend` with `supports_sink() -> True` (removing the `has_sink=capability.major != 12` hack). | Reproducible 1-command deploy across `node0–node3` with zero drift. |
| **P0** | **Zero-Code Launch Matrix Sweep** | Benchmark **Profile A (`TP=4+EP=4`)**, **Profile B (`DP=4+EP=4`)**, **Profile D (`DP=2+EP=2`, `PP=2`)**, and `--max-num-batched-tokens 8192/16384` + DSpark `N ∈ {3, 5, 7}` across `c ∈ {1, 2, 4, 8, 16, 32, 64}`. | Immediate **1.4×–2.2×** speedup on `c=1` (`TP=4+EP=4`) and high-`c` throughput (`DP=4+EP=4`, 0 attn collectives). |
| **P1** | **Eliminate Runtime Triton JIT Spikes** | Register `_count_expert_num_tokens` (`fused_moe/utils.py:168`, `BLOCK_SIZE ∈ {16..1024}`) and `_topk_log_softmax_kernel` (`sample/logprob.py:17`, `TOPK_BLOCK_SIZE ∈ {1..64}`, `fp32/bf16`) in `kernel_warmup.py` via `VllmTritonJitKernel`. | Eliminates cold-start latency spikes on first new batch bucket & first `logprobs` request. |
| **P1** | **DSpark SM12.1 Optimization (Adaptive Verify + Top-K Markov)** | (1) Expose varlen-CG support on `Ernie5_5FlashInferMLASparseSM120Backend` to unlock `enable_adaptive_verification=true` using the checkpoint's trained `confidence_head`. (2) Implement `apply_markov_bias_gathered()` on `Ernie5_5DSparkForCausalLM` so `dspark_draft_topk=64` skips 99% of the 7 sequential `[256, 201216]` Markov GEMMs. | Shifts DSpark crossover from `c < 4` up toward `c = 8–16` and cuts single-stream draft overhead by ~1–2 ms/step. |
| **P2** | **SM12.1 Kernel & Collective Optimizations** | (1) Benchmark Humming vs Marlin vs CuTeDSL/b12x MXFP4 MoE kernels on SM12.1 (with `VLLM_HUMMING_MOE_GEMM_TYPE` tuning). (2) Fuse `VHAPostMix` (`TP=1`) + inverse RoPE + quant for SM12.1 (`attention.py:461` currently gates `_fuse_vha_quant` on DeepGEMM TMA scales). (3) Evaluate DeepEP / NVSHMEM IBGDA over RoCEv2 vs NCCL `allgather_reducescatter` for `DP=4+EP=4`. | Further 15–35% kernel + interconnect gains on GB10. |

---

## 7. Wave-2 through Wave-5 SM12.1 Kernel, Prefill, Decode & Roofline Findings (Verified Live)

This section documents all post-audit engineering discoveries, verified SM12.1 kernel replacements, hardware rooflines, and rejected paths established during live profiling on **2× and 4× NVIDIA DGX Spark (`GB10`, `sm_121`, `273 GB/s` unified RAM/node, dual-rail `200 GbE` ConnectX-7 RoCEv2)**.

### 7.1 Hardware Roofline & Full Sub-Layer Attribution (`4×` Ring, `TP=4 + EP=4`)

1. **Autoregressive `B=1` Decode Roofline (`66 tok/s` Physical Ceiling on 4× Sparks):**
   - Each decode token reads ~`3.57 GB/Spark` across 43 layers (10 routed MXFP4 experts/layer across 4 EP ranks + shared experts + FP8 attention + 12-bit packed LM head + norms/routers).
   - At `273 GB/s` peak unified-RAM bandwidth: `3.57 GB / 273 GB/s = 13.1 ms` DRAM + `~2.0 ms` one-shot `sparknet` RoCEv2 collectives = **`15.1 ms/tok` (`66 tok/s` hard physical floor for pure AR)**. Exceeding `66 tok/s` on 4× Sparks fundamentally requires speculative decoding (`DSpark`) + adaptive verification.
2. **Cold Prefill `T=8192` Full Sub-Layer Attribution (`3,225 ms` = `2,540 tok/s` baseline on 4× Ring):**
   - **37 × HCA layers (`66.39 ms/L` = `2,456 ms`):** `mhc1+2 = 7.88 ms`, `inproj = 2.37 ms`, `prep+mla = 5.82 ms`, `vha = 11.88 ms` (`440 ms` total, dominated by `37 × 128 MiB` BF16 all-reduces = `329 ms` on the `22.67 GB/s` 200GbE wire!), `oproj = 10.07 ms` (`373 ms`, incl. a `99.5 ms` DeepGEMM `fp8_einsum` `T=8192` cliff!), `route+fc1 = 7.40 ms` (`274 ms`, of which FP32 SIMT `GateLinear` burned **`246 ms`!**), `moe = 20.96 ms` (`776 ms`).
   - **6 × DSA layers (`128.20 ms/L` = `769 ms`):** `attn = 75.33 ms/L` (`452 ms`), driven by FlashInfer SM120 sparse MLA reading scattered `2,176` KV rows/token across `64K` without Q-tile sharing (`53.0 ms/L` random indices vs `16.4 ms/L` block-correlated). Direct microbenchmark confirmed our byte-packed hybrid KV layout (`SPARK_PATCH_KV_PACK=1`) has **0.0% DSA latency penalty** (`16.42 ms` vs `16.44 ms`).
   - **Wire Limit:** `200 GbE` switchless dual-rail ring achieves `22.67 GB/s` busBW (`91%` of `25 GB/s` raw line rate). `37 × VHA AR` (`329 ms`) + `wo_b AR` (`164 ms`) + `MoE AR` (`279 ms`) = `772 ms` of pure wire time per 8K chunk unless overlapped with compute.

### 7.2 Wave-2 to Wave-5 Verified Optimizations & Root-Cause Fixes

| Optimization / Flag | Root Cause Solved | Measured Impact (SM12.1 GB10) |
|---|---|---|
| **Byte-Packed Hybrid KV Grouping** (`SPARK_PATCH_KV_PACK=1`) | Stock `_get_packed_kv_cache_groups()` split 37 tiny HCA layers (`1,728 B`/layer) across 5 underfilled `378 KiB` blocks (`95.5%` wasted padding). | Slashes per-request `1M` KV footprint **`14.02 → 8.26 GiB` (`1.70×` capacity)** with fewer groups (`16 → 12`), zero DSA penalty, unlocking **full `1,048,576` (`1M`) context on just 2× Sparks (`1.03×` 1M)** and **`2.59×–4.67×` concurrent `1M` on 4× Sparks**. |
| **DSA `max_num_batched_tokens >= 2048` Invariant** | Values `< 2048` break DSA `swa_topk_buffer` indexing; values `> 2048` inflate HCA SWA block reservation (`window + mbt`). | `2×` profiles use `mbt=2048` (preserving `1M` KV capacity); `4×` Ring uses `mbt=8192` (`2.59×` concurrent `1M` still free while maximizing prefill throughput). |
| **Probabilistic Vocab-Parallel DSpark + 64-Cand `multimax`** (`SPARK_DSPARK_VOCAB_PARALLEL=1`, `SPARK_DSPARK_DRAFT_TAU=1.0`) | Prior greedy-only VP drafting bypassed probabilistic drafting at official `T=1.0, top_p=0.95` defaults, and `draft_tau=0.85` over-sharpened `q` relative to `p` (`20.9%` vs `29.1%` acceptance at `T=1.0`). | Restores fast TP-sharded Markov drafting at official `T=1.0, top_p=0.95` (`+33.7%` on real 8-turn high-reasoning coding: `59.55 → 79.64 tok/s` avg / `93.51` peak, **`0/8` loops**). |
| **`mhc_sm121(twopass)` Unlocked for `T=1..32`** (`SPARK_PATCH_MHC=1`) | Original guard gated `twopass` at `16 <= T <= 32`, accidentally skipping `T=1..15` (`B=1` AR `T=1` & `B=1` DSpark verify `T=7, 8`). | `1.71×–1.81×` faster mHC decode (`0.0` max error at `T=1,4,8`; `1e-6` at `T=7`, CG-safe) → **`-2.9 ms/step`** across all 43 layers. |
| **cuBLAS `BF16x2` (`w_hi + w_lo`) `GateLinear`** (`SPARK_PATCH_GATE_BF16X2=1`, `kernels/moe/gate_bf16x2.py`) | PyTorch `allow_tf32=False` + `bf16x3_router_gemm_cutedsl` NVVM failure on `sm_121a` caused all 42 ERNIE 5.5 router GEMMs to run on slow FP32 SIMT cores (`246.2 ms` at `T=8192`). | **`T=8`:** `1.7–2.1 ms → 0.8 ms` (`2.2–2.6×`, `100.00%` top-10 exact). **`T=4096`:** `125.5 → 44.8 ms` (`-80.7 ms`). **`T=8192`:** `246.2 → 85.1 ms` (**`-161.1 ms`**, `99.99%` top-10 parity). |
| **Fused `vha_v_add_inv_rope_fp8_quant` + L2-Tiled `fp8_einsum` + In-Place `sigmoid_mul` + 2-Chunk HCA Pipeline** (`SPARK_PATCH_HCA_OPROJ_FUSED=1`, `SPARK_HCA_PIPELINE=1`) | (a) TP>1 HCA `o + V@z` → inverse RoPE → per-128 UE8M0 FP8 quant launched 3 separate full-tensor passes. (b) DeepGEMM `fp8_einsum` (`wo_a`, 37 layers) hit a `3.53×` L2-thrashing cliff at `T=8192` (`99.5 ms` vs `28.0 ms` tiled at `T_tile=1024`). (c) `vha_ar` & `wo_b_ar` serialized against GPU compute. | **Fused VHA+invRoPE+FP8 (`100%` UE8M0 parity):** `T=8`: `1.5→0.7ms`; `T=8192`: `103.3→58.8ms` (TP4), `189.3→94.9ms` (TP2). **Tiled `fp8_einsum`:** `99.5→28.0ms` (TP4), `198→91ms` (TP2). **2-chunk Compute/RDMA pipeline** (gated at `T >= 4096` so `2×` `mbt=2048` avoids a separate CUDA-stream allocator pool) lifted 4× `8K` cold prefill to **`2,800 tok/s` (`+22.8%`)** and `64K` to **`2,663 tok/s` (`+35.0%`)**, and 2× `32K` to **`2,003.5 tok/s` (`+26.7%`)**! |
| **512-Expert Coupon-Collector MoE Cost Curve** (`SPARK_DSPARK_MOE_COST_CURVE=1`) | Startup `AdaptiveVerificationManager` cost tables ran identical dummy tokens → fake flat `verify_ms[1..8] = [22.8..25.9 ms]` (`+3.1 ms`), blinding Dinkelbach to ERNIE 5.5's real `+22 ms` MoE expert-fanout slope (`512*(1-(502/512)^m)`). | Corrects the verification cost curve for `m <= 64` (`bw_gb_s=250.0`), trimming wasted low-confidence prose draft tails (`step_ms`: `51.1 → 39.6 ms`) without hurting high-batch `B=32` scaling. |
| **Bidirectional 12-Bit Packed LM Head Sharing** (`SPARK_PACKED_BF16_LM_HEAD=1`) | Draft `_compute_draft_logits_local` can run before target `LogitsProcessor._apply_head` during startup profiling, causing a duplicate `618 MB` packed copy + `1.6 GiB` peak spike on `TP=2`. | Both paths now share `_spark_packed_draft_bf16` / `_spark_packed_lm_head` bidirectionally (`0` extra VRAM), securing `1M` context reliability on both `2× Direct` and `2×2 Parallelized` (`4x-dual-tp2-fast`). |
| **GB10 UMA Safetensors Page-Cache Eviction + `kv_cache_memory_bytes` `empty_cache()` Fix** (`SPARK_PATCH_UMA_FADVISE=1`) | (1) On GB10 unified memory, reading `154 GiB` of `.safetensors` leaves `3–5.5 GiB` of clean Linux page cache in unified RAM right when `cudaMemGetInfo()` runs. (2) Upstream `vllm/v1/worker/gpu_worker.py:650-672`'s `kv_cache_memory_bytes` branch runs `profile_run()` **without** `_scoped_allocator_max_split(20)` or `torch.cuda.empty_cache()`, leaving `~14 GiB` of transient activation segments pinned in the CUDA allocator right when KV cache is allocated! | Wraps `Worker.init_device`, `Worker.load_model` (`os.posix_fadvise(..., POSIX_FADV_DONTNEED)` recovering `+3.2–4.6 GiB` without root), and `Worker.determine_available_memory` (adding `_scoped_allocator_max_split(20)` + `gc.collect()` + `torch.cuda.empty_cache()` before KV allocation, preventing `14 GiB` OOM spikes on `tp2-dspark` `1M`). |

### 7.3 `2×2` Parallelized Topology (`4x-dual-tp2-fast` / `4x-dual-tp2-max-batch`)

On 4× DGX Sparks, `spark-ernie` supports **both** 4-node switchless ring (`TP=4 + EP=4`, lowest single-stream latency) **and** `2×2` Parallelized (`2 × (TP=2 + EP=2)` independent replicas behind a built-in prefix-aware OpenAI router on `:8000` -> `:8001` + `:8002`, `spark_ernie_runtime/router.py`):
- **Zero cross-pair RDMA collectives:** Each pair uses direct dual-rail 200GbE `sparknet` (`10.3 µs` AR) with full `1,048,576` (`1M`) context (`1.03×` 1M per pair = **`2.06×` independent `1M` pools**).
- **Prefix-aware sticky routing:** Routes multi-turn conversations and repeated long prefixes to the replica holding their warm VRAM KV cache while load-balancing concurrent requests across both pairs (`~2×` aggregate concurrency & parallel cold prefill scaling, **`~3,900–4,000 tok/s` aggregate cold prefill**!).
- **Memory budget tuning for `tp2-dspark` `1M`:** Uses `SPARK_KV_CACHE_MEMORY_BYTES=9663676416` (`9.0 GiB` -> deterministic **`1,143,108` KV tokens = `1.09×` `1M` ctx** on every Spark regardless of rank-0 `EngineCore`/`APIServer` host-RAM offset), `SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE=64`, `SPARK_MAX_BATCHED_TOKENS=2048`, `SPARK_HCA_PIPE_MIN_T=4096`, `VLLM_ENGINE_READY_TIMEOUT_S=1800` (cold NVMe load of `79.5 GiB` + Humming repack takes `~8–10 min`, brushing up against vLLM's `600s` default), `SPARK_PATCH_UMA_FADVISE=1`, and bidirectional 12-bit packed LM head sharing.

### 7.4 Rejected / Abandoned Approaches (Do Not Revisit Without Hardware/Driver Changes)

- **Mixing `HummingMoE W4A16` (decode) + `W4A8` (prefill) on one weight tensor:** Incompatible packed layouts; keeping two copies exceeds GB10 unified RAM on `TP=2`.
- **`HummingFP8 W8A8` on SM12.1:** Register-pressure limited on GB10; keep selective `HummingFP8 W8A16` at small `M` with automatic `DeepGEMM W8A8` crossover at `M >= 512`.
- **`bf16x3_router_gemm_cutedsl` on `sm_121a`:** Fails NVVM compilation (`NVVM_ERROR_COMPILATION: unsupported operation`); replaced by cuBLAS `BF16x2` (`w_hi + w_lo`).
- **Naive Triton `bf16x2` router GEMM (`226 ms` at 8K):** 2.65× slower than 2× cuBLAS `torch.mm` (`85.1 ms`).
- **64 KiB Linux page-size kernel without out-of-tree UVM patches:** Not viable on stock DGX OS `7.6.0` (`7.0.0-1025-nvidia` + `kho=off` + BIOS `5.36_0ACUM028` / VBIOS `9A.0B.34.00.00` + CX7 FW `28.50.1002` is the verified optimal host stack, recovering `+3–6 GiB` unified RAM and full unthrottled `22.67 GB/s` dual-rail 200GbE).
- **`SPARK_PATCH_FC1_LATENT_TP=1` on `aux_stream`:** Extra synchronization & allocator pool overhead outweighed compute savings on GB10 (`fc2_latent_tp` fused into existing `shared_experts` AR remains enabled and exact).


### 7.5 Triple-API (`Chat`, `Responses`, `Anthropic /v1/messages`) & Hybrid-KV Prefix-Cache Fixes (`v0.5.0`)

Four subtle upstream vLLM `0.32.0` + `vllm-ernie` incompatibilities were discovered during `100K → 1M` ladder testing and Anthropic SDK / Claude Code integration, and fixed cleanly in `scripts/serve_spark_ernie.sh` and `spark_ernie_runtime/patch.py` (`SPARK_PATCH_ANTHROPIC_COMPAT=1`):

1. **Hybrid SWA 256-Token Boundary Rollback (`--prefix-cache-retention-interval 2048`):** Stock vLLM defaults `prefix_cache_retention_interval=0`, discarding earlier sliding-window KV blocks on ERNIE 5.5's 37 HCA layers so any follow-up suffix that diverges inside the final 256-token block rolls back the entire prefix cache. Retaining SWA checkpoints every `2048` tokens (exact multiple of ERNIE 5.5's 256-token block size) guarantees `95.9%–97.3%` prefix cache hits across all boundary offsets (`12.4×–15.8×` faster warm TTFT at 13K, smooth linear warm TTFT across `100K → 1M`).
2. **Accurate Cache Hit Telemetry (`--enable-prompt-tokens-details`):** Populates `prompt_tokens_details.cached_tokens` on OpenAI Chat, `input_tokens_details.cached_tokens` on OpenAI Responses, and `usage.cache_read_input_tokens` on Anthropic `/v1/messages`.
3. **Anthropic `x-api-key` + Inline System Reminder Preservation:** Patches `AuthenticationMiddleware.verify_token` to accept Anthropic's `x-api-key` header alongside `Authorization: Bearer`, and patches `AnthropicServingMessages._detect_merge_inline_system` to return `False` when ERNIE 5.5's Python renderer (`encoding_ernie.py:240`) handles inline system messages natively — preventing Claude Code `<system-reminder>` blocks from mutating message 0 and invalidating the KV cache (`96.4%` verified warm hit!).
4. **Reasoning Effort Normalization + Disabled Thinking (`--anthropic-disabled-thinking-effort none`):** Normalizes `reasoning_effort` values `none`/`off` (`enable_thinking=False`), `minimal` → `low`, and `max`/`xhigh` → `high` before ERNIE 5.5's renderer so Anthropic `thinking: {"type": "disabled"}` and OpenAI Responses `reasoning: {"effort": "none"|"minimal"|"max"}` work cleanly without `ValueError`.
