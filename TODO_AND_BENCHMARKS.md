# DGX Spark (GB10, SM12.1) ERNIE 5.5 Flash Preview — Runtime Speedup TODO & Live Benchmarks

**Goal:** Build a clean, all-in-one, shareable **2×–4× DGX Spark (`SM12.1`)** Python overlay package & GitHub repository for `ERNIE-5.5-Flash-Preview` (and `dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`) that maximizes **both single-stream (`B=1`) speed AND max concurrency / max context window** with zero cache duplication and verified prefix caching.

---

## Master Engineering TODO Checklist

### Stage 1 — Deep 2× DGX Spark (`TP=2 + EP=2`) Optimization
- [x] **RDMA Dual-Rail 200 Gb/s CX7 Tuning** (`bench/nccl/RESULTS.md`, `configs/nccl_dual_rail_env.sh`):
  - Proven `4ch, ^LL128, CROSS_NIC=0` jumps NCCL busBW `12.5 → 21.5 GB/s` (`1.72×`), avoiding the naive 2-rail 16ch/LL128 regression trap.
- [x] **MXFP4 MoE Backend Breakthrough** (`bench/moe/RESULTS.md`):
  - Replaced `DEEPGEMM_MXFP4` (`56–127 GB/s`) with **`HummingIndexedExperts`** (`227–242 GB/s`, **`1.83×–3.55×` faster** across `M=1..2048`, avoiding `MXFP8` M=2048 cliff).
  - Built & unit-tested custom **SM12.1 PTX 2-kernel MXFP4 decode** (`156.7 µs` EP=2 / `67.6 µs` EP=4 at `M=1`, `3.55×` vs DeepGEMM, `1.18×` vs Humming).
- [x] **SM12.1 FlashInfer HCA/DSA Dispatch + Fused Router + GB10 mHC** (`bench/mhc_hca/RESULTS.md`, `spark_ernie_runtime/patch.py`):
  - Fixed FlashInfer SM120 `dsv4` (`HCA`, `1.20–1.25×` at `B<=4`) and `dsv3_2` (`DSA`, `3.20–5.87×` at `B=16..64`) crossover thresholds.
  - Bit-exact single-kernel fused ERNIE 5.5 router (`1.56–2.03×`, `max_abs=0.0`).
  - GB10 L2-aware mHC kernels (`1.65×` at `T=32`, `1.46×` at `T=2048` prefill).
- [x] **Live E2E Run 1 (`tp2-ar`, `PIECEWISE`):** Hit **`281.6 tok/s`** at `B=64` on 1 replica (**`3.59×`** vs prior `78.4` max / **`5.18×`** vs `54.4` stock `B=8`), `1,423.7 tok/s` prefill at 10K tokens; diagnosed `PIECEWISE` `B=1` CPU launch overhead (`13.9` vs `22.2 tok/s` under `FULL_AND_PIECEWISE`).
- [x] **Live E2E Run 2 (`FULL_AND_PIECEWISE` `tp2-ar` `B=1..128` + `tp2-dspark`):** Completed! `tp2-ar` hit `21.71` `B=1` / `537.6` peak `B=128`; `tp2-dspark` hit `51.6` eng / `48.8` `algo_off` / `100.2` peak `B=8`.
- [x] **Single-Stream Pack v2 Live Verification on 2× Spark (`tp2-ar 128K` + `tp2-dspark 64K`):**
  - [x] Selective `HummingFP8` safe linear (`222 GB/s` on aligned linears, automatic `DeepGemm` fallback on unaligned `fused_qkv_a_proj` + direct-weight `.wo_a`/`.kv_b_proj`/`.fused_wkv_wgate` + per-`layer` fallback handle).
  - [x] Replicated `shared_experts` (`−42` ARs/step): **huge batch/prefill win** (`+20%` to `+49%` per-request decode at `B=4..128` on AR, `B=4` DSpark jumped to **`23.43 tok/s/req` (`+38%` vs v1)**!), while at `B=1` pure AR the `+528 MiB/tok` DRAM read (`+2.4 ms`) outweighed saving 42 tiny `[1, 4096]` ARs (`−0.8 ms`) → keep `SPARK_REPLICATE_SHARED_EXPERTS=0` for `B=1` single-stream profile and `=1` for batch/prefill profile!
  - [x] DSpark gathered Markov `top-k=64` (`apply_markov_bias_gathered()` with FP32/BF16 cast): live verified on `2×` Spark → **`47.36 tok/s`** (`off_code`), **`58.63 tok/s` e2e** (`high_math`), **`62.33 tok/s` e2e** (`high_zh`, `3.0×` pure AR!), **`23.43 tok/s/req`** at `B=4`, **`14.51 tok/s/req`** at `B=8` (beating AR even at `B=8`!), **`1,595 tok/s`** cold prefill at 32K!
- [x] **DSpark Deep Single-Stream & `christopherowen` (`spark-ds41f` / `dgx-spark-networking`) Speedup Integration:**
  - [x] **Unlocked `enable_adaptive_verification=true`** using ERNIE 5.5's trained `confidence_head` (`model.mtp.confidence_head.projection.weight`) via `FlashInferMLASparseMetadataBuilder.get_varlen_cudagraph_max_query_len()` + flattened Indexer patch.
  - [x] **Integrated `sparknet v0.6.0` (`christopherowen/dgx-spark-networking`, Apache-2.0) One-Shot RoCE Collectives (`10.3 µs` 2× direct / `15.3 µs` 4× switchless ring vs `70–92 µs` NCCL = `6.0×–6.8×` faster!!)** into `spark_ernie_runtime/patch.py`.
  - [x] **Ported `dspark_main_proj_tp`** (`ColumnParallelLinear(..., gather_output=True)`, cuts DSpark draft `main_proj` read `2×–4×`) + **`MALLOC_ARENA_MAX=2` / `MALLOC_MMAP_THRESHOLD_=1048576`** unified-RAM allocator hardening from `spark-ds41f`.
  - [x] **Ported Dinkelbach marginal-rate adaptive verification rule + 512-expert coupon-collector MoE cost curve + bidirectional shared 12-bit lossless packed BF16 TileLang LM head & `markov_w2`**.
  - [x] **Verified Probabilistic Vocab-Parallel DSpark (`draft_tau=1.0`, 64-cand `multimax`) at official `T=1.0, top_p=0.95, reasoning=high`** across `2× Direct`, `2×2 Parallelized`, and `4× Ring`.
  - [x] **Evaluated host OS/kernel/FW upgrades across all 4 Sparks:** Upgraded to DGX OS `7.6.0`, Linux `7.0.0-1025-nvidia` + `kho=off`, BIOS `5.36_0ACUM028` / VBIOS `9A.0B.34.00.00`, ConnectX-7 FW `28.50.1002` (16/16 ports `200000Mb/s`, `MTU 9000`, `22.67 GB/s` busBW; 64 KiB kernel rejected due to out-of-tree UVM requirement).

### KV Cache Minimization, Pooled Zero-Duplication Cache & True Prefix Caching (2× & 4× Spark)
- [x] **Exact KV Cache Footprint Audited & Minimized (3.15× Recovery):**
  - Restoring `--max-num-batched-tokens 2048` cut HCA SWA reservation by `3.15×` → **`643,717` KV tokens from only `13.11 GiB` (`21.3 KiB/token` total across all 43 layers = `4.91×` concurrent `128K` contexts on 2× Spark AR!!)**, and **`269,015` KV tokens from `9.25 GiB` (`4.10×` concurrent `64K` contexts with DSpark!!)**.
  - Caught Linux unified-RAM OOM on 98K (`~86K` tok) single-request indexer scoring at `GPU_UTIL=0.78`; lowered 2× Spark default to `GPU_UTIL=0.75` (`~440K+` KV tokens + extra OS headroom).
- [x] **Zero-Duplication Pooled KV Cache Architecture Audited:**
  - Verified from source (`vllm_ernie/models/ernie5_5/nvidia/attention.py:934-941`) that ERNIE 5.5 explicitly forbids `DCP > 1` (`pcp_size == 1, dcp_size == 1`), so **zero-duplication pooled KV cache across Sparks is achieved via `DP=2+EP=2` / `DP=4+EP=4` (`TP=1`)** (~1.2M+ pooled KV tokens on 2× Spark, ~5M+ pooled KV tokens on 4× Spark!), while `TP=4+EP=4` already gives `~2.7M` KV tokens (`~55 GiB` free KV/Spark at `21.3 KiB/tok`)!
- [x] **True VRAM Prefix Caching Verified Live on BOTH AR & DSpark (100% Exact Recall!):**
  - **`tp2-ar` v2 (`128K`):** 8K (`7.3K` tok): `4.63s → 0.268s` (**`17.3×`**); 16K (`14.5K`): `8.95s → 0.676s` (**`13.2×`**); 32K (`28.9K`): `17.84s → 0.601s` (**`29.7×`**); **64K (`57.7K` tok): `39.18s → 0.488s` (`80.2×` faster TTFT!!, `20.9 tok/s` warm decode at 58K ctx, only `11.1%` KV used, `100%` 12/12 exact secret code recall, `98.8%` warm hit rate in vLLM logs!)**.
  - **`tp2-dspark` v2 (`64K`, `top-k=64`):** 8K (`7.3K`): `5.00s → 0.465s` (**`10.7×`**); 16K (`14.5K`): `9.16s → 0.543s` (**`16.9×`**); **32K (`28.9K`): `18.12s → 0.539s` (`33.6×` faster TTFT!!, `36.8 tok/s` warm decode, `100%` 9/9 exact recall!)**.

### Stage 2 — 4× DGX Spark Switchless Dual-Rail Ring (`TP=4 + EP=4` & `DP=4 + EP=4`)
- [x] **4× Switchless Dual-Rail Ring Verified Live (`node0↔node1↔node2↔node3`, Zero Switch):**
  - `NCCL 2.30.7` bulk ring (`/opt/nccl-switchless-4hca/libnccl.so.2.30.7`): 8 MiB BF16 AR parity `== 10.0` (`NCCL_RING4_BULK_OK`).
  - `sparknet oneshot-ring4` CUDA-graph AR: **`15.3 µs`** `[1, 4096]` BF16 (**`6.0×` vs `~91 µs` NCCL**), **`28.3 µs`** `[8, 4096]`, **`94.8 µs`** `[64, 4096]` (`parity OK`).
- [x] **`TP=4 + EP=4` + DSpark + `sparknet` + Vocab-Parallel Greedy Markov (Single-Stream Rocket):**
  - `41.5 GiB` weights/Spark, **`44.52 GiB` KV cache = `2,114,201` KV tokens (`16.13×` concurrent `128K` contexts!!)**
  - **`off_code` (`B=1`, 384 tok): `93.55 tok/s`** (`89.23` e2e, `0.199s` TTFT!); **`high_zh` (`B=1`): `93.61 tok/s` e2e**; **12-turn growing-context coding: `93–117.1 tok/s`** (`5.86` mean accept len, `69.4%` draft accept, `12/12 + Turn-1` `100%` recall!).
  - **VRAM Prefix Caching up to 98K (`86,572` tok):** `53.1s (1,629 tok/s) → 0.638s` (**`83.2×` faster TTFT!!**), **`85.8 tok/s` warm decode at 86.5K ctx**, `100%` (`12/12`) recall, zero OOM!
  - **Concurrency:** `B=2: 81.8` (`46.8`/req), `B=4: 102.1` (`32.6`/req), `B=8: 160.7` (`26.1`/req), `B=16: 235.3` (`19.4`/req), **`B=32: 275.9` agg / `287.3 tok/s` peak** at only `3.1%` KV cache!
- [x] **2× Spark Final DSpark v5 (`sparknet 10.3 µs` + `vocab_parallel_greedy`) & Wave-5 `1M` Verification:** Completed (see Sections 9 & 10).
- [x] **Harsh Stress Testing & Strict Parity Gates:**
  - Multi-turn long-context stress (`12`-turn growing coding + `8K/16K/32K/64K/98K` prefix cache hit + `100%` receipt recall + zero loops + zero OOM).
  - High-concurrency stress (`B=32` DSpark 4×, `B=64/128` AR 2× — `32/32` & `128/128` clean, zero NCCL/sparknet hangs).
  - Greedy output parity (`math_high`: `48 km/h`, `zh_high`: `48 公里/小时`, `algo_off`: `O(log n)`).

- [x] **Cold Prefill Speedup Push (Wave-3 → Wave-5):**
  - [x] `VHAPostMix` BF16 All-Reduce + fused Triton VHA + `fc2_latent_tp` + cuBLAS `BF16x2` `GateLinear` (`-161 ms`/8K) + fused `vha_v_add_inv_rope_fp8_quant` + L2-tiled `fp8_einsum` (`T_tile=1024`) + 2-chunk HCA Compute/RDMA pipeline (`T>=4096`): lifted 4× Ring cold prefill to **`2,800.3 tok/s` (8K)** & **`2,663.4 tok/s` (64K, `+35%`)**, 2× Direct 32K to **`2,003.5 tok/s` (`2.00×` stock)**, and **`~4,000 tok/s` dual-pair aggregate capacity in `2×2` Parallelized mode**!
- [x] **Concurrency & Multi-Stream Batching Optimization:**
  - [x] SM12.1 Adaptive Verification + Dinkelbach marginal rule + 512-expert coupon-collector MoE cost curve (`m<=64`, `bw=250 GB/s`) + Drafter Padding Mask + `2×2` Parallelized prefix-aware router (`4x-dual-tp2-fast`): `4×` Ring reaches `214–243 tok/s` @ `B=8`, `316–355 tok/s` @ `B=16`, up to `445–497 tok/s` @ `B=32`; `2×` Direct reaches **`168.3 tok/s` @ `B=8` code (`3.09×` stock)**; `2×2` Parallelized scales linearly to **`338.1 tok/s` @ `B=16` code (`2.01×` single 2× pair & `4.31×` stock)**!
- [x] **2× Direct, 4× Ring & 2×2 Parallelized Wave-5 Live Verification (`1M` Context):**
  - [x] Verified `2x-1m-fast` (`1,143,108` KV tok = `1.09×` `1M`), `4x-1m-fast` (`2,712,218` KV tok = `2.59×` `1M`), and `4x-dual-tp2-fast` (`2,286,216` pooled KV tok = `2.18×` `1M`), all passing `8/8 stop`, **`0/8` loops**, `100%` Turn-1 recall at official `T=1.0, top_p=0.95, reasoning=high`.
  - [x] Recorded all verified Wave-5 numbers in `README.md`, `TODO_AND_BENCHMARKS.md`, and `docs/ARCHITECTURE_AND_FLAGS_AUDIT.md`.

### Stage 3 — Shareable All-in-One 2×–4× DGX Spark GitHub Repository
- [x] Clean installable Python package (`pyproject.toml`, `spark_ernie_runtime`, `spark-ernie` CLI, `.gitignore`, zero manual venv hacks).
- [x] One-command launchers & tested profiles for 2× Spark and 4× Spark (`2x-1m-fast`, `2x-1m-max-context`, `2x-max-batch`, `4x-1m-fast`, `4x-1m-max-context`, `4x-dual-tp2-fast`, `4x-dual-tp2-max-batch`).
- [x] Full reproducible benchmark suite (`100K → 1M` ladder, `B=1..64` concurrency, harsh multi-API prefix-cache suite) + polished GitHub `README.md` with exact measured 2×, 2×2, and 4× Spark tables.

---

## Live End-to-End Benchmark Log (`dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`)

| Run | Mode / Stack | `B=1` AR/DSpark | `B=8` | `B=16` | `B=32` | `B=64` | `B=128` | Prefill | KV Used @ Peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock Baseline (2×) | `TP=2+EP=2`, 1-rail, `DEEPGEMM`, `FULL_AND_PIECEWISE` | `18.4` AR / `22.5` DSpark | `54.4` | `78.4` | — | — | — | — | `4.6%` @ B=8 |
| Opt Run 1 (2×) | `TP=2+EP=2` AR, Dual-Rail `4ch,^LL128`, `HUMMING` idx, `PIECEWISE`, v1 overlay | `13.9` *(no FULL CG)* | `69.6` | `112.0` | `179.2` | **`281.6`** (`256.5` e2e) | — | `1,424 tok/s` (10K) | `17.0%` @ B=64 |
| Opt Run 2A (2×) | `TP=2+EP=2` AR, Dual-Rail, `HUMMING` idx, **`FULL_AND_PIECEWISE`**, v1, `MAX_SEQS=128` | **`21.71`** AR (`21.2` e2e) | **`91.7`** | **`138.8`** | **`202.0`** | **`303.0`** (`326` pk) | **`464.6`** (**`537.6`** pk!) | `1,677 tok/s` (3K) | `29.9%` @ B=128 |
| Opt Run 2B (2×) | `TP=2+EP=2` **DSpark**, same + **`FULL_AND_PIECEWISE`**, v1 (`MAX_SEQS=8`) | **`51.6`** eng / **`48.8`** `off` / **`30.0`** prose | **`94.2`** (`100.2` pk!) | — | — | — | — | `1,186 tok/s` (3K) | `3.1%` @ B=8 |
| Opt Run 3A (2×) | **`tp2-ar` v2 (`128K`)**: `HummingFP8` safe + `replicated shared_experts` + `MAX_BATCHED=2048` (`643,717` KV tok!) | `21.6` AR (`23.4` `algo`) | **`111.5`** (`13.94`/req) | **`175.2`** (`10.95`/req) | **`300.8`** (`9.40`/req) | **`356.9` pk** (`6.60`/req) | **`504.7` pk** (`4.69`/req) | **`1,620 tok/s` (29K)** / **`1,474` (58K)**; **80.2× warm TTFT!** | `25.2%` @ B=128 / `11.1%` @ 58K |
| Opt Run 3B (2×) | **`tp2-dspark` v2 (`64K`)**: same + **DSpark gathered Markov `top-k=64`** (`269,015` KV tok!) | **`62.3`** e2e `zh` / **`58.6`** `math` / **`47.4`** `code` / **`30.2`** `prose` | **`116.1`** (`14.51`/req; **`B=4: 23.43`/req = `93.7`!**) | — | — | — | — | **`1,595 tok/s` (29K)**; **33.6× warm TTFT!** | `10.1%` @ 29K |
| Opt Run 4A (2×) | **`tp2-ar` v4 (`128K`) + `sparknet` `10.3 µs` One-Shot RoCE** (`replicate_shared=0`, `538K` KV tok) | **`24.25`** AR (`24.2` code, `0.164s` TTFT!) | **`97.9`** (`100.8` pk) | **`146.7`** (`156.8` pk) | **`223.5`** (`233.5` pk) | **`333.9`** (**`352.0` pk!**) | — | `1,560 tok/s` (7K, `13.7×` warm) | `22.2%` @ B=64 |
| Opt Run 5A (4×) | **`tp4-dspark` v1 (`128K`) + `sparknet` `15.3 µs` Ring4** (`top-k=64`, `2.05M` KV tok!) | `75.8` code / `92.3` `zh` / `52.6` prose | `150.1` (`23.4`/req) | `217.4` (`17.6`/req) | — | — | — | **`1,896 tok/s` (29K)** / **`1,629` (86.5K)**; **83.2× warm TTFT!** | `1.7%` @ B=16 |
| Opt Run 5B (4×) | **`tp4-dspark` v2 (`128K`) + `sparknet` `15.3 µs` Ring4 + Vocab-Parallel Greedy Markov** (`2,114,201` KV tok = `16.13×` 128K!) | **`93.55`** code / **`93.61`** `zh` / **`117.1`** multi-turn / **`57.5`** prose | **`160.7`** (`26.1`/req; `B=4: 102.1`) | **`235.3`** (`19.4`/req) | **`275.9`** (**`287.3` pk!**) | — | — | **`1,898 tok/s` (7K)** / **`1,782` (29K)**; **`32.1×` warm TTFT (`71.6` tok/s)** | **`3.1%` @ B=32!** |
| Opt Run 6 (2×) | **2× Final `2x-1m-fast` Wave-5 (`1M` ctx)** (`1,143K` KV tok = `1.09×` `1M`) | **`51.3`** code / **`77.0`** `zh` / **`56.0` avg (`69.7` pk)** 8-turn `T=1.0` | **`168.3`** code / **`139.4`** prose | — | — | — | — | **`2,003.5 tok/s` (32K, `24.9×` warm!)** | `100%` recall |

### Key Platform Discoveries Logged
- **3.15× HCA SWA KV-Cache Trap Solved:** Setting `--max-num-batched-tokens 8192` caused vLLM's `SlidingWindowSpec` (`window + max_num_batched_tokens`) to reserve `8,320` SWA slots/req across all 37 HCA layers, slashing KV capacity from `272,072` tokens (from `12.07 GiB` at `max_num_batched_tokens=2048`, `44.4 KiB/tok`) down to `126,202` tokens (from `17.67 GiB`, `140 KiB/tok`)! Restoring `--max-num-batched-tokens 2048` recovers **3.15× more KV cache tokens (~350K+ tokens on 2× Spark → 128K–256K max context!)** AND eliminates the unified-RAM OOM spike seen on 11.8K chunked prefill at `GPU_UTIL=0.82`.
- **GB10 Unified-RAM Ceiling:** Keep `--gpu-memory-utilization 0.78` on 121.7 GiB unified-RAM Sparks (`0.82` left only `4.6 GiB` OS/activation RAM and triggered Linux OOM on 8K-chunk prefill).

---

## 9. Final Live-Verified Milestones (1M Native Context + Byte-Packed KV + 12-bit Packed BF16 LM Head)

### 9.1 2× DGX Spark `2x-single-stream` v5 (`tp2-dspark` + `sparknet 10.3 µs` + `vocab_parallel_greedy` + `main_proj_tp`)
- **Weights / KV:** `79.68 GiB`/Spark, `10.04 GiB` KV → **`327,369` KV tokens** (`5.00×` concurrent `64K`, up from `269K` before dropping `draft_logits`!).
- **Single-Stream `B=1`:**
  - `off_code` (384 tok): **`71.89 tok/s`** (`68.65` e2e, `0.253s` TTFT — **`+51.7%`** over prior 2× DSpark `47.4`, **`3.91×`** over stock 2× AR `18.4`!). Mean accept length **`5.16`** (`59.4%` draft accept)!
  - `off_prose`: **`38.98 tok/s`**; `algo_off`: **`46.50 tok/s`** (`0.218s` TTFT); `high_math`: **`54.36` e2e** (`125.1` vis); `high_zh`: **`65.76` e2e** (`141.3` vis).
- **Concurrency (`max_tokens=256`):** `B=1: 35.3`, `B=2: 57.2` (`33.3`/req), `B=4: 74.3` (`22.0`/req), **`B=8: 100.2 tok/s`** (`105.5 tok/s` peak).
- **True VRAM Prefix Caching (`28,899` tok):** `18.28s (1,581 tok/s) → 0.589s` (**`31.0×` faster TTFT**), `44.6 tok/s` warm decode, `3/3 (100%)` exact recall.

### 9.2 Byte-Packed Hybrid KV-Cache Grouping (`_patch_kv_cache_packing`, `SPARK_PATCH_KV_PACK=1`)
- **Root Cause Found:** Stock vLLM `_get_packed_kv_cache_groups()` ran `_approximate_gcd` on layer counts, splitting ERNIE 5.5's `37` tiny HCA compressed layers (`1,728 B`/layer at `bs=256`) into **5 separate groups** of `13.8 KiB` inside `378 KiB` blocks (`95.5%` empty wasted zeros — burning **`5.8+ GiB`** per `1M` request for just `260 MiB` of HCA data!).
- **Fix:** Pack single-page-size buckets to `floor(anchor_bytes / layer_page_bytes)` layers/group, putting all HCA compressed layers into **1 single group** with zero kernel changes and bit-exact KV.
- **Live 4× Spark Ring Verification (`max_model_len=1,048,576`):**
  - Stock vLLM: `16` groups, `14.02 GiB` per `1M` context (`2.91×` concurrent `1M`).
  - **Byte-Packed KV (`SPARK_PATCH_KV_PACK=1`):** **`12` groups** (`64×6, 256×1, 8×5`), **`8.26 GiB` per `1M` context (`1.70×` MORE concurrent `1M` contexts!)** → **`4,898,718` KV tokens = `4.67×` CONCURRENT FULL `1,048,576` (`1M`) CONTEXTS LIVE IN VRAM!**

### 9.3 Lossless 12-bit Packed BF16 TileLang LM Head (`kernels/lm_head/packed_bf16_lm_head.py`, `SPARK_PACKED_BF16_LM_HEAD=1`)
- **Isolated GB10 SM12.1 Live Parity & Speed:**
  - `1e-6` exact FP32 parity vs `torch.mm(..., out_dtype=torch.float32)` + CUDA-graph capture & replay verified (`0.015%` exceptions).
  - **TP=4 (`[50304, 4096]`):** **`1.29×–1.33×` faster** (`1,763 µs → 1,364 µs` FP32 at `M=8`, `235 GB/s` eff BW).
  - **TP=2 (`[100608, 4096]`):** **`1.36×–1.79×` faster** (`4,675 µs → 2,607 µs` at `M=1`; `3,561 µs → 2,615 µs` at `M=8`, saving `1.9–2.1 ms/step`!).
- **Critical Multi-Node Fix:** Raised default `SPARKNET_ROCE_TIMEOUT_MS` from `10000` (10s) → `120000` (120s) so cold TileLang/Triton JIT compilation on one node never trips `__trap()` on peers waiting in `sparknet` collectives.

### 9.4 4× DGX Spark Ring `4x-1m-fast` Final (`1,048,576` Context + Byte-Packed KV + 12-bit Packed BF16 LM Head)
- **Live Server:** `max_model_len=1,048,576`, `41.5 GiB` weights/Spark, `40.76 GiB` KV → **`4,898,718` KV tokens (`4.67×` concurrent `1M` contexts!)**.
- **Single-Stream `B=1`:**
  - **`off_code` (384 tok): `106.86 tok/s` decode (`101.60 tok/s` e2e, `0.186s` TTFT!!)** — **`+14.2%`** jump from 12-bit packed LM head alone (`93.55 → 106.86 tok/s`)!
  - `off_prose`: **`54.39 tok/s`** (`0.231s` TTFT); `algo_off`: **`64.75 tok/s`**; `high_math`: **`80.13` e2e** (`182.1` vis, `0.860s` TTFT); `high_zh`: **`87.41` e2e** (`198.1` vis).
- **Live `115,407`-Token (`128K`!) VRAM Prefix Cache inside `1M` Context:**
  - `82.89s (1,392 tok/s) → 1.073s` (**`77.2×` faster TTFT!!**), **`79.2 tok/s` warm decode at `115K` context**, **`3/3 (100%)` exact recall**!
  - `28,899` tok: `15.78s (1,831 tok/s) → 0.527s` (`29.9×` faster, `3/3` recall); `7,272` tok: `4.33s → 0.388s` (`11.2×` faster, **`103.7 tok/s` warm decode**, `3/3` recall).
- **Concurrency (`max_tokens=256`):** `B=1: 55.3` (`57.7`/req), `B=2: 85.4` (`46.6`/req), `B=4: 107.7` (`34.7`/req), `B=8: 166.8` (`25.6`/req), `B=16: 229.7` (`19.0`/req), **`B=32: 312.0 tok/s`** (`+13.1%` over pre-packed-LM-head `275.9`!).
- **OpenAI API Verification (`spark-ernie test`):** `4/4 PASS` — `/v1/chat/completions` (`reasoning_effort=high`), `/v1/responses` streaming, VRAM prefix caching, and coding decode.
- **200GbE Auto-Discovery (`spark-ernie discover`):** Live-verified on both 2× direct dual-rail and shuffled 4× switchless ring (automatically solved exact physical ring order and all 200GbE MTU 9000 CX7 rails).

---

## 10. Wave-3, Wave-4 & Wave-5 Milestones (`v0.4.0`: `4× Ring`, `2× Direct 1M`, `2×2 Parallelized 1M`, OS/FW Audit & GB10 Roofline)

### 10.1 Host OS, Kernel & Firmware Verification (All 4 Sparks Unthrottled)
- Upgraded and verified across all 4 DGX Sparks:
  - **DGX OS `7.6.0`** + Linux **`7.0.0-1025-nvidia`** + `nvidia-spark-grub-kho` (`kho=off`).
  - **BIOS `5.36_0ACUM028`** / **VBIOS `9A.0B.34.00.00`** (+3–6 GiB unified RAM recovered per Spark).
  - **ConnectX-7 FW `28.50.1002`**, 16/16 ports linked at **`200000Mb/s`**, `MTU 9000`, zero link drops/throttling.
  - Live NCCL dual-rail ring busBW: **`22.67 GB/s`** (`91%` of `25 GB/s` raw 200GbE line rate); `sparknet` one-shot latency: `10.3–20.1 µs` (2×) / `15.3–24.2 µs` (4× ring).

### 10.2 Wave-3 → Wave-5 Kernel Breakthroughs & Roofline Physics
- **GB10 Physical Ceilings Established:**
  - Pure AR `B=1` decode on 4× Sparks (`4 × 273 GB/s`): `~3.57 GB/Spark/tok` → `13.1 ms` DRAM + `2.0 ms` RoCE = **`15.1 ms` (`66 tok/s` physical floor)**. Faster single-stream decode fundamentally requires DSpark + adaptive verification.
  - Cold prefill `T=8192` full sub-layer attribution (`3,225 ms` baseline): `37 × HCA` (`2,456 ms`, dominated by `37 × 128 MiB` VHA BF16 AR = `329 ms` on the `22.67 GB/s` wire, FP32 SIMT `GateLinear` = **`246 ms`**, and `fp8_einsum` `T=8192` L2 cliff = **`99.5 ms`**) + `6 × DSA` (`769 ms`, scattered `2,176` KV rows/tok; byte-packed KV verified **0.0% penalty**: `16.42 ms` vs `16.44 ms`).
- **Wave-3 (`4x_ring_1m_wave3*.json`):** Fused Triton `VHAPostMix` (`4.23×` prefill / `2.2–3.4×` decode) + `fc2_latent_tp` + Dinkelbach marginal rule + `mbt=4096` → `2,486 tok/s` 8K prefill, `445.4 tok/s` prose / `496.6 tok/s` code @ `B=32`, `59.55 avg / 76.53 peak` 8-turn `T=1.0` coding (`0/8` loops).
- **Wave-4 (`4x_ring_1m_wave4*.json`):** Probabilistic Vocab-Parallel DSpark (64-cand `multimax`, `mbt=8192`) → lifted 8-turn `T=1.0` coding to **`77.11 avg / 82.98 peak`** (`+29.5%`), `112.5 tok/s` `high_zh`, `2,481 tok/s` 32K & `2,405 tok/s` 64K cold prefill, `438.6` prose / `479.9` code @ `B=32`.
- **Wave-5 (`4x_ring_1m_wave5*.json`, `2x_direct_1m_wave5*.json`, `2x2_parallel_1m_wave5*.json`):**
  1. **`mhc_sm121(twopass)` unlocked for `T=1..32`** (`1.71–1.81×` faster at `T=1,7,8`, **`-2.9 ms/step`**).
  2. **cuBLAS `BF16x2` (`w_hi + w_lo`) `GateLinear`** (`kernels/moe/gate_bf16x2.py`): `T=8: 1.7–2.1 → 0.8 ms` (`100.00%` top-10 exact); `T=8192: 246.2 → 85.1 ms` (**`-161.1 ms`**, `99.99%` top-10).
  3. **Fused `vha_v_add_inv_rope_fp8_quant` + L2-tiled `fp8_einsum` (`T_tile=1024`) + in-place `sigmoid_mul` + 2-chunk HCA Compute/RDMA pipeline (`T>=4096`)**: `fp8_einsum` `99.5 → 28.0 ms` (TP4) / `198 → 91 ms` (TP2); fused VHA+invRoPE+FP8 `103.3 → 58.8 ms` (TP4) / `189.3 → 94.9 ms` (TP2).
  4. **512-Expert Coupon-Collector MoE Cost Curve (`m<=64`, `bw=250 GB/s`) + `draft_tau=1.0`:** Fixed flat dummy-token startup cost table (`step_ms` prose `51.1 → 39.6 ms`) and `T=1.0` coding acceptance (`20.9% → 29.1%`).
  5. **Bidirectional 12-bit Packed LM Head Sharing + GB10 UMA `posix_fadvise(DONTNEED)` + Upstream vLLM `kv_cache_memory_bytes` `empty_cache()` Fix (`SPARK_PATCH_UMA_FADVISE=1`):** Evicts `3.2–4.6 GiB` of safetensors page cache without root AND fixes upstream `gpu_worker.py:650-672` pinning `~17.1 GiB` of `profile_run()` segments before KV allocation — delivering deterministic **`1,143,108` KV tokens (`1.09×` `1M`)** on every `2×` and `2×2` Spark boot!

### 10.3 Complete Wave-5 Live Verification Summary (`T=1.0, top_p=0.95, reasoning=high`, Zero Loops)
- **`4×` Switchless Ring (`4x-1m-fast`, `2,712,218` KV tok = `2.59×` `1M`):**
  - 8-Turn Coding (`T=1.0, high`): **`79.64 tok/s` avg / `93.51 tok/s` peak** (`+33.7%` vs Wave-3, `8/8 stop`, **`0/8` loops**, `100%` Turn-1 recall).
  - Single-Stream: `code_T1: 70.9` (`+33.8%`), `prose_T1: 52.1`, `zh_T1: 100.7` (`105.58` greedy), `off_prose: 54.60`.
  - Cold Prefill (`3/3` recall): **8K: `2,800.3 tok/s`** (`2.46s` -> `0.334s` warm `7.4×`), **64K: `2,663.4 tok/s`** (`20.52s`, `+35.0%` vs Wave-3 -> `0.517s` warm `39.7×`, `104.9 tok/s` warm dec).
  - Concurrency: Prose `B=8: 214.0 (+13.2%)`, `B=16: 316.0`, `B=32: 379.2` (up to `445.4` with `m<=64` cap); Code `B=8: 214.9`, `B=16: 347.1`, `B=32: 432.4` (up to `496.6`).
- **`2×` Direct (`2x-1m-fast`, `1,083,305–1,143,108` KV tok = `1.03×–1.09×` FULL `1M`!):**
  - 8-Turn Coding (`T=1.0, high`): **`55.97 tok/s` avg / `69.67 tok/s` peak** (`8/8 stop`, **`0/8` loops**, `100%` Turn-1 recall).
  - Single-Stream: `high_zh: 76.95` (`73.61` e2e), `algo_off: 67.78` (`0.181s` TTFT), `high_math: 58.79` (`48 km/h`), `off_code: 51.32` (`49.1` @ T=1), `off_prose: 36.80–41.59`.
  - Cold Prefill (`3/3` recall): **8K: `1,971.0 tok/s`** (`3.49s` -> `0.415s` warm `8.4×`), **32K (`27,341` tok): `2,003.5 tok/s`** (`13.65s`, `+26.7%` vs Wave-1 & **`2.00×` stock!** -> `0.548s` warm `24.9×`, `59.6 tok/s` warm dec).
  - Concurrency: Prose `B=8: 139.35 tok/s` (**`2.56×` stock `54.4`**); Code `B=8: 168.32 tok/s` (**`+68%` vs Wave-1 & `3.09×` stock `54.4`**!).
- **`2×2` Parallelized (`4x-dual-tp2-fast`, `2,286,216` pooled KV tok = `2.18×` `1M`, `:8000` prefix-aware router):**
  - Smoke test through `:8000` router: `4/4 PASS`.
  - 8-Turn Coding (`T=1.0, high`): `8/8 stop`, **`0/8` loops**, `100%` Turn-1 recall (`~50–60` e2e tok/s per turn with sticky prefix affinity).
  - Concurrency (`B=1..16`, `2×` scaling across BOTH independent `TP=2` pairs): Code `B=2: 96.00`, `B=4: 131.28`, `B=8: 210.87`, **`B=16: 338.10 tok/s`** (**`2.01×` single `2×` pair & `4.31×` stock `78.4`**!); Prose `B=2: 72.34`, `B=4: 114.48`, `B=8: 186.43`, `B=16: 199.17+`.
  - Router v0.4.0 polish (unit-tested `<5 ms` first-token SSE latency + exact `['0','1']` concurrent spillover): switched SSE proxy from `resp.read(4096)` to line-by-line `resp.readline()` and tightened spillover to `inflight[preferred] == min_load` so dual-pair `2×32K` parallel cold prefill saturates BOTH pairs (**`~4,000 tok/s` aggregate capacity**).


---

## 11. `v0.5.0` Full `100K → 1M` Context Ladder, `B=1..64` Scaling & Harsh Multi-API Prefix-Cache Verification

### 11.1 `4×` Switchless Ring (`4x-1m-fast`) Full `1,000,031` (`1M`) Context Ladder & `B=64` Scaling
- **Benchmark Scripts & JSONs:** `benchmarks/bench_twitter_1m.py`, `benchmarks/twitter_4x_ring_1m_part1.json`, `benchmarks/twitter_4x_ring_1m_ladder.json`.
- **Single-Stream (`B=1`, DSpark ON, `0` loops):**
  - Easy Code (`T=1.0, top_p=0.95` / `T=0`): **`127.01`** / **`139.79 tok/s`** (`0.144s–0.148s` TTFT).
  - Hard Code (`high, T=1.0`): **`57.61 tok/s`** (`0.152s` TTFT).
  - Prose (`off T=1.0` / `T=0` / `high T=1.0`): **`59.16`** / **`64.16`** / **`57.32 tok/s`** (`0.098s–0.149s` TTFT).
  - Chinese Math (`high, T=1.0`): **`104.77 tok/s`** (`0.134s` TTFT).
- **Concurrency Sweep (`B=1..64`, `T=1.0, top_p=0.95`, DSpark ON, `0` loops across EVERY batch!):**
  - **Easy Code:** `B=1: 89.2 (93.8/req)` · `B=2: 141.1 (76.5/req)` · `B=4: 213.5 (60.9/req)` · `B=8: 309.2 (42.3/req)` · `B=16: 441.4 (31.1/req)` · **`B=32: 611.0 (22.4/req)`** · **`B=64: 853.78 tok/s (15.80/req)`**!
  - **Prose:** `B=1: 54.6 (56.2/req)` · `B=2: 70.0 (37.4/req)` · `B=4: 125.8 (33.6/req)` · `B=8: 210.6 (29.2/req)` · `B=16: 278.3 (19.3/req)` · **`B=32: 415.1 (14.7/req)`** · **`B=64: 604.46 tok/s (10.63/req)`**!
- **Full `100K → 1,000,031` (`1M`) Context Ladder (`20/20 = 100%` start+end needle recall, `0` loops):**
  - `100K`: `2,469 tok/s` (`40.5s`) | warm **`0.729s`** | dec **`104.1–113.5` code / `53.9–55.4` prose**
  - `200K`: `1,826 tok/s` (`54.8s`) | warm **`1.411s`** | dec **`117.4` code / `53.0` prose**
  - `300K`: `1,447 tok/s` (`69.1s`) | warm **`1.890s`** | dec **`103.2–105.2` code / `50.7–51.3` prose**
  - `400K`: `1,175 tok/s` (`85.1s`) | warm **`2.104s`** | dec **`117.3` code / `49.3–54.0` prose**
  - `500K`: `1,009 tok/s` (`99.1s`) | warm **`2.828s`** | dec **`114.5` code / `52.4` prose**
  - `600K`: `874 tok/s` (`114.5s`) | dec **`96.4` code / `48.4` prose**
  - `700K`: warm **`3.497s`** | dec **`88.7` code / `47.6` prose**
  - `800K`: `713 tok/s` (`140.2s`) | dec **`93.3` code / `48.9` prose**
  - `900K`: warm **`4.545s`** | dec **`83.9` code / `42.4` prose**
  - **`1,000K` (`1,000,031` = `1M`!):** `+100K` `552.6 tok/s` (`181.0s`) | warm **`4.985s`** | dec **`80.7` code / `43.0` prose** | `True/True`!
  - Set `4x-1m-fast` default `SPARK_GPU_MEM_UTIL=0.68` so every `4×` Ring boot retains `~38+ GiB` free unified RAM/node for OOM-proof `1M` DSA indexer scratch even at `max_num_seqs=32, mbt=8192`.

### 11.2 `2×` Direct (`2x-1m-fast`) Full Category, `B=1..16` & `100K → 400K` Suite
- **JSON:** `benchmarks/twitter_2x_direct_1m.json`.
- **Single-Stream (`B=1`, DSpark ON, `0` loops):** Easy Code `T=1.0` **`84.43 tok/s`** / `T=0` **`92.84 tok/s`**, Hard Code `high` **`41.34`**, Prose `off T=1.0` **`42.63`** / `T=0` **`43.48`** / `high` **`41.80`**, Chinese Math `high` **`74.47 tok/s`**.
- **Concurrency (`B=1..16`, `T=1.0, top_p=0.95`, `0` loops):**
  - **Easy Code:** `B=1: 61.6 (63.9/req)` · `B=2: 97.8 (52.4/req)` · `B=4: 149.6 (41.8/req)` · `B=8: 230.9 (32.7/req)` · **`B=16: 306.53 tok/s (21.25/req)`** (**`3.91×` stock `78.4`!**).
  - **Prose:** `B=1: 38.4 (39.5/req)` · `B=2: 61.3 (32.0/req)` · `B=4: 93.9 (24.9/req)` · `B=8: 148.5 (20.0/req)` · **`B=16: 167.04 tok/s (11.49/req)`**.
- **Growing Context (`100K → 400K`, `8/8` recall, `0` loops, smooth SWA retention):**
  - `8K`: `1,900 tok/s` (`4.34s` → `0.380s` warm `11.4×`); `32K`: `1,930 tok/s` (`16.90s` → `0.485s` warm `34.8×`).
  - `100K`: `1,748 tok/s` (`57.2s`) | warm **`1.083s`** | dec **`80.5` code / `42.0` prose**
  - `200K`: `1,356 tok/s` (`74.1s`) | warm **`1.566s`** | dec **`72.3` code / `37.9` prose**
  - `300K`: `1,111 tok/s` (`90.4s`) | warm **`1.715s`** | dec **`70.8` code / `37.8` prose**
  - `400K`: `941 tok/s` (`106.6s`) | warm **`2.248s`** | dec **`75.6` code / `37.6` prose**

### 11.3 Harsh Prefix-Cache & Triple-API Verification (`Chat` + `Responses` + `Anthropic /v1/messages`)
- **4 Upstream vLLM 0.32.0 + ERNIE 5.5 Bugs Identified & Fixed (`SPARK_PATCH_ANTHROPIC_COMPAT=1`):**
  1. `prefix_cache_retention_interval` defaulted to `0` → kept only final 256-token SWA block, causing `100K`-token rollbacks on boundary suffixes. Enabled `--prefix-cache-retention-interval 2048` by default.
  2. `--enable-prompt-tokens-details` defaulted off → reported `cached_tokens=0` / `cache_read_input_tokens=0` even on `96%+` hits. Enabled by default.
  3. `AuthenticationMiddleware.verify_token` rejected Anthropic `x-api-key` headers, and `AnthropicServingMessages._detect_merge_inline_system(None)` falsely merged mid-conversation system reminders into message 0 (busting prefix cache) even though `encoding_ernie.py` natively supports inline system messages. Patched cleanly in `patch.py`.
  4. `encoding_ernie.py` raised `ValueError` on `reasoning_effort` in `{"none", "off", "minimal", "max"}` and vLLM defaulted `--anthropic-disabled-thinking-effort` to `auto` (`low`). Patched normalizer in `patch.py` and passed `--anthropic-disabled-thinking-effort none`.
- **Verified `4/4 PASS` on `2×` Direct (`benchmarks/harsh_prefix_apis_2x.json`):**
  - `[1]` Chat SWA 256-token boundary torture (`12,967` tok): `6.532s` cold → **`0.447s–0.526s`** warm (**`95.9%–96.8%`**, `12.4×–14.6×` speedup, `4/4` recall).
  - `[2]` Chat multi-turn `reasoning=high`: **`97.3%` (`0.465s`)** preserved / **`96.0%` (`0.515s`)** stripped.
  - `[3]` OpenAI `/v1/responses`: `6.554s` cold → **`0.422s`** warm (**`96.7%`**, `12,544` cached, **`15.5×`**).
  - `[4]` Anthropic `/v1/messages` + `count_tokens`: `6.547s` cold → **`0.414s`** warm `off` (**`96.8%`**, `12,544` `cache_read`, **`15.8×`**), **`0.767s`** warm `high` + inline `<system-reminder>` (**`96.4%`** cache hit, **`85.63 tok/s`** decode!), `count_tokens=12,955`.
- **Verified `4/4 PASS` on `4×` Ring (`benchmarks/harsh_prefix_apis_4x.json`, with API key auth):**
  - `[1]` Chat SWA 256-token boundary torture (`12,967` tok): `4.649s` cold → **`0.336s–0.366s`** warm (**`95.9%–96.8%`**, `12.7×–13.8×` speedup, `4/4` recall).
  - `[2]` Chat multi-turn `reasoning=high`: **`97.5%` (`0.375s`)** preserved / **`95.9%` (`0.378s`)** stripped.
  - `[3]` OpenAI `/v1/responses`: `4.647s` cold → **`0.352s`** warm (**`96.7%`**, `12,544` cached, **`13.2×`**).
  - `[4]` Anthropic `/v1/messages` + `count_tokens` (`x-api-key` auth): `4.666s` cold → **`0.346s`** warm `off` (**`96.8%`**, `12,544` `cache_read`, **`13.5×`**), **`0.381s`** warm `high` + inline `<system-reminder>` (**`96.4%`** cache hit, **`12.2×` faster**, **`81.45 tok/s`** decode!), `count_tokens=12,955`.
