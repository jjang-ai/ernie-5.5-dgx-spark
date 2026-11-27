# ERNIE 5.5 Flash Preview — mHC & HCA/DSA Kernel Microbenchmarks on DGX Spark (GB10, SM12.1)

Measured on `node3` (`NVIDIA GB10`, Compute Capability `12.1`, PyTorch `2.13.0+cu130`, vLLM `0.32.0`, Triton `3.7.1`, TileLang `0.1.12`, FlashInfer `0.7.0.post1`). All benchmarks run isolated single-layer kernels (no full checkpoint loaded).

---

## 1. GB10 (SM12.1) Hardware Profile (`node3`)

| Property | Measured Value |
|---|---|
| GPU | NVIDIA GB10 (SM 12.1) |
| Streaming Multiprocessors (`multi_processor_count`) | **48 SMs** |
| L2 Cache Size | **24.00 MiB** (`25,165,824` bytes) |
| Unified Memory (`total_memory`) | **121.69 GiB** LPDDR5X |
| Max Threads / SM | 1,536 |
| Registers / SM | 65,536 |
| Shared Memory / SM (Block Default / Block Opt-In) | **100 KB** (48 KB default / **99 KB** opt-in) |
| Programmatic Dependent Launch (`ENABLE_PDL`) | `True` |
| DeepGEMM (`has_deep_gemm`, `is_deep_gemm_supported`) | `True` (`DeepGemmQuantScaleFMT.UE8M0`) |
| FlashInfer SM120 Sparse MLA | `True` (`dsv4`, `dsv3_2`) |
| STREAM Copy BW (4 MB, L2-resident) | **1,000.6 GB/s** (8.4 µs) |
| STREAM Copy BW (16 MB / 64 MB / 256 MB / 512 MB, LPDDR5X) | **226.3 / 218.6 / 217.8 / 223.2 GB/s** (vs 273 GB/s theoretical) |

### Architectural Implications for ERNIE 5.5 on GB10
1. **48 SMs vs 148–170+ SMs (GB300 / RTX 5090):** Upstream heuristics tuned on datacenter Blackwell (`_FUSED_POST_PRE_MAX_TOKENS = 32`, `n_splits = 48`, uncalibrated FlashInfer SM120 `dsv3_2` decode threshold `max_num_tokens = 64`) mismatch GB10's 48-SM, 223 GB/s LPDDR5X balance.
2. **24 MiB L2 Cache Cliff (~4.5× BW ratio: 1,000 GB/s L2 vs 223 GB/s DRAM):**
   - Each mHC site's FP32 `fn` weight is `24 × 16,384 × 4 B = 1.573 MB`. Across 86 call sites (`135 MB`), `fn` is **always DRAM-cold** during decode (`1.573 MB / 223 GB/s = 7.05 µs` theoretical floor).
   - For prefill (`T >= 512`), `residual_cur` (`T × 4 × 4096 × 2 B`) crosses the 24 MiB L2 boundary (`T=128`: 4.2 MB L2-hot; `T=512`: 16.8 MB partial spill; `T=2048`: **67.1 MB** full DRAM spill -> 3× roundtrip across 3 kernels = 304 MB traffic!). Chunking `T` into **256-token L2-resident slices** (`8.4 MB << 24 MiB L2`) keeps `residual_cur` in L2 across post → pre-norm GEMM → epilogue.
3. **99 KB/CTA SMEM limit:** Triton prefill kernels using `num_stages=3` overflow 99 KB (`106,496 B > 101,376 B`) and must use `num_stages=2`.

---

## 2. mHC (Hyper-Connections) Profiling & Optimization (86 Call Sites/Token)

ERNIE 5.5 uses `hc_mult = 4`, `hidden_size = 4096`, `sinkhorn_iters = 20`, RMS factor `1 / (sqrt(mean) + eps)`, with 1× `mhc_pre_broadcast` + **85× `mhc_fused_post_pre_tilelang`** (with fused RMSNorm) + 1× `mhc_post` + 1× `hc_head`.

### 2.1 Upstream Baseline Breakdown (`bench_mhc_sm121.py`)

Measured with rotating 24-bank `fn` (`37.7 MB > 24 MiB L2`) inside CUDA Graphs (`Graph Cold`) and single-bank (`Graph Hot` / `Eager Hot`):

| `T` | Upstream Dispatch Path | Eager Hot | Graph Hot | Graph Cold | 85× Cold | Sub-k1 | Sub-k2 | Sub-k3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `fused(tile_n=2, s=8)` + `pre_big_fuse` | 68.8 µs | 8.2 µs | 11.2 µs | 0.95 ms | 4.1 µs | 6.1 µs | — |
| 2 | `fused(tile_n=2, s=8)` + `pre_big_fuse` | 67.8 µs | 8.8 µs | 11.4 µs | 0.97 ms | 6.1 µs | 6.1 µs | — |
| 4 | `fused(tile_n=2, s=8)` + `pre_big_fuse` | 67.6 µs | 11.3 µs | 11.6 µs | 0.99 ms | 8.2 µs | 6.2 µs | — |
| 8 | `fused(tile_n=2, s=8)` + `pre_big_fuse` | 67.7 µs | 17.9 µs | 13.4 µs | 1.14 ms | 11.1 µs | 6.2 µs | — |
| 16 | `fused(tile_n=6, s=8)` + `pre_big_fuse` | 67.3 µs | 21.0 µs | 20.2 µs | 1.72 ms | 18.1 µs | 6.2 µs | — |
| 32 | `fused(tile_n=6, s=8)` + `pre_big_fuse` | 68.0 µs | 30.8 µs | 28.8 µs | 2.45 ms | 27.4 µs | 6.2 µs | — |
| 64 | 3-kern: `post` + `DeepGEMM(s=48)` + `pre` | 67.3 µs | 15.5 µs | 17.5 µs | 1.49 ms | 6.2 µs | 8.2 µs | 6.2 µs |
| 128 | 3-kern: `post` + `DeepGEMM(s=24)` + `pre` | 66.3 µs | 21.6 µs | 23.1 µs | 1.96 ms | 8.2 µs | 10.2 µs | 8.2 µs |
| 512 | 3-kern: `post` + `DeepGEMM(s=6)` + `pre` | 218.6 µs | 231.6 µs | 232.5 µs | 19.76 ms | 173.7 µs | 32.1 µs | 36.4 µs |
| 2048 | 3-kern: `post` + `DeepGEMM(s=1)` + `pre` | 1369.2 µs | 1302.8 µs | 1312.0 µs | 111.52 ms | 692.7 µs | 251.2 µs | 401.2 µs |

Other entry points (Graph Hot, 1 call/forward):
- `mhc_pre_tilelang`: 12.3 µs (`T=1..64`), 16.4 µs (`T=128`), 54.0 µs (`T=512`), 655.5 µs (`T=2048`)
- `mhc_pre_broadcast_tilelang`: 10.3 µs (`T=1..32`), 12.3 µs (`T=64`), 118.1 µs (`T=512`), 554.7 µs (`T=2048`)
- `mhc_post_tilelang`: 4.1 µs (`T=1..32`), 6.2 µs (`T=64`), 170.9 µs (`T=512`), 692.3 µs (`T=2048`)
- `hc_head_fused_kernel_tilelang`: 8.4 µs (`T=1`), 10.7 µs (`T=32`), 17.9 µs (`T=64`), 106.2 µs (`T=512`), 407.6 µs (`T=2048`)

### 2.2 Root Causes Identified in Upstream mHC
1. **Severe `T=16..32` inversion (`T=32` is 1.65× slower than `T=64`!):** `_MHC_FUSED_TILELANG_KERNEL` launches grid `(T, n_tiles, split_k)` with 1 token per CTA. At `T=32`, it reloads the `1.57 MB` `fn` matrix **32× from L2 (50.3 MB)** AND recomputes `new_r` **4×** (`12×` at `T=1..15`).
2. **Slow prefill epilogue (`_MHC_PRE_BIG_FUSE_TILELANG_KERNEL`):** Uses `threads=96` (3 warps, only 64 threads for collapse) with a 2-pass `output_shared` SMEM stash, taking **445.4 µs** at `T=2048` vs **352.0 µs** (**1.27× faster**) for our single-pass register-resident Triton epilogue (`_mhc_epilogue_norm_kernel`, `num_warps=8`).
3. **L2 Thrashing at `T=2048`:** Running all 2048 tokens per kernel evicts `residual_cur` (`67.1 MB`) from the 24 MiB L2 cache twice (`mhc_post` write -> `DeepGEMM` read -> `pre_big_fuse` read = **304 MB** DRAM traffic).

### 2.3 Verified SM12.1 Optimized Kernel (`kernels/mhc_hca/mhc_sm121.py`)

| `T` | Upstream Cold | `mhc_fused_post_pre_sm121` Cold | Speedup | 85× Saved | Parity `max_abs` (`res/post/comb/li`) | `rel_l2` (`li`) |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 12.1 µs | **11.8 µs** | **1.02×** | +0.02 ms | `0.0e+00 / 1.2e-07 / 4.5e-08 / 0.0e+00` | `0.00e+00` |
| 16 | 19.0 µs | **16.9 µs** | **1.12×** | **+0.18 ms** | `2.0e-03 / 1.2e-07 / 6.0e-08 / 1.2e-04` | `4.77e-07` |
| 32 | 34.4 µs | **20.9 µs** | **1.65×** | **+1.15 ms** | `9.8e-04 / 1.2e-07 / 8.9e-08 / 1.2e-07` | `3.29e-10` |
| 64 | 27.5 µs | **26.9 µs** | **1.02×** | +0.05 ms | `0.0e+00 / 3.6e-07 / 1.5e-07 / 7.8e-03` | `1.57e-05` |
| 512 | 238.9 µs | **235.2 µs** | **1.02×** | **+0.31 ms** | `0.0e+00 / 1.8e-07 / 1.2e-07 / 1.6e-02` | `1.64e-05` |
| 2048 | 1417.9 µs | **972.3 µs** | **1.46×** | **+37.88 ms** | `0.0e+00 / 1.8e-05 / 1.0e-05 / 3.1e-02` | `1.53e-04` |

*(Note: Single-kernel atomic barrier prototype `_mhc_single_kernel_t1_lean` verified bit-exact parity `0.00e+00`, but L2 atomic contention across 48 CTAs yielded 15.0 µs vs 11.8 µs for our 2-kernel PDL design.)*

---

## 3. HCA (37 Layers), DSA (6 Layers) & Lightning Indexer on SM12.1

### 3.1 Critical Architectural & Fallback Findings on SM12.1

1. **DSA SM120 (`dsv3_2`, `H=64, topk=2176`, 6 layers) is UNCALIBRATED on GB10:**
   - `$HOME/.cache/flashinfer/0.7.0.post1/121a/autotune/sparse_mla_sm120_cpb.json` contains calibration constants **only for `dsv4`**, completely omitting `dsv3_2`.
   - Consequently, FlashInfer routes all `T <= 64` DSA batches to its uncalibrated `DECODE_SPLITK` kernel, causing a **5.87× regression at `B=64`** (`2527.0 µs` vs `430.5 µs` on `PREFILL_SWAPAB` -> **+12.58 ms/token wasted across 6 layers**!) and **3.05×–3.20× regressions at `B=16..32`**.
2. **HCA SM120 (`dsv4`, `H=64, SWA=128 + C128A`, 37 layers) Crossover Lookup Bug:**
   - GB10's `sparse_mla_sm120_cpb.json` sets `"dsv4|64|128": 0` (keyed on SWA width=128 alone, ignoring `extra_sparse_indices`), forcing **even `B=1..4` single-user decode** onto `PREFILL_MG_DUAL` (`24.6–41.0 µs`) instead of `sparse_mla_sm120_decode_dsv4` (`20.5–32.8 µs`, or `14.2–14.3 µs` with single-block `cpb`), wasting **+0.15 to +0.30 ms/token** across 37 layers.
3. **MXFP4 Indexer Disabled on SM12.1:**
   - `dsa_indexer_uses_fp4` explicitly rejects SM12x (`requires Blackwell datacenter GPUs sm_10x`). Active SM12.1 Lightning Indexer runs **FP8** (`hadamard_rows` + `per_token_group_quant_fp8` + DeepGEMM `fp8_fp4_paged_mqa_logits` / `fp8_fp4_mqa_logits` + `top_k_per_row`).
4. **`@eager_break_during_capture` on `Ernie5_5DSAAttention._indexer_and_attn`:**
   - Breaks CUDA Graphs 6× per forward pass. Isolated host dispatch overhead for DSA attention alone at `B=1` is **25.0 µs/layer** (`45.7 µs` eager vs `20.7 µs` graphed = **+0.15 ms/token** across 6 layers, plus indexer launch overhead).
5. **Unfused DSA Indexer & Sparse Index Kernels:**
   - `merge_swa_topk` + `triton_convert_req_index_to_global_index` writes and immediately re-reads `[T, 2176] int32` (142 MB traffic at `T=8192`).
   - `hadamard_rows(q)` + `per_token_group_quant_fp8(q)` + `weights * q_scale * weights_scale` writes and immediately re-reads `[T, 64, 128] bf16` (268 MB traffic at `T=8192`).

### 3.2 Baseline Attention & Indexer Latencies (`bench_hca_dsa_sm121.py`)

#### HCA FlashInfer SM120 (`trtllm_batch_decode_sparse_mla_dsv4`, 37 layers)
- **Decode (Graph):** `B=1..16`: `24.6–25.3 µs` (`kv=512..8192`, `0.91–0.94 ms`/37L), `30.6–31.8 µs` (`kv=16384`, `1.13–1.18 ms`/37L), `40.7–41.5 µs` (`kv=32768`, `1.51–1.53 ms`/37L). `B=64`: `69.7–73.2 µs` (`kv<=8K`, `2.58–2.71 ms`/37L), `86.1 µs` (`16K`, `3.18 ms`), `129.9 µs` (`32K`, `4.81 ms`).
- **Prefill:** `seq=512`: `831.3 µs` (`30.76 ms`/37L); `seq=2048`: `3143.3 µs` (`116.30 ms`/37L); `seq=8192`: `12770.7 µs` (`472.51 ms`/37L).
- **HCA/DSA Post-Attention Ops (already near DRAM BW roof at prefill):**
  - `vha_inv_rope_fp8_quant`: `4.3 µs` (`T=1`), `6.2 µs` (`T=16`), `11.9 µs` (`T=64`), `238.9 µs` (`T=512`), `963.0 µs` (`T=2048`, 94% DRAM roof), `3871.3 µs` (`T=8192`).
  - `sigmoid_mul_quant_fp8` (8K HCA / 16K DSA): `3.0 / 3.2 µs` (`T=1`), `4.1 / 6.2 µs` (`T=64`), `379.1 / 749.6 µs` (`T=2048`, 99% DRAM roof), `1481.8 / 2904.2 µs` (`T=8192`).

#### Lightning Indexer DeepGEMM FP8 MQA Logits + Top-K (6 layers)
- **Decode (`meta + logits + topk_decode`):**
  - `B=1`: `20.2 µs` (`512`), `19.8 µs` (`2K`), `33.4 µs` (`8K`), `43.4 µs` (`16K`), `71.1 µs` (`32K`, `0.43 ms`/6L)
  - `B=16`: `19.6 µs` (`512`), `19.7 µs` (`2K`), `41.4 µs` (`8K`), `66.4 µs` (`16K`), `167.0 µs` (`32K`, `1.00 ms`/6L)
  - `B=64`: `20.3 µs` (`512`), `27.4 µs` (`2K`), `153.4 µs` (`8K`), `279.9 µs` (`16K`), `545.2 µs` (`32K`, `3.27 ms`/6L)
- **Prefill (`logits + topk_prefill`):**
  - `seq=512`: `28.4 µs` (`0.17 ms`/6L); `seq=2048`: `324.7 µs` (`1.95 ms`/6L); `seq=8192`: `4164.9 µs` (`24.99 ms`/6L).

### 3.3 Verified SM12.1 Attention & Indexer Kernel Wins

#### A. GB10-Calibrated HCA & DSA Dispatch (`kernels/mhc_hca/sm120_attn_dispatch.py`)

| Layer Type | Shape | Upstream | `sm120_attn_dispatch` | Speedup | Full-Model Saved | `max_abs_err` |
|---|---|---:|---:|---:|---:|---:|
| **HCA (37L)** | `B=1, kv=8192` | 24.6 µs | **20.6 µs** | **1.20×** | **+0.15 ms/tok** | `1.91e-06` |
| **HCA (37L)** | `B=1, kv=32768` | 41.0 µs | **32.8 µs** | **1.25×** | **+0.30 ms/tok** | `1.91e-06` |
| **HCA (37L)** | `B=4, kv=32768` | 41.0 µs | **33.1 µs** | **1.24×** | **+0.29 ms/tok** | `1.91e-06` |
| **DSA (6L)** | `B=4, topk=2176` | 62.9 µs | **59.5 µs** | **1.06×** | +0.02 ms/tok | `1.22e-04` |
| **DSA (6L)** | `B=8, topk=2176` | 118.5 µs | **100.4 µs** | **1.18×** | **+0.11 ms/tok** | `1.22e-04` |
| **DSA (6L)** | `B=16, topk=2176` | 358.2 µs | **111.8 µs** | **3.20×** | **+1.48 ms/tok** | `1.22e-04` |
| **DSA (6L)** | `B=32, topk=2176` | 708.0 µs | **232.0 µs** | **3.05×** | **+2.86 ms/tok** | `1.22e-04` |
| **DSA (6L)** | `B=64, topk=2176` | 2527.0 µs | **430.5 µs** | **5.87×** | **+12.58 ms/tok** | `1.22e-04` |

#### B. Fused DSA Indexer Triton Kernels (`kernels/mhc_hca/dsa_indexer_fused.py`, 6 layers)

| Kernel | `T=1` | `T=16` | `T=64` | `T=512` | `T=2048` | `T=8192` | Parity |
|---|---:|---:|---:|---:|---:|---:|---|
| `fused_merge_swa_topk_to_global` | 4.2→**2.6 µs** (1.60×) | 9.1→**3.2 µs** (2.83×) | 6.0→**4.1 µs** (1.47×) | 34.3→**10.2 µs** (3.35×) | 296.4→**159.3 µs** (1.86×, +0.82 ms) | 1318.1→**634.8 µs** (2.08×, **+4.10 ms**) | **Bit-exact (`True`)** |
| `fused_hadamard_fp8_quant_q` | 6.2→**3.6 µs** (1.74×) | 8.3→**4.1 µs** (2.01×) | 9.1→**6.8 µs** (1.34×) | 73.2→**31.7 µs** (2.31×) | 547.3→**235.7 µs** (2.32×, +1.87 ms) | 2193.6→**931.7 µs** (2.35×, **+7.57 ms**) | `q_rel_l2 ~1.5e-2`, `w_rel_l2 0–2.3e-2` (FP8 boundary) |

---

## 4. End-to-End Estimated Savings Summary (Full 43-Layer Model)

| Operating Regime | mHC (85 sites) | HCA (37L) | DSA MLA (6L) | DSA Indexer Fused (6L) | **Total Saved / Step** |
|---|---:|---:|---:|---:|---:|
| **Decode `B=1, kv=32K`** | +0.02 ms | **+0.30 ms** | 0.00 ms | +0.03 ms | **+0.35 ms/token** *(+0.15 ms more if DSA graph break removed)* |
| **Decode `B=16, kv=8K`** | +0.18 ms | 0.00 ms | **+1.48 ms** | +0.06 ms | **+1.72 ms/step** |
| **Decode `B=32, kv=8K`** | **+1.15 ms** | 0.00 ms | **+2.86 ms** | +0.04 ms | **+4.05 ms/step** |
| **Decode `B=64, kv=32K`** | +0.05 ms | 0.00 ms | **+12.58 ms** | +0.02 ms | **+12.65 ms/step** |
| **Prefill `seq=2048`** | **+37.88 ms** | 0.00 ms | 0.00 ms | **+2.69 ms** | **+40.57 ms/prefill** |
| **Prefill `seq=8192`** | *(~+151 ms est.)* | 0.00 ms | 0.00 ms | **+11.67 ms** | **~+163 ms/prefill** |

## 5. Deliverables
- [bench_mhc_sm121.py](bench/mhc_hca/bench_mhc_sm121.py)
- [bench_hca_dsa_sm121.py](bench/mhc_hca/bench_hca_dsa_sm121.py)
- [mhc_sm121.py](kernels/mhc_hca/mhc_sm121.py)
- [sm120_attn_dispatch.py](kernels/mhc_hca/sm120_attn_dispatch.py)
- [dsa_indexer_fused.py](kernels/mhc_hca/dsa_indexer_fused.py)
- [test_parity_and_speed.py](kernels/mhc_hca/test_parity_and_speed.py)
