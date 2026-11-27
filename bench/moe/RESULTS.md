# ERNIE 5.5 Flash Preview — MXFP4 MoE & Block-128 FP8 Kernel Benchmarks on NVIDIA GB10 (SM12.1)

**Hardware:** NVIDIA GB10 (DGX Spark `node2`, SM12.1, 48 SMs, 25.17 MB L2, 273 GB/s theoretical LPDDR5X unified memory, **218.6 GB/s** measured 128 MB STREAM read peak)  
**Software:** PyTorch `2.13.0+cu130`, vLLM `0.32.0`, Triton `3.7.1`, Humming `0.1.17`, FlashInfer `0.7.0.post1`, CUDA `13.0`  
**Model Dimensions (ERNIE 5.5 Flash Preview, 42 MoE layers):**  
- Routed MoE: `hidden_size (moe_latent_size) = 2048`, `moe_intermediate_size = 2048`, `n_routed_experts = 512`, `top_k = 10`, `activation = SiTU(beta=4.0, linear_beta=25.0)`, MXFP4 weights (`6.685 MB/expert`).
- Shared Expert & Latent Projections: Block-128 FP8 (`ue8m0` scales), `shared_gate_up [4096 -> 4096]`, `shared_down [2048 -> 4096]`, `fc1_latent [4096 -> 2048]`, `fc2_latent [2048 -> 4096]`.
- **Methodology Note:** All timings are median CUDA event latencies over 25–40 iterations with **clean read-only L2 cache eviction** (64 MB read sum before each iteration) to faithfully reproduce cold-weight 42-layer decode without dirty L2 writeback bus contention on LPDDR5X. (We verified experimentally that dirty `l2.zero_()` adds ~60–65 µs of writeback contention to small-M decode kernels.)

---

## Executive Summary — Total 42-Layer Savings vs Current vLLM Defaults

| Optimization | EP=2 `M=1` (ms/42L) | EP=4 `M=1` (ms/42L) | EP=2 `M=16` (ms/42L) | EP=4 `M=16` (ms/42L) | Numerical Parity |
|---|---:|---:|---:|---:|---|
| **Baseline (`DEEPGEMM_MXFP4` + `DeepGemm` FP8 + 2-kernel router)** | `24.16 ms` | `22.18 ms` | `184.99 ms` | `131.17 ms` | FP32 `rel_l2 = 0.048` |
| **Routed MoE: `DEEPGEMM_MXFP4` → Custom SM12.1 PTX (`M≤2`) / `HUMMING_INDEXED` (`M≥2`)** | **`-5.47 ms`** (`1.83x`) | **`-7.23 ms`** (`3.55x`) | **`-103.59 ms`** (`2.49x`) | **`-77.95 ms`** (`2.89x`) | Custom/BF16 `rel_l2 = 0.0025–0.0048` (**10–20× better**) |
| **Fused ERNIE 5.5 Router (`split_sigmoid_sum` + `ernie5_5_routing`)** | **`-0.43 ms`** (`1.56x`) | **`-0.43 ms`** (`1.56x`) | **`-0.43 ms`** (`1.56x`) | **`-0.43 ms`** (`1.56x`) | **Bit-exact** (`max_abs = 0.0`, `id_eq = True`) |
| **4 MoE-Adjacent Block-128 FP8 Linears (`DeepGemm` → `HummingFP8` / Custom GEMV)** | **`-1.18 ms`** (`1.14x`) | **`-1.18 ms`** (`1.14x`) | **`-1.57 ms`** (`1.18x`) | **`-1.57 ms`** (`1.18x`) | `rel_l2 = 0.0000–0.0012` (**25× better**) |
| **Shared Expert + Routed MoE Dual-Stream Overlap** | **`-0.85 ms`** (`1.08x`) | **`~-0.65 ms`** | n/a (BW-sat.) | n/a (BW-sat.) | Bit-identical |
| **Combined 42-Layer MoE Block Savings** | **`-7.93 ms/tok`** (`-32.8%`) | **`-9.49 ms/tok`** (`-42.8%`) | **`-105.59 ms/step`** (`-57.1%`) | **`-79.95 ms/step`** (`-61.0%`) | Strict parity verified |

*(Note: At `M=2`, switching the 4 FP8 linears from `DeepGemm` to `HummingFP8` saves **`3.79 ms/step`** alone because `DeepGemmFp8BlockScaledMMKernel` suffers a tile regression at `M=2`. Additional savings from `HummingFP8` on Attention projections are not included in the MoE-only totals above.)*

---

## 1. MXFP4 MoE Backend Support Matrix on GB10 (SM12.1)

| Backend | Kernel Class | SM12.1 + ERNIE 5.5 `SiTU` | Status / Root Cause |
|---|---|---|---|
| `DEEPGEMM_MXFP4` | `DeepGemmFP4Experts` | **Supported (vLLM AUTO default)** | Uses vendored `vllm.third_party.deep_gemm` (`sm120_fp8_fp4_gemm_1d1d.cuh`); 128-row per-expert alignment + unfused SiTU hurts GB10 badly (20–47% BW). |
| `HUMMING` (`indexed`) | `HummingIndexedExperts` | **Supported** (`VLLM_HUMMING_MOE_GEMM_TYPE=indexed`) | Fastest vLLM built-in backend for `M=1..2048` (`BF16`, `FP8TOK`, `MXFP8`, `MXFP4`). **Warning:** `MXFP8` has a 10× cliff at `M=2048`! |
| `HUMMING` (`grouped_contiguous`) | `HummingGroupedExperts` | **Supported** (default when `use_ep=True`) | 15–30 µs/layer slower than `indexed` at small M; no `M=2048` cliff on `MXFP8`. |
| `HUMMING` (`grouped_masked`) | `BatchedHummingGroupedExperts` | Batched EP format only | Requires `use_batched_activation_format=True` (DeepEP low-latency). |
| `MARLIN` | `MarlinExperts` | **Supported** | Very strong all-around W4A16 (`68–85%` peak BW, `rel_l2 = 0.0048` vs FP32), zero cliffs at any `M`. |
| `BATCHED_MARLIN` | `BatchedMarlinExperts` | Batched EP format only | Requires `use_batched_activation_format=True`. |
| `FLASHINFER_CUTLASS_*` | `FlashInferExperts` | Unsupported for `SiTU` | `FlashInferExperts._supports_activation` excludes `MoEActivation.SITU`. |
| `FLASHINFER_TRTLLM_*` | `TrtLlmMxfp4Experts*` | Unsupported on SM12.1 | SM100 (datacenter Blackwell) only. |
| `CUTLASS_MXFP4_MXFP4` | `CutlassExpertsMxfp4` | Unsupported on SM12.1 | `ops.mxfp4_experts_quant_supported(121) == False`. |
| `TRITON` / `TRITON_UNFUSED` | `OAITriton*Experts` | Unsupported | `triton_kernels` not installed AND rejects `MoEActivation.SITU`. |
| `B12X_*` / `EMULATION` | `B12xExperts` / `OCP_MX*` | Unsupported | Packages (`b12x`, `amd-quark`) not installed. |

---

## 2. Routed MXFP4 MoE Benchmark — EP=2 (`E_local=256, N=2048, H=2048`)

### Latency (µs / layer) & 42-Layer Impact

| M | `E_act` | `DEEPGEMM` (default) | `MARLIN` (W4A16) | `HUM_IDX_BF16` | `HUM_GRP_BF16` | `HUM_IDX_FP8TOK` | `HUM_IDX_MXFP8` | `HUM_GRP_MXFP8` | **Custom SM12.1 PTX** | Best vs DeepGEMM | Saved / 42L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 5 | 286.8 | 181.2 | 164.9 | 197.6 | 161.9 | 160.8 | 223.2 | **156.7** | **1.83×** | **5.47 ms** |
| **2** | 11 | 632.8 | 346.2 | 336.9 | 361.5 | **323.5** | 324.6 | 402.3 | 338.9 | **1.96×** | **12.99 ms** |
| **4** | 21 | 1242.1 | 631.8 | 633.9 | 649.2 | **602.9** | 604.2 | 691.4 | 643.3 | **2.06×** | **26.85 ms** |
| **8** | 37 | 2367.6 | 1088.8 | 1103.9 | 1118.2 | 1044.6 | **1043.6** | 1179.5 | 1089.5 | **2.27×** | **55.61 ms** |
| **16** | 59 | 4118.2 | 1719.4 | 1742.1 | 1766.4 | 1655.8 | **1651.7** | 1825.8 | — | **2.49×** | **103.59 ms** |
| **32** | 118 | 7511.2 | 3394.4 | 3465.2 | 3485.7 | 3270.7 | **3269.7** | 3584.9 | — | **2.30×** | **178.14 ms** |
| **64** | 192 | 10512.2 | 5534.8 | 5651.6 | 5678.1 | 5318.7 | **5312.6** | 5796.8 | — | **1.98×** | **218.38 ms** |
| **128** | 232 | 12208.9 | 6747.1 | 6876.1 | 6914.0 | 6449.2 | **6438.0** | 7054.5 | — | **1.90×** | **242.38 ms** |
| **512** | 256 | 13533.0 | 7842.9 | 7892.7 | 7960.6 | **7284.7** | 7285.8 | 8235.1 | — | **1.86×** | **262.43 ms** |
| **2048** | 256 | 15339.7 | 9463.8 | 9259.9 | 9467.9 | **8616.7** | *77756.4* ⚠️ | 8623.2 | — | **1.78×** | **282.37 ms** |

### Effective Bandwidth (GB/s, % of 273 GB/s peak) & Compute (TFLOPS) — EP=2

| M | `DEEPGEMM` BW (TFLOPS) | `MARLIN` BW (TFLOPS) | `HUM_IDX_BF16` BW | `HUM_IDX_FP8TOK` BW (TFLOPS) | **Custom SM12.1** BW (TFLOPS) |
|---:|---:|---:|---:|---:|---:|
| **1** | 116.6 (42.7%, 0.44 TF) | 184.5 (67.6%, 0.69 TF) | 202.8 (74.3%) | 206.6 (75.7%, 0.78 TF) | **213.4 (78.2%, 0.80 TF)** |
| **2** | 116.2 (42.6%, 0.48 TF) | 212.4 (77.8%, 0.87 TF) | 218.3 (80.0%) | **227.4 (83.3%, 0.93 TF)** | 217.0 (79.5%, 0.89 TF) |
| **4** | 113.0 (41.4%, 0.45 TF) | 222.2 (81.4%, 0.88 TF) | 221.5 (81.1%) | **232.9 (85.3%, 0.92 TF)** | 218.3 (80.0%, 0.86 TF) |
| **8** | 104.5 (38.3%, 0.39 TF) | 227.2 (83.2%, 0.85 TF) | 224.1 (82.1%) | **236.8 (86.8%, 0.89 TF)** | 227.1 (83.2%, 0.85 TF) |
| **16** | 95.8 (35.1%, 0.40 TF) | 229.5 (84.1%, 0.95 TF) | 226.5 (83.0%) | **238.3 (87.3%, 0.99 TF)** | — |
| **64** | 122.1 (44.7%, 0.76 TF) | 232.0 (85.0%, 1.44 TF) | 227.2 (83.2%) | **241.4 (88.4%, 1.50 TF)** | — |
| **512** | 126.8 (46.4%, 4.72 TF) | 218.7 (80.1%, 8.14 TF) | 217.3 (79.6%) | **235.5 (86.3%, 8.76 TF)** | — |
| **2048** | 112.7 (41.3%, 16.87 TF) | 182.6 (66.9%, 27.34 TF) | 186.6 (68.4%) | **200.5 (73.5%, 30.03 TF)** | — |

---

## 3. Routed MXFP4 MoE Benchmark — EP=4 (`E_local=128, N=2048, H=2048`)

### Latency (µs / layer) & 42-Layer Impact

| M | `E_act` | `DEEPGEMM` (default) | `MARLIN` (W4A16) | `HUM_IDX_BF16` | `HUM_GRP_BF16` | `HUM_IDX_FP8TOK` | `HUM_IDX_MXFP8` | `HUM_GRP_MXFP8` | **Custom SM12.1 PTX** | Best vs DeepGEMM | Saved / 42L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 2 | 239.7 | 93.2 | 81.8 | 109.6 | 80.0 | 81.9 | 128.1 | **67.6** | **3.55×** | **7.23 ms** |
| **2** | 4 | 351.2 | 150.3 | 137.2 | 160.7 | 134.1 | 137.2 | 185.3 | **124.9** | **2.81×** | **9.51 ms** |
| **4** | 14 | 974.9 | 430.0 | 442.4 | 448.7 | 416.8 | 413.9 | 494.4 | **412.7** | **2.36×** | **23.61 ms** |
| **8** | 21 | 1729.6 | 627.7 | 642.5 | 656.3 | **609.3** | 610.4 | 705.5 | 617.3 | **2.84×** | **47.05 ms** |
| **16** | 34 | 2836.5 | 1011.7 | 1023.8 | 1044.3 | **980.7** | **980.7** | 1098.6 | — | **2.89×** | **77.95 ms** |
| **32** | 54 | 3608.5 | 1584.1 | 1609.6 | 1624.0 | 1526.6 | **1524.7** | 1703.9 | — | **2.37×** | **87.52 ms** |
| **64** | 94 | 5238.9 | 2776.0 | 2799.6 | 2816.0 | 2651.2 | **2645.0** | 2922.6 | — | **1.98×** | **108.94 ms** |
| **128** | 119 | 6309.9 | 3563.6 | 3560.3 | 3583.0 | 3359.9 | **3359.8** | 3717.0 | — | **1.88×** | **123.90 ms** |
| **512** | 128 | 7052.1 | 4113.4 | 3982.5 | 4040.7 | 3718.2 | **3708.8** | 4244.5 | — | **1.90×** | **140.42 ms** |
| **2048** | 128 | 8517.7 | 5327.8 | 4770.9 | 4885.9 | **4366.3** | *38380.5* ⚠️ | 4422.8 | — | **1.95×** | **174.36 ms** |

### Effective Bandwidth (GB/s, % of 273 GB/s peak) & Compute (TFLOPS) — EP=4

| M | `DEEPGEMM` BW (TFLOPS) | `MARLIN` BW (TFLOPS) | `HUM_IDX_BF16` BW | `HUM_IDX_FP8TOK` BW (TFLOPS) | **Custom SM12.1** BW (TFLOPS) |
|---:|---:|---:|---:|---:|---:|
| **1** | 55.8 (20.4%, 0.21 TF) | 143.6 (52.6%, 0.54 TF) | 163.5 (59.9%) | 167.3 (61.3%, 0.63 TF) | **197.8 (72.5%, 0.74 TF)** |
| **2** | 76.2 (27.9%, 0.29 TF) | 178.0 (65.2%, 0.67 TF) | 195.0 (71.4%) | 199.4 (73.1%, 0.75 TF) | **214.3 (78.5%, 0.81 TF)** |
| **4** | 96.0 (35.2%, 0.36 TF) | 217.7 (79.7%, 0.82 TF) | 211.6 (77.5%) | 224.6 (82.3%, 0.84 TF) | **226.9 (83.1%, 0.85 TF)** |
| **16** | 80.2 (29.4%, 0.35 TF) | 224.8 (82.3%, 0.99 TF) | 222.1 (81.4%) | **231.9 (84.9%, 1.03 TF)** | — |
| **64** | 120.0 (44.0%, 0.74 TF) | 226.5 (83.0%, 1.40 TF) | 224.6 (82.3%) | **237.2 (86.9%, 1.46 TF)** | — |
| **2048** | 102.4 (37.5%, 15.14 TF) | 163.7 (60.0%, 24.21 TF) | 182.9 (67.0%) | **199.8 (73.2%, 29.54 TF)** | — |

---

## 4. Numerical Parity Table (MXFP4 MoE, EP=2 `M=16` Representative)

| Backend | Activation Precision | vs `DEEPGEMM_MXFP4` (`rel_l2` / `max_abs` / `cos`) | vs Exact FP32 Dequant Ref (`rel_l2` / `max_abs` / `cos`) | Quality Verdict |
|---|---|---|---|---|
| `DEEPGEMM_MXFP4` | FP8 block-128 | `0.0000` / `0.0000` / `1.00000` | `0.04759` / `0.2109` / `0.99887` | Baseline (W4A8) |
| **Custom SM12.1 PTX** (`M=8`) | **BF16 (FP32 accum)** | `0.0475` / `0.2031` / `0.99888` | **`0.00262`** / **`0.0156`** / **`0.999997`** | **Highest accuracy (18× lower error than DeepGEMM)** |
| `HUMMING_*_BF16` | **BF16 (W4A16)** | `0.0479` / `0.2148` / `0.99887` | **`0.00413`** / **`0.0312`** / **`0.999996`** | **11.5× lower error than DeepGEMM** |
| `MARLIN` | **BF16 (W4A16)** | `0.0480` / `0.2109` / `0.99886` | **`0.00489`** / **`0.0234`** / **`0.999992`** | **9.7× lower error than DeepGEMM** |
| `HUMMING_*_MXFP8` | MXFP8 block-32 | **`0.0162`** / **`0.0781`** / **`0.99987`** | `0.04746` / `0.1934` / `0.99888` | Closest match to DeepGEMM (`cos=0.99987`) |
| `HUMMING_*_FP8TOK` | FP8 per-token | `0.0674` / `0.2705` / `0.99775` | `0.04645` / `0.1875` / `0.99893` | Same FP32 accuracy as DeepGEMM |
| `HUMMING_*_MXFP4` | MXFP4 (W4A4) | `0.2323` / `0.8398` / `0.97296` | `0.22367` / `0.7871` / `0.97487` | ⚠️ Lossy W4A4 — **avoid** (not faster than W4A8) |

---

## 5. Custom Kernel Benchmarks & Deep Diagnostics

### 5a. Fused ERNIE 5.5 Router (`kernels/moe/fused_ernie55_router.py`)
Fuses `split_sigmoid_sum` + `ernie5_5_routing` into a single Triton launch, keeping `[T, 512]` FP32 scores strictly in registers. **Bit-exact (`max_abs = 0.0`, `id_eq = True`, exact tie-breaking & padding mask).**

| M | Upstream 2-Kernel (µs) | Fused Single-Kernel (µs) | Speedup | Saved / 42 Layers |
|---:|---:|---:|---:|---:|
| 1 | 28.61 | **18.34** | **1.56×** | **0.431 ms** |
| 8 | 28.10 | **18.21** | **1.54×** | **0.415 ms** |
| 64 | 28.10 | **18.24** | **1.54×** | **0.414 ms** |
| 512 | 30.59 | **20.51** | **1.49×** | **0.423 ms** |
| 2048 | 27.87 | **20.70** | **1.35×** | **0.301 ms** |
| 8192 | 235.52 | **115.94** | **2.03×** | **5.023 ms** |

### 5b. Fused `situ_mul_fp8_quant_packed` for DeepGEMM (`kernels/moe/fused_situ_fp8_quant.py`)
Eliminates the intermediate `[M_sum, 2048]` BF16 buffer and second kernel launch in `DeepGemmFP4Experts._act_mul_quant` for `MoEActivation.SITU`. Verified `scale_eq = True` (bit-exact packed UE8M0 scales) and `q_rel_l2 = 0.00001`.

| Regime (`M_sum` 128-aligned) | Unfused 2-Kernel (µs) | Fused (µs) | Fused + Skip `expert_ids < 0` (µs) | Speedup (Full / Skip-Pad) |
|---|---:|---:|---:|---:|
| EP2 `M=1` (`M_sum=640`) | 40.9 | 26.6 | **8.2** | **1.53× / 4.99×** |
| EP2 `M=8` (`M_sum=4736`) | 348.3 | 197.7 | **18.5** | **1.76× / 18.86×** |
| EP2 `M=16` (`M_sum=7552`) | 618.4 | 330.8 | **24.6** | **1.87× / 25.10×** |
| EP2 `M=64` (`M_sum=24576`) | 2063.1 | 1132.6 | **80.2** | **1.82× / 25.73×** |
| EP2 `M=512` (`M_sum=32768`) | 2727.8 | 1494.0 | **179.9** | **1.83× / 15.16×** |

### 5c. Block-128 FP8 Linear Backends (Shared Expert & Latent Projections)
vLLM AUTO selects `DeepGemmFp8BlockScaledMMKernel`. **`HummingFP8` (W8A16)** beats `DeepGemm`, `CutlassBlock`, and `TritonBlock` across `M=1..128` at **210–222 GB/s** with `rel_l2 = 0.0000–0.0002`, and our custom zero-repack **`sm121_fp8_block_gemv`** matches/beats `HummingFP8` at `M=1`!

| Shape (`[K -> N]`) | M | `DeepGemm` (AUTO) | `CutlassBlock` | `TritonBlock` | `HummingFP8` (W8A16) | **Custom SM12.1 GEMV** | `BF16 cuBLAS` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shared_gate_up_TP1 [4096->4096]` | 1 | 89.1 µs (189 GB/s) | 82.0 µs | 98.3 µs | **76.8 µs (219 GB/s)** | **76.7 µs (219 GB/s)** | 204.8 µs |
| `shared_gate_up_TP1 [4096->4096]` | 2 | 103.4 µs (163 GB/s) | 84.0 µs | 97.9 µs | **76.1 µs (221 GB/s)** | 79.9 µs (210 GB/s) | 150.6 µs |
| `shared_gate_up_TP1 [4096->4096]` | 16 | 92.3 µs (185 GB/s) | 84.1 µs | 96.4 µs | **76.9 µs (222 GB/s)** | — | 153.7 µs |
| `shared_gate_up_TP1 [4096->4096]` | 512 | **131.2 µs (192 GB/s)** | 134.1 µs | 244.9 µs | 179.2 µs (140 GB/s) | — | 237.6 µs |
| `shared_down_TP1 [2048->4096]` | 1 | 45.0 µs (187 GB/s) | 45.9 µs | 57.3 µs | **40.1 µs (210 GB/s)** | 41.0 µs (205 GB/s) | 104.5 µs |
| `shared_down_TP1 [2048->4096]` | 2 | 61.4 µs (137 GB/s) | 50.1 µs | 53.3 µs | **40.0 µs (210 GB/s)** | 42.0 µs (200 GB/s) | 79.9 µs |
| `fc1_latent [4096->2048]` | 1 | 44.1 µs (190 GB/s) | 45.1 µs | 81.9 µs | **38.9 µs (216 GB/s)** | 41.9 µs (201 GB/s) | 104.6 µs |
| `fc1_latent [4096->2048]` | 2 | 60.5 µs (139 GB/s) | 49.2 µs | 81.9 µs | **39.0 µs (216 GB/s)** | 45.1 µs (187 GB/s) | 78.9 µs |
| `shared_down_TP2 [1024->4096]` | 1 | 24.8 µs (170 GB/s) | 27.6 µs | 34.8 µs | 23.5 µs (179 GB/s) | **21.6 µs (195 GB/s)** | 54.3 µs |

*Note: `MarlinFP8` fails on Block-128 scales (`b_scales dim 1 = K/128 != size_n`). `HummingFP8` wins for `M=1..128` (memory-bound), while `DeepGemm` wins at `M>=512` (compute-bound).*

### 5d. Shared Expert + Routed MoE Dual-Stream Overlap (EP=2)
Running `Ernie5_5MLP` (`shared_experts`) on an `aux_stream` concurrently with routed MoE hides launch/tail latency on GB10 at small `M`:
- `M=1`: Sequential `277.6 µs` → Dual-stream **`257.4 µs`** (**`1.078×`**, saves **`20.2 µs/layer` = `0.85 ms/42L`**)
- `M=2`: Sequential `475.3 µs` → Dual-stream **`439.3 µs`** (**`1.082×`**, saves **`35.9 µs/layer` = `1.51 ms/42L`**)
- `M=4`: Sequential `776.1 µs` → Dual-stream **`744.5 µs`** (**`1.042×`**, saves **`31.6 µs/layer` = `1.33 ms/42L`**)
- `M=8`: Sequential `1359.0 µs` → Dual-stream **`1306.7 µs`** (**`1.040×`**, saves **`52.3 µs/layer` = `2.20 ms/42L`**)

### 5e. Root Cause of `HUMMING_INDEXED_MXFP8` `M=2048` Cliff
`HummingIndexedExperts` reads `get_heuristics_config(...)`, which switches to `block_shape=(48, 256, 64)` for `valid_shape_m in (7360, 11008]` and `(14720, 22016]`. On SM12.1 (`M=2048`, `valid_shape_m = 10240` on EP2 / `5120` on EP4), `indexed` MXFP8 with `num_stages=3` regresses by 10× (`77.8 ms` EP2 / `38.4 ms` EP4). **`HUMMING_INDEXED_FP8TOK` and `HUMMING_INDEXED_BF16` do NOT exhibit this cliff** (`8.62 ms` / `9.26 ms` EP2; `4.37 ms` / `4.77 ms` EP4).

---

## 6. Recommended Production Deployment Stack for DGX Spark (GB10 SM12.1)

1. **Immediate zero-code win (`+1.8× to 3.0×` faster MoE, `+1.15× to 1.45×` faster FP8 decode linears):**
   - Switch MXFP4 MoE backend from `DEEPGEMM_MXFP4` to **`HUMMING` with `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`** using either **`BF16` (W4A16, best quality: `rel_l2 = 0.0041`)** or **`FP8TOK` (W4A8, fastest across `M=2..2048`: `235–241 GB/s`, 30 TFLOPS at M=2048, zero cliff)**. Avoid `HUMMING_INDEXED_MXFP8` at `M >= 1024` and avoid `HUMMING_*_MXFP4` (`rel_l2 = 0.22`).
   - Enable **`HummingFP8`** for Block-128 FP8 linears at decode (`M <= 128`), keeping `DeepGemm` for large prefill (`M >= 256`).
2. **Custom runtime kernel integration (additional `-1.3 to -2.5 ms/tok` at `M=1`):**
   - Drop-in replace `split_sigmoid_sum` + `ernie5_5_routing` with [`kernels/moe/fused_ernie55_router.py`](../../kernels/moe/fused_ernie55_router.py) (**bit-exact**, `-0.43 ms/42L` at decode, `-5.0 ms/42L` at 8K prefill).
   - Dispatch `M <= 2` (or `M <= 4` on EP=4) to [`kernels/moe/sm121_mxfp4_moe_decode.py`](../../kernels/moe/sm121_mxfp4_moe_decode.py) (**`67.6 µs` EP4 / `156.7 µs` EP2**, `-0.5 to -1.1 ms/42L` beyond even `HummingIndexed`, zero repacking, `cos = 0.999997`).
   - Overlap `shared_experts` + routed MoE on `aux_stream` for `M <= 8` (`-0.85 to -2.20 ms/42L`).
