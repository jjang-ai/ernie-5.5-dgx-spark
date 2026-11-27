# `ernie-5.5-dgx-spark`

**All-in-one, switchless 2×–4× NVIDIA DGX Spark (`GB10`, `SM12.1`) 1M-context runtime, custom SM12.1 kernels, automatic 200GbE QSFP cable discovery, prefix-aware `2×2` parallelized router, and OpenAI Chat/Responses CLI for [`ERNIE-5.5-Flash-Preview`](https://huggingface.co/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4) MXFP4 on vLLM.**

- **Official Uncensored Model (Hugging Face):** **[`dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`](https://huggingface.co/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4)** — optional one-command download & parallel 200GbE cluster fan-out built right into `./install.sh --download-model` and `spark-ernie download`!
- **All Three DGX Spark Topologies Supported Out of the Box (`1,048,576` `1M` Native Context on Every One!):**
  1. **`2×` Direct (`2x-1m-fast`, `TP=2 + EP=2`):** Full **`1,048,576` (`1M`) context (`1.09×` `1M` = `1,143,108` KV tokens)** with DSpark on just 2 Sparks! **`55.97 tok/s` avg / `69.67 tok/s` peak** on real 8-turn `T=1.0, top_p=0.95, reasoning=high` coding (`0/8` loops), **`2,003.5 tok/s`** 32K cold prefill (**`2.00×` stock**), and **`168.3 tok/s`** @ `B=8` code (**`3.09×` stock `54.4`**)!
  2. **`4×` Switchless Ring (`4x-1m-fast`, `TP=4 + EP=4`):** Lowest single-stream latency! **`79.64 tok/s` avg / `93.51 tok/s` peak** on real 8-turn `T=1.0, top_p=0.95, reasoning=high` coding (**`+33.7%`** vs Wave-3, `8/8 stop`, **`0/8` loops**, `100%` Turn-1 recall), up to **`105.6–106.9 tok/s`** single-stream (`zh` / greedy code), **`2,800 tok/s`** 8K & **`2,663 tok/s`** 64K cold prefill (**`+35%`** vs Wave-3!), up to **`445–497 tok/s`** @ `B=32`, and **`2.59×–4.67×` concurrent `1M` contexts (`2.71M–4.90M` KV tokens)**!
  3. **`2×2` Parallelized (`4x-dual-tp2-fast` / `4x-dual-tp2-max-batch`):** Two independent `TP=2 + EP=2` `1M`-context replicas behind a built-in zero-dependency **prefix-aware OpenAI load-balancing router** (`:8000` -> `:8001` + `:8002`)! Zero cross-pair RoCE traffic, **`2,286,216` independent pooled KV tokens (`2.18×` `1M`)**, **`338.1 tok/s`** @ `B=16` code (**`2.01×`** single `2×` pair & **`4.31×` stock**), `8/8 stop` / **`0/8` loops** multi-turn coding with sticky prefix cache, and **`~4,000 tok/s` dual-pair aggregate cold prefill capacity** (`2 × 2,003.5 tok/s`)!
- **Zero switch required & zero hardcoded IPs:** Automatic 200GbE ConnectX-7 RoCEv2 cable & 4-node ring + `2×2` pair discovery (`spark-ernie discover`) — even if you pass your hosts out of physical ring order!
- **Official sampling & zero-loop reliability:** `--generation-config auto` enforces ERNIE 5.5's recommended `temperature=1.0, top_p=0.95, do_sample=true`, `reasoning_effort=high` default, VRAM prefix caching ON by default (`25–83×` faster warm TTFT), DSpark Adaptive Verification ON by default, and clean `download` / `discover` / `start` / `stop` / `restart` (`reboot`) / `status` / `logs` / `chat` (`/v1/chat/completions` & `/v1/responses` streaming) / `test` commands!
- **Zero destructive edits** to upstream `vllm` or `vllm-ernie` wheels — all `17` SM12.1 runtime patches load cleanly via `sitecustomize.py` / `spark_ernie_runtime.patch.install()` and are individually toggleable via environment variables.

---

## Verified Live Performance (`ERNIE-5.5-Flash-Preview` MXFP4 on GB10, Wave-5 `v0.4.0`)

All numbers below were measured end-to-end on real **2× and 4× NVIDIA DGX Spark (`GB10`, `sm_121`, 121.7 GiB unified RAM, DGX OS `7.6.0` / Linux `7.0.0-1025-nvidia` + `kho=off`, ConnectX-7 FW `28.50.1002` unthrottled `200GbE` MTU 9000)** serving [`dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`](https://huggingface.co/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4):

### Real Multi-Turn Coding (`T=1.0, top_p=0.95, reasoning_effort=high`, 8 Growing Turns, Zero Loops)
| Topology & Profile (`v0.4.0` Wave-5) | Context Window | Avg Decode (`tok/s`) | Peak Turn (`tok/s`) | Clean Stops | Loops | Turn-1 Recall | VRAM KV Capacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`2×` Direct (`2x-1m-fast`)** | **`1,048,576` (`1M`)** | **`55.97`** | **`69.67`** | **8 / 8** | **0 / 8** | **100%** | **`1,143,108` tok (`1.09×` 1M)** |
| **`2×2` Parallelized (`4x-dual-tp2-fast`, `:8000` router)** | **`1,048,576` (`1M` × 2)** | `~50–60` e2e/stream *(×2 streams!)* | `~70` /stream | **8 / 8** | **0 / 8** | **100%** | **`2,286,216` tok (`2.18×` 1M)** |
| **`4×` Switchless Ring (`4x-1m-fast`)** | **`1,048,576` (`1M`)** | **`79.64`** *(+33.7% vs Wave-3!)* | **`93.51`** | **8 / 8** | **0 / 8** | **100%** | **`2,712,218` tok (`2.59×` 1M)** |

### Single-Stream (`B=1`) & Cold Prefill Speedups (`2×` vs `2×2` vs `4× Ring`)
| Cluster & Profile | Context Window | `code` (`T=1.0` / greedy) | `prose` (`T=1.0` / greedy) | `high_zh` (`B=1` dec / e2e) | `algo_off` (`B=1`) | Cold Prefill (`8K–64K`) | TTFT (short) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Stock 2× Spark** (1-rail, `DEEPGEMM`, AR) | `128K` | `20.4` | `18.4` | — | — | `~1,000` | `~0.45s` |
| **2× `2x-max-batch`** (`tp2-ar` + `sparknet` 10.3 µs) | `128K` | `24.2` | `24.1` | `23.6` | `24.2` | `1,620` | `0.164s` |
| **2× `2x-1m-fast` (Wave-5, DEFAULT 2×)** | **`1,048,576` (`1M`)** | **`49.1` / `51.3`** *(`71.9` grd-draft)* | **`37.9` / `36.8–41.6`** | **`76.95` / `73.61`** | **`67.78`** | **`2,003.5`** *(32K, 2.00× stock!)* | **`0.149s`** |
| **2×2 `4x-dual-tp2-fast` (Wave-5, Dual `TP=2`)** | **`1,048,576` (`1M` × 2)** | `51.0` e2e/stream | `36.4` e2e/stream | `75.59` e2e/stream | `63.08` e2e | **`~4,000` agg** *(`2 × 2,003`)* | **`~0.15s`** *(v0.4.0 SSE)* |
| **4× Ring `4x-1m-fast` (Wave-3)** | **`1,048,576` (`1M`)** | `53.0` / `71.3` | `48.3` | `99.7` e2e | `100.5` | `2,486` (8K) / `1,973` (64K) | `0.101s` |
| **4× Ring `4x-1m-fast` (Wave-5, DEFAULT 4×)** | **`1,048,576` (`1M`)** | **`70.9` / `75.3`** *(`106.9` grd-draft)* | **`52.1` / `54.6`** | **`105.58` / `101.29`** *(`100.7` @ T=1)* | **`92.66–100.9`** | **`2,800.3`** (8K) / **`2,663.4`** (64K, +35%!) | **`0.113s`** |

### Concurrency & Batched Decode Throughput (`tok/s` aggregate)
| Cluster & Profile | `B=1` | `B=2` | `B=4` | `B=8` | `B=16` | `B=32` | `B=64` | `B=128` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stock 2× Spark** | `18.4` | — | `43.5` | `54.4` | `78.4` | — | — | — |
| **2× `2x-1m-fast` (Wave-5, prose)** | `40.3` | `61.6` | `95.4` | **`139.4`** (`19.5`/req, **2.56× stock!**) | — | — | — | — |
| **2× `2x-1m-fast` (Wave-5, code)** | `43.7` | `66.7` | `100.9` | **`168.3`** (`24.0`/req, **3.09× stock!**) | — | — | — | — |
| **2× `2x-max-batch`** (`tp2-ar`, `128K`) | `24.0` | — | `64.7` | `97.9` | `146.7` | `223.5` | **`333.9`** (`352` pk) | **`464.6`** (**`537.6` pk!**) |
| **2×2 `4x-dual-tp2-fast` (Wave-5, prose)** | `38.5` | `72.3` | `114.5` | **`186.4`** (`28.5`/req) | `199.2+` | — | — | — |
| **2×2 `4x-dual-tp2-fast` (Wave-5, code)** | `37.3` | **`96.0`** | **`131.3`** | **`210.9`** (`40.3`/req) | **`338.1`** (**`2.01×` 2×, `4.31×` stock!**) | — | — | — |
| **4× Ring `4x-1m-fast` (Wave-5, prose)** | `49.5` | `78.5` | `130.6` | **`214.0`** (`29.1`/req, **+13.2%**) | **`316.0`** (`21.2`/req) | `379.2–445.4`\* | — | — |
| **4× Ring `4x-1m-fast` (Wave-5, code)** | `57.1` | `95.2` | `133.8` | `214.9` (`31.6`/req) | **`347.1`** (`24.6`/req) | `432.4–496.6`\* | — | — |

*\*Wave-3/4 achieved `438.6–445.4 tok/s` prose / `479.9–496.6 tok/s` code at `B=32`; `v0.4.0` caps the coupon-collector MoE cost curve at `m <= 64` (`bw=250 GB/s`) to preserve both `B=1..16` Wave-5 wins and the full `B=32` ceiling.*

### True VRAM Prefix Caching & Long-Context Recall (`100%` Exact Recall)
| Cluster & Profile | Prompt Length | Cold Prefill (`tok/s`) | Cold TTFT | Warm TTFT | TTFT Speedup | Secret Code Recall |
|---|---:|---:|---:|---:|---:|---:|
| **2× `2x-1m-fast` (Wave-5)** | `6,881` (`8K`) | **`1,971.0 tok/s`** | `3.49s` | **`0.415s`** | **`8.4×`** (`59.5` dec) | `3/3 (100%)` |
| **2× `2x-1m-fast` (Wave-5)** | `27,341` (`32K`) | **`2,003.5 tok/s`** | `13.65s` | **`0.548s`** | **`24.9×`** (`59.6` dec) | `3/3 (100%)` |
| **2× `2x-max-batch`** | `57,735` (`64K`) | `1,474 tok/s` | `39.18s` | **`0.488s`** | **`80.2×`** | `3/3 (100%)` |
| **4× Ring `4x-1m-fast` (Wave-5)** | `6,902` (`8K`) | **`2,800.3 tok/s`** | `2.46s` | **`0.334s`** | **`7.4×`** (`99.7` dec) | `3/3 (100%)` |
| **4× Ring `4x-1m-fast` (Wave-4/5)** | `27,337` (`32K`) | **`2,481.2 tok/s`** | `11.02s` | **`0.418s`** | **`31.7×`** (`104.8` dec) | `3/3 (100%)` |
| **4× Ring `4x-1m-fast` (Wave-5)** | `54,665` (`64K`) | **`2,663.4 tok/s`** | `20.52s` | **`0.517s`** | **`39.7×`** (`104.9` dec) | `3/3 (100%)` |
| **4× Ring `4x-1m-fast` (`1M` ctx)** | **`115,407` (`128K`)** | `1,392 tok/s` | `82.89s` | **`1.073s`** | **`77.2×`** (`79.2` dec) | **`3/3 (100%)`** |

---

## Recommended Host OS, Kernel & Firmware (Unthrottled 200GbE + Extra Unified RAM)

Older DGX Spark kernels/firmware throttled ConnectX-7 RoCEv2 links and reserved `3–6 GiB` of extra unified RAM. For best stability and full `1M`-context headroom on `2×` and `4×` Sparks, upgrade all nodes to:
- **DGX OS:** `7.6.0` (Ubuntu 24.04) with Linux kernel **`7.0.0-1025-nvidia`** and `nvidia-spark-grub-kho` (`kho=off` in `/proc/cmdline` — avoids KHO unified-RAM reservation).
- **System / GPU Firmware:** `BIOS 5.36_0ACUM028` & `VBIOS 9A.0B.34.00.00` (recovers `+3–6 GiB` unified RAM/Spark).
- **ConnectX-7 NIC Firmware:** `28.50.1002` with `MTU 9000` on all active 200GbE QSFP ports (verified **`22.67 GB/s`** NCCL busBW = `91%` of `25 GB/s` raw line rate across all 4 Sparks, zero link drops or throttling).

---

## Quick Start (`2×`, `4× Ring`, or `2×2 Parallelized` DGX Spark)

### 1. Install (with Optional One-Command Model Download & Cluster Sync)
Run on each DGX Spark inside your Python venv containing `vllm` + `vllm-ernie` + `tilelang`:
```bash
git clone https://github.com/jjang-ai/ernie-5.5-dgx-spark.git
cd ernie-5.5-dgx-spark
./install.sh

# Strongly recommended: install sparknet v0.6.0 for 10.3 us (2x) / 15.3 us (4x) one-shot RoCEv2 collectives
pip install --no-deps git+https://github.com/christopherowen/dgx-spark-networking@v0.6.0
```
Want `./install.sh` to also **download [`dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`](https://huggingface.co/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4) and parallel-sync it across all 2 or 4 Sparks automatically**?
```bash
./install.sh --download-model --hosts spark0,spark1,spark2,spark3
```
Or download/distribute anytime after `spark-ernie discover`:
```bash
spark-ernie download                                  # downloads default model & parallel-syncs across cluster
spark-ernie download --no-sync                        # local only (or NFS shared storage)
spark-ernie download --dest /data/models/ernie55      # custom directory
```

> **Note for 4× Switchless Ring users:** For bulk collectives on a 4-node switchless 4-HCA ring (without a 200GbE switch), install a patched 4-HCA switchless NCCL library (e.g. `/opt/nccl-switchless-4hca/libnccl.so.2.30.7`, auto-detected by `spark-ernie discover` or set via `--nccl-lib`). Two-node direct pairs (`2×` and `2×2` Parallelized) work with stock NCCL out of the box!

### 2. Auto-Discover Your 200GbE QSFP Cables (Zero Manual IP Editing!)
`spark-ernie discover` probes your Sparks via SSH (`ibdev2netdev` + `ip -j addr`), detects every active 200GbE ConnectX-7 rail, **automatically solves BOTH the 4-node physical ring order AND the two independent `2×2` direct pairs**, and saves `~/.config/spark-ernie/cluster.json`:
```bash
# 2x DGX Spark direct dual-rail pair:
spark-ernie discover --hosts spark0,spark1

# 4x DGX Spark (supports BOTH 4x Switchless Ring AND 2x2 Parallelized modes!):
spark-ernie discover --hosts spark0,spark1,spark2,spark3
```

### 3. Initialize Your Model Path & Optional API Key
All settings live locally in `~/.config/spark-ernie/config.json` (`chmod 600`):
```bash
spark-ernie init \
  --model /path/to/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4 \
  --profile auto \
  --reasoning-effort high \
  --api-key auto
```
(`--profile auto` picks **`4x-1m-fast`** on 4 Sparks and **`2x-1m-fast`** on 2 Sparks — both with `1,048,576` context, VRAM prefix caching, byte-packed KV, probabilistic vocab-parallel DSpark, and Wave-5 SM12.1 kernels enabled by default!)

### 4. Start, Monitor, Reboot, and Manage API Keys
```bash
spark-ernie start                   # Launches all nodes (and router in 2x2 mode) and waits for READY
spark-ernie status                  # Node RAM, model, max_model_len, KV %, prefix hit %, DSpark accept %
spark-ernie logs -f                 # Follow rank 0 logs (or --rank 1..3)
spark-ernie test                    # 4-check live smoke test (Chat high, Responses, VRAM prefix cache, coding tok/s)

spark-ernie stop                    # Cleanly stops vLLM workers (and router) across all nodes
spark-ernie restart                 # (or `spark-ernie reboot`) Clean stop + start

# API key management (passed via VLLM_API_KEY env var — never exposed in `ps` or git)
spark-ernie api-key show            # Masked display (--reveal to print full key)
spark-ernie api-key generate        # Rotate to a new sk-spark-... key
spark-ernie api-key set my-secret   # Set a custom key
spark-ernie api-key clear           # Disable auth for trusted local LAN
```

### 5. Choose Profiles Anytime (`2× Direct` vs `4× Ring` vs `2×2 Parallelized`)
```bash
spark-ernie profiles                                   # List all 7 built-in profiles
spark-ernie restart --profile 4x-1m-fast               # 4x Ring: 1M ctx, 79.6 avg / 93.5 pk T=1.0 code, 2,800 pp/s
spark-ernie restart --profile 4x-1m-max-context        # 4x Ring: 1M ctx Pure AR (~5.5x+ concurrent 1M)
spark-ernie restart --profile 4x-dual-tp2-fast         # 2x2 Parallelized: 2x(TP=2 1M DSpark) + prefix-aware router (:8000)
spark-ernie restart --profile 4x-dual-tp2-max-batch    # 2x2 Parallelized: 2x(TP=2 128K AR) up to ~930–1,075 tok/s @ B=256
spark-ernie restart --profile 2x-1m-fast               # 2x Direct: 1M ctx, 56.0 avg / 69.7 pk T=1.0 code, 2,004 pp/s
spark-ernie restart --profile 2x-1m-max-context        # 2x Direct: 1M ctx Pure AR (max concurrent 1M on 2x)
spark-ernie restart --profile 2x-max-batch             # 2x Direct: 128K ctx, up to 537.6 tok/s peak @ B=128
```
You can also run the zero-dependency prefix-aware load-balancing router standalone against any pair of replicas:
```bash
python3 -m spark_ernie_runtime.router --host 0.0.0.0 --port 8000 --backends http://replica0:8001,http://replica1:8002
```

---

## Streaming Chat Completions & Responses API Usage

### Built-In Streaming CLI (`T=1.0, top_p=0.95, reasoning_effort=high` official defaults!)
The server passes `--generation-config auto` so API clients automatically inherit ERNIE 5.5's official `generation_config.json` (`temperature=1.0`, `top_p=0.95`, `do_sample=true`) to prevent greedy CoT stalls. `spark-ernie chat` also defaults to `temperature=1.0`, `top_p=0.95`, `reasoning_effort=high`, `max_tokens=8192`:
```bash
# Interactive multi-turn REPL (automatic VRAM prefix caching across turns!)
spark-ernie chat

# Single-shot with OpenAI Chat Completions (/v1/chat/completions, T=1.0, top_p=0.95, high reasoning)
spark-ernie chat "Implement a lock-free SPSC ring buffer in C++20"

# Single-shot with OpenAI Responses API (/v1/responses, streaming)
spark-ernie chat --api responses --effort high "Derive Sinkhorn-Knopp matrix scaling"

# Fast non-thinking coding mode (override sampling anytime with --temperature / --top-p / --top-k)
spark-ernie chat --effort off "Write a Python LRU cache with TTL expiration"
```

### OpenAI Python SDK (`/v1/chat/completions` & `/v1/responses`)
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="YOUR_API_KEY")

# 1. Chat Completions (streaming + official T=1.0, top_p=0.95 + high reasoning)
stream = client.chat.completions.create(
    model="ernie55",
    messages=[{"role": "user", "content": "Explain RoCEv2 PFC and ECN."}],
    temperature=1.0,
    top_p=0.95,
    max_tokens=8192,
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}},
)
for chunk in stream:
    delta = chunk.choices[0].delta
    rc = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
    if rc:
        print(rc, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)

# 2. Responses API (streaming + high reasoning)
with client.responses.stream(
    model="ernie55",
    input="Write a fast Python async rate limiter.",
    temperature=1.0,
    top_p=0.95,
    reasoning={"effort": "high"},
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

---

## What's Inside (`17` Non-Destructive SM12.1 Runtime Optimizations)

All 17 patches live in [`spark_ernie_runtime/patch.py`](spark_ernie_runtime/patch.py) and `kernels/`:

1. **cuBLAS `BF16x2` (`w_hi + w_lo`) Tensor-Core MoE Router (`SPARK_PATCH_GATE_BF16X2=1`):** [`kernels/moe/gate_bf16x2.py`](kernels/moe/gate_bf16x2.py) replaces FP32 SIMT `GateLinear` (which burned `246.2 ms` at 8K prefill on GB10!) with Tensor-Core `BF16x2` (`100.00%` exact top-10 at `T=8`, `99.99%` at `T=8192`; **`-161.1 ms` per 8K chunk** and **`2.2–2.6×` faster decode**).
2. **Fused HCA `vha_v_add_inv_rope_fp8_quant` + L2-Tiled `fp8_einsum` + 2-Chunk Compute/RDMA Pipeline + In-Place `sigmoid_mul` (`SPARK_PATCH_HCA_OPROJ_FUSED=1`, `SPARK_HCA_PIPELINE=1`):** [`kernels/mhc_hca/vha_triton.py`](kernels/mhc_hca/vha_triton.py) fuses `o + V@z` + inverse RoPE + per-128 UE8M0 FP8 quant (`100%` scale parity, `-44.5 ms` TP4 / `-94.4 ms` TP2 at 8K), fixes DeepGEMM `fp8_einsum`'s `3.53×` L2-thrashing cliff at `T=8192` via `T_tile=1024` (`99.5 → 28.0 ms` TP4, `198 → 91 ms` TP2), and overlaps `vha_ar` & `wo_b_ar` RDMA on `comm_stream` with GPU compute (`T >= 4096`) for **`2,800 tok/s` 8K / `2,663 tok/s` 64K** (`+35%`) cold prefill!
3. **GB10 L2-Aware `mhc_sm121(twopass)` Unlocked for `T=1..32` (`SPARK_PATCH_MHC=1`):** [`kernels/mhc_hca/mhc_sm121.py`](kernels/mhc_hca/mhc_sm121.py) runs register-resident `twopass` across `1 <= T <= 32` (`1.71–1.81×` faster at `T=1, 7, 8`, **`-2.9 ms/step`** across all 43 layers) and L2-chunked prefill at `T >= 512`.
4. **Probabilistic Vocab-Parallel DSpark + 64-Cand `multimax` (`SPARK_DSPARK_VOCAB_PARALLEL=1`, `SPARK_DSPARK_VP_PROB=1`, `SPARK_DSPARK_DRAFT_TAU=1.0`):** [`kernels/dspark/vocab_parallel_greedy.py`](kernels/dspark/vocab_parallel_greedy.py) shards `markov_w2` across TP ranks (`2×–4×` less Markov BW), skips the full-vocab draft gather (`+2.34 GiB` KV), and runs probabilistic drafting calibrated for official `T=1.0, top_p=0.95` sampling (`+33.7%` on real 8-turn high-reasoning coding, `0/8` loops).
5. **512-Expert Coupon-Collector MoE Cost Curve + Dinkelbach Marginal Adaptive Verification (`SPARK_PATCH_ADAPTIVE_VERIFY=1`, `SPARK_DSPARK_VERIFY_RULE=marginal`, `SPARK_DSPARK_MOE_COST_CURVE=1`):** Unlocks `enable_adaptive_verification=true` on SM12.1 (`FlashInferSM120` varlen-CG + `Indexer` flattened) and replaces the flat dummy-token startup cost table with ERNIE 5.5's true 512-expert coupon-collector slope (`m <= 64`, `bw=250 GB/s`), cutting prose step time `51.1 → 39.6 ms` while scaling batched decode cleanly.
6. **Fused SM12.1 Triton `VHAPostMix` + BF16 All-Reduce (`SPARK_PATCH_VHA_TRITON=1`, `SPARK_PATCH_VHA_BF16_AR=1`):** Register-tiled FP32-accumulated BF16 Triton kernels (`4.23×` faster on 4× prefill, `4.00×` on 2× prefill, `2.2–3.4×` faster decode) + half-size BF16 RDMA payload across all 43 layers.
7. **Row-Sharded `fc2_latent_proj` Fused Into `shared_experts` All-Reduce (`SPARK_PATCH_FC2_LATENT_TP=1`):** `4×` less `fc2` compute/DRAM/VRAM on TP=4 (`2×` on TP=2), **zero extra collectives**, 100% mathematically exact.
8. **Byte-Packed Hybrid KV-Cache Grouping (`SPARK_PATCH_KV_PACK=1`):** Packs all 37 tiny HCA compressed layers into 1 KV group (`16 → 12` groups), slashing per-request `1M`-context KV footprint **`14.02 → 8.26 GiB` (`1.70×` more `1M` capacity, `0.0%` DSA penalty)** — enabling full **`1,048,576` (`1M`) context even on 2× Sparks with DSpark!**
9. **Bidirectional Shared Lossless 12-bit Packed BF16 TileLang LM Head + `markov_w2` (`SPARK_PACKED_BF16_LM_HEAD=1`, `SPARK_PACKED_BF16_MARKOV=1`):** [`kernels/lm_head/packed_bf16_lm_head.py`](kernels/lm_head/packed_bf16_lm_head.py) (`1e-6` exact FP32 parity, CUDA-graph safe, shared bidirectionally across target & draft LM heads with **0 extra VRAM**: `1.29×` faster on TP=4, `1.36–1.79×` on TP=2).
10. **GB10 Unified-Memory Safetensors Page-Cache Eviction + `kv_cache_memory_bytes` `empty_cache()` Fix (`SPARK_PATCH_UMA_FADVISE=1`):** Unprivileged `os.posix_fadvise(..., POSIX_FADV_DONTNEED)` after `load_model()` immediately recovers `+3.2–4.6 GiB` of unified RAM before KV sizing (zero root), and wraps `Worker.determine_available_memory` with `_scoped_allocator_max_split(20)` + `torch.cuda.empty_cache()` to fix upstream vLLM's `~17 GiB` pinned-segment bug when `--kv-cache-memory-bytes` is set!
11. **One-Shot RoCEv2 Collectives via `sparknet` (`SPARK_USE_SPARKNET=1`):** Drop-in `CudaCommunicator` overlay inside `FULL_AND_PIECEWISE` CUDA graphs: **`10.3 µs`** on 2× direct (`6.8×` vs NCCL) and **`15.3 µs`** on 4× switchless ring (`6.0×` vs NCCL).
12. **DSpark Drafter Padding-Row MoE Mask (`SPARK_PATCH_DSPARK_PAD_MASK=1`):** Skips all 3 draft MoE layers on CUDA-graph padding rows.
13. **Column-Parallel DSpark `main_proj` (`SPARK_DSPARK_MAIN_PROJ_TP=1`):** Shards `[12288 → 4096]` across TP ranks (`2×–4×` less draft projection read).
14. **Humming MXFP4 Indexed MoE (`242 GB/s`) + Safe Selective `HummingFP8` Linears (`222 GB/s`)** (`SPARK_PATCH_HUMMING_FP8_SAFE=1`).
15. **FlashInfer SM120 CPB Fix (`SPARK_PATCH_FLASHINFER_CPB=1`):** `1.20–1.25×` HCA decode & `3.20–5.87×` DSA decode at `B=16..64`.
16. **Bit-Exact Fused ERNIE 5.5 MoE Router (`SPARK_PATCH_ROUTER=1`):** [`kernels/moe/fused_ernie55_router.py`](kernels/moe/fused_ernie55_router.py) (`1.56–2.03×`, `max_abs=0.0`).
17. **GB10 Unified-RAM & Startup Hardening:** `MALLOC_ARENA_MAX=2`, `MALLOC_MMAP_THRESHOLD_=1048576`, `VLLM_ENGINE_READY_TIMEOUT_S=1800` (prevents 600s timeout during cold TP=2 NVMe loads), `CUDA_HOME=/usr/local/cuda`, `SPARKNET_ROCE_TIMEOUT_MS=120000`, and tuned Dual-Rail NCCL (`4ch, ^LL128`, switchless bidirectional ring4).

---

## Documentation & Benchmarks
- Complete engineering log & full Wave-1 through Wave-5 benchmark tables: [`TODO_AND_BENCHMARKS.md`](TODO_AND_BENCHMARKS.md)
- Deep runtime, roofline & flag audit: [`docs/ARCHITECTURE_AND_FLAGS_AUDIT.md`](docs/ARCHITECTURE_AND_FLAGS_AUDIT.md)
- Subsystem microbenchmarks: [`bench/nccl/RESULTS.md`](bench/nccl/RESULTS.md), [`bench/moe/RESULTS.md`](bench/moe/RESULTS.md), [`bench/mhc_hca/RESULTS.md`](bench/mhc_hca/RESULTS.md)
- Reproducible benchmark JSONs & scripts: [`benchmarks/`](benchmarks/) (`bench_wave3.py`, `bench_multiturn_coding.py`, `2x_direct_1m_wave5*.json`, `2x2_parallel_1m_wave5*.json`, `4x_ring_1m_wave5*.json`).

## Credits & License
- Apache-2.0 (`LICENSE`).
- One-shot RoCEv2 collectives (`sparknet`), vocab-parallel greedy Markov drafting, column-parallel `main_proj`, exact 12-bit packed BF16 LM head layout, and GB10 unified-RAM tuning draw directly on [Christopher Owen's `dgx-spark-networking` & `spark-ds41f`](https://github.com/christopherowen/spark-ds41f) (Apache-2.0).
- Built for [vLLM](https://github.com/vllm-project/vllm) + `vllm-ernie` + [FlashInfer](https://github.com/flashinfer-ai/flashinfer) + [Humming Kernels](https://github.com/humming-ai/humming-kernels) + [TileLang](https://github.com/tile-ai/tilelang).
