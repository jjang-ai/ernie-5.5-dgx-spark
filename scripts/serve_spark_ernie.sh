#!/usr/bin/env bash
# Portable DGX Spark (GB10, SM12.1) launcher for ERNIE 5.5 Flash Preview MXFP4.
# Works out of the box with `spark-ernie discover` + `spark-ernie start` (zero hardcoded paths/IPs).
#
# Usage:
#   serve_spark_ernie.sh <mode> <topology> <rank> <model_path> [port] [master_port]
#
# Modes:
#   tp2-dspark   2x DGX Spark TP=2 + EP=2 + DSpark (default 2x fast 1M-context profile)
#   tp2-ar       2x DGX Spark TP=2 + EP=2 Pure AR (max concurrent 1M contexts / max batch)
#   tp4-dspark   4x DGX Spark TP=4 + EP=4 + DSpark (default 4x fast 1M-context profile)
#   tp4-ar       4x DGX Spark TP=4 + EP=4 Pure AR (max concurrent 1M contexts)
#   dp2-ar       2x DGX Spark DP=2 + EP=2 (experimental zero-KV-dup)
#   dp4-ar       4x DGX Spark DP=4 + EP=4 (experimental zero-KV-dup)
set -euo pipefail

MODE="${1:?mode}"
TOPO="${2:-custom}"
RANK="${3:?rank}"
MODEL="${4:?model_path}"
PORT="${5:-8000}"
MPORT="${6:-29801}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-load discovered node topology from ~/.config/spark-ernie/cluster.json when not pre-set
CLUSTER_JSON="${SPARK_CLUSTER_CONFIG:-${HOME}/.config/spark-ernie/cluster.json}"
if [ -z "${MASTER_ADDR:-}" ] && [ -f "$CLUSTER_JSON" ]; then
    eval "$(python3 - "$CLUSTER_JSON" "$RANK" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1]))
r = int(sys.argv[2])
n = cfg["nodes"][r]
kv = {
    "MASTER_ADDR": cfg["master_addr"],
    "WORLD_SIZE": str(cfg["world_size"]),
    "VLLM_HOST_IP": n["host_ip"],
    "NCCL_SOCKET_IFNAME": n["oob_iface"],
    "GLOO_SOCKET_IFNAME": n["oob_iface"],
    "TP_SOCKET_IFNAME": n["oob_iface"],
    "NCCL_IB_HCA": n["nccl_ib_hca"],
    "SPARKNET_ROCE_TOPOLOGY": cfg["topology"],
    "SPARKNET_ROCE_PEER_HCAS": json.dumps(n["sparknet_peer_hcas"], separators=(",", ":")),
    "SPARKNET_ROCE_SUBNETS": json.dumps(n["sparknet_subnets"], separators=(",", ":")),
}
if cfg.get("nccl_lib_path"):
    kv["SPARK_NCCL_LIB"] = cfg["nccl_lib_path"]
for k, v in kv.items():
    print(f"export {k}={shlex.quote(v)}")
PY
)"
fi

if [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: MASTER_ADDR is not set and $CLUSTER_JSON was not found." >&2
    echo "Run 'spark-ernie discover --hosts <host0>,<host1>[,<host2>,<host3>]' first." >&2
    exit 2
fi

NCCL_PROFILE="${SPARK_NCCL_PROFILE:-balanced}"
source "$REPO_ROOT/configs/nccl_dual_rail_env.sh" "$NCCL_PROFILE"

# Ensure TileLang uses system CUDA 13 toolkit matching SM12.1 headers
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "${CUDA_HOME}/bin" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# Unified-RAM glibc malloc tuning to prevent arena fragmentation on 121.7 GiB GB10
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-1048576}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# TP=2 cold NVMe load of 79.5 GiB MXFP4 + Humming repack takes ~8-10 min; avoid 600s default timeout
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

# Optional sparknet one-shot RoCEv2 collectives (10.3 us 2x / 15.3 us 4x ring!)
export SPARK_USE_SPARKNET="${SPARK_USE_SPARKNET:-1}"
if [ "$SPARK_USE_SPARKNET" = "1" ]; then
    export SPARKNET_ROCE_CACHE_DIR="${SPARKNET_ROCE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sparknet_roce}"
    export SPARKNET_ROCE_PROXY_CPU="${SPARKNET_ROCE_PROXY_CPU:-fast}"
    export SPARKNET_ROCE_TIMEOUT_MS="${SPARKNET_ROCE_TIMEOUT_MS:-120000}"
    export SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES="${SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES:-2097152}"
    if [ "${SPARKNET_ROCE_TOPOLOGY:-}" = "ring4" ]; then
        export SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES="${SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES:-2097152}"
        export SPARKNET_ROCE_ALLGATHER_MAX_BYTES="${SPARKNET_ROCE_ALLGATHER_MAX_BYTES:-8388608}"
    else
        export SPARKNET_ROCE_ALLGATHER_MAX_BYTES="${SPARKNET_ROCE_ALLGATHER_MAX_BYTES:-8388608}"
    fi
fi

# Activate SM12.1 ERNIE 5.5 runtime overlay across all vLLM subprocesses
export SPARK_ERNIE_RUNTIME="${SPARK_ERNIE_RUNTIME:-1}"
export SPARK_PATCH_FLASHINFER_CPB="${SPARK_PATCH_FLASHINFER_CPB:-1}"
export SPARK_PATCH_ROUTER="${SPARK_PATCH_ROUTER:-1}"
export SPARK_PATCH_MHC="${SPARK_PATCH_MHC:-1}"
export SPARK_PATCH_HUMMING_FP8_SAFE="${SPARK_PATCH_HUMMING_FP8_SAFE:-1}"
export SPARK_REPLICATE_SHARED_EXPERTS="${SPARK_REPLICATE_SHARED_EXPERTS:-0}"
export SPARK_PATCH_DSPARK_MARKOV="${SPARK_PATCH_DSPARK_MARKOV:-1}"
export SPARK_DSPARK_TOPK="${SPARK_DSPARK_TOPK:-64}"
export SPARK_PATCH_ADAPTIVE_VERIFY="${SPARK_PATCH_ADAPTIVE_VERIFY:-1}"
export SPARK_DSPARK_VERIFY_RULE="${SPARK_DSPARK_VERIFY_RULE:-marginal}"
export SPARK_DSPARK_MAIN_PROJ_TP="${SPARK_DSPARK_MAIN_PROJ_TP:-1}"
export SPARK_DSPARK_VOCAB_PARALLEL="${SPARK_DSPARK_VOCAB_PARALLEL:-1}"
export SPARK_PATCH_KV_PACK="${SPARK_PATCH_KV_PACK:-1}"
export SPARK_KV_PACK_MAX_DSA_SPLITS="${SPARK_KV_PACK_MAX_DSA_SPLITS:-2}"
export SPARK_PACKED_BF16_LM_HEAD="${SPARK_PACKED_BF16_LM_HEAD:-1}"
export SPARK_PACKED_BF16_MARKOV="${SPARK_PACKED_BF16_MARKOV:-1}"
export SPARK_PATCH_DSPARK_PAD_MASK="${SPARK_PATCH_DSPARK_PAD_MASK:-1}"
export SPARK_PATCH_VHA_BF16_AR="${SPARK_PATCH_VHA_BF16_AR:-1}"
export SPARK_PATCH_VHA_TRITON="${SPARK_PATCH_VHA_TRITON:-1}"
export SPARK_PATCH_FC2_LATENT_TP="${SPARK_PATCH_FC2_LATENT_TP:-1}"
export SPARK_PATCH_FC1_LATENT_TP="${SPARK_PATCH_FC1_LATENT_TP:-0}"
export SPARK_PATCH_GATE_BF16X2="${SPARK_PATCH_GATE_BF16X2:-1}"
export SPARK_PATCH_HCA_OPROJ_FUSED="${SPARK_PATCH_HCA_OPROJ_FUSED:-1}"
export SPARK_HCA_PIPELINE="${SPARK_HCA_PIPELINE:-1}"
export SPARK_DSPARK_MOE_COST_CURVE="${SPARK_DSPARK_MOE_COST_CURVE:-1}"
export SPARK_DSPARK_VP_PROB="${SPARK_DSPARK_VP_PROB:-1}"
export SPARK_DSPARK_DRAFT_TAU="${SPARK_DSPARK_DRAFT_TAU:-1.0}"
export SPARK_DSPARK_TOP_P="${SPARK_DSPARK_TOP_P:-0.0}"
export VLLM_MOE_SKIP_PADDING="${VLLM_MOE_SKIP_PADDING:-1}"
export PYTHONPATH="$REPO_ROOT/spark_ernie_runtime:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Humming MXFP4 MoE (indexed GEMM) + HummingFP8 safe selective linears -> 219-242 GB/s!
export VLLM_HUMMING_MOE_GEMM_TYPE="${VLLM_HUMMING_MOE_GEMM_TYPE:-indexed}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

MOE_BACKEND="${SPARK_MOE_BACKEND:-humming}"
LINEAR_BACKEND="${SPARK_LINEAR_BACKEND:-humming}"
CG_MODE="${SPARK_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
SERVED_NAME="${SPARK_SERVED_MODEL_NAME:-ernie55}"
BIND_HOST="${SPARK_BIND_HOST:-0.0.0.0}"
MAX_SEQS="${SPARK_MAX_NUM_SEQS:-32}"
MAX_BATCHED="${SPARK_MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_UTIL="${SPARK_GPU_MEM_UTIL:-0.78}"
MAX_LEN="${SPARK_MAX_MODEL_LEN:-1048576}"
DSPARK_K="${SPARK_DSPARK_NUM_TOKENS:-7}"
DSPARK_ADAPTIVE="${SPARK_DSPARK_ADAPTIVE:-1}"
PREFIX_CACHE="${SPARK_ENABLE_PREFIX_CACHING:-1}"

if [ "$SPARK_DSPARK_VP_PROB" = "1" ]; then
    DSPARK_SAMPLE="${SPARK_DSPARK_SAMPLE_METHOD:-probabilistic}"
elif [ "$SPARK_DSPARK_VOCAB_PARALLEL" = "1" ]; then
    DSPARK_SAMPLE="${SPARK_DSPARK_SAMPLE_METHOD:-greedy}"
else
    DSPARK_SAMPLE="${SPARK_DSPARK_SAMPLE_METHOD:-probabilistic}"
fi
export SPARK_DSPARK_SAMPLE_METHOD="$DSPARK_SAMPLE"

DSPARK_EXTRA=""
if [ "$DSPARK_ADAPTIVE" = "1" ]; then
    DSPARK_EXTRA=',"enable_adaptive_verification":true'
fi

EXTRA_ARGS=()
# Always expose cached_tokens / cache_read_input_tokens across OpenAI Chat, Responses & Anthropic /v1/messages
if [ "${SPARK_PROMPT_TOKENS_DETAILS:-1}" = "1" ]; then
    EXTRA_ARGS+=(--enable-prompt-tokens-details)
fi
# Ensure Anthropic thinking={"type":"disabled"} truly disables thinking on ERNIE 5.5 (mapped cleanly by patch.py)
if [ "${SPARK_PATCH_ANTHROPIC_COMPAT:-1}" = "1" ]; then
    EXTRA_ARGS+=(--anthropic-disabled-thinking-effort none)
fi
if [ "$PREFIX_CACHE" = "1" ]; then
    EXTRA_ARGS+=(--enable-prefix-caching)
    # Retain periodic SWA checkpoints every 2048 tokens (multiple of 256 block size) so hybrid HCA+DSA
    # prefix caching never rolls back 100K+ tokens on partial hits across Chat, Responses & Anthropic!
    RETENTION_INTERVAL="${SPARK_PREFIX_CACHE_RETENTION_INTERVAL:-2048}"
    if [ -n "$RETENTION_INTERVAL" ] && [ "$RETENTION_INTERVAL" != "0" ]; then
        EXTRA_ARGS+=(--prefix-cache-retention-interval "$RETENTION_INTERVAL")
    fi
else
    EXTRA_ARGS+=(--no-enable-prefix-caching)
fi
if [ "$RANK" != "0" ] && [[ "$MODE" != dp* ]]; then
    EXTRA_ARGS+=(--headless)
fi
# API key is passed via VLLM_API_KEY env var (vLLM reads it automatically without exposing it in ps)

case "$MODE" in
    tp2-ar)
        PAR_ARGS=(--tensor-parallel-size 2 --enable-expert-parallel --nnodes 2 --node-rank "$RANK")
        SPEC_ARGS=()
        ;;
    tp2-dspark)
        PAR_ARGS=(--tensor-parallel-size 2 --enable-expert-parallel --nnodes 2 --node-rank "$RANK")
        SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K},\"draft_sample_method\":\"${DSPARK_SAMPLE}\"${DSPARK_EXTRA}}")
        MAX_SEQS="${SPARK_MAX_NUM_SEQS:-8}"
        GPU_UTIL="${SPARK_GPU_MEM_UTIL:-0.86}"
        MAX_BATCHED="${SPARK_MAX_NUM_BATCHED_TOKENS:-2048}"
        CG_MAX_SIZE="${SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE:-64}"
        KV_MEM_BYTES="${SPARK_KV_CACHE_MEMORY_BYTES:-9663676416}"
        export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
        ;;
    tp4-ar)
        PAR_ARGS=(--tensor-parallel-size 4 --enable-expert-parallel --nnodes 4 --node-rank "$RANK")
        SPEC_ARGS=()
        ;;
    tp4-dspark)
        PAR_ARGS=(--tensor-parallel-size 4 --enable-expert-parallel --nnodes 4 --node-rank "$RANK")
        SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K},\"draft_sample_method\":\"${DSPARK_SAMPLE}\"${DSPARK_EXTRA}}")
        ;;
    dp2-ar)
        PAR_ARGS=(--tensor-parallel-size 1 --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-start-rank "$RANK" --data-parallel-address "$MASTER_ADDR" --data-parallel-rpc-port "$MPORT" --enable-expert-parallel)
        SPEC_ARGS=()
        ;;
    dp4-ar)
        PAR_ARGS=(--tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 1 --data-parallel-start-rank "$RANK" --data-parallel-address "$MASTER_ADDR" --data-parallel-rpc-port "$MPORT" --enable-expert-parallel)
        SPEC_ARGS=()
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

# Resolve vllm binary portably
VLLM_BIN="${SPARK_VLLM_BIN:-}"
if [ -z "$VLLM_BIN" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/vllm" ]; then
        VLLM_BIN="${VIRTUAL_ENV}/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
        VLLM_BIN="$(command -v vllm)"
    else
        echo "ERROR: 'vllm' executable not found. Activate your venv or set SPARK_VLLM_BIN." >&2
        exit 2
    fi
fi

CG_MAX_SIZE="${CG_MAX_SIZE:-${SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE:-}}"
KV_MEM_BYTES="${KV_MEM_BYTES:-${SPARK_KV_CACHE_MEMORY_BYTES:-}}"
if [ -n "$KV_MEM_BYTES" ] && [ "$KV_MEM_BYTES" != "0" ]; then
    EXTRA_ARGS+=(--kv-cache-memory-bytes "$KV_MEM_BYTES")
fi
COMP_JSON="{\"cudagraph_mode\":\"${CG_MODE}\"}"
if [ -n "$CG_MAX_SIZE" ]; then
    COMP_JSON="{\"cudagraph_mode\":\"${CG_MODE}\",\"max_cudagraph_capture_size\":${CG_MAX_SIZE}}"
fi

if [[ "$MODE" == dp* ]]; then
    exec "$VLLM_BIN" serve "$MODEL" \
      --served-model-name "$SERVED_NAME" --trust-remote-code --host "$BIND_HOST" --port "$PORT" \
      --generation-config auto \
      "${PAR_ARGS[@]}" \
      --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype fp8 \
      --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
      --compilation-config "$COMP_JSON" \
      --kernel-config "{\"moe_backend\":\"${MOE_BACKEND}\",\"linear_backend\":\"${LINEAR_BACKEND}\"}" \
      --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice \
      "${SPEC_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    exec "$VLLM_BIN" serve "$MODEL" \
      --served-model-name "$SERVED_NAME" --trust-remote-code --host "$BIND_HOST" --port "$PORT" \
      --generation-config auto \
      --distributed-executor-backend mp \
      "${PAR_ARGS[@]}" \
      --master-addr "$MASTER_ADDR" --master-port "$MPORT" \
      --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype fp8 \
      --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
      --compilation-config "$COMP_JSON" \
      --kernel-config "{\"moe_backend\":\"${MOE_BACKEND}\",\"linear_backend\":\"${LINEAR_BACKEND}\"}" \
      --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice \
      "${SPEC_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
