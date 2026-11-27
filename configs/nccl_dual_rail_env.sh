#!/usr/bin/env bash
# Generic, zero-hardcoded-IP dual-rail RoCEv2 NCCL environment profile for 2x-4x DGX Spark (GB10).
#
# Usage:
#   source configs/nccl_dual_rail_env.sh [profile]
#
# Expects node-specific variables (auto-populated by `spark-ernie discover` + `spark-ernie start`):
#   MASTER_ADDR, VLLM_HOST_IP, NCCL_SOCKET_IFNAME, NCCL_IB_HCA, WORLD_SIZE
# Optional:
#   SPARK_NCCL_LIB   Path to patched switchless 4-HCA libnccl.so (for 4-node switchless ring)
#   profile          balanced (default: ^LL128, 4 channels) | throughput (Simple, 4 channels)
#
# CRITICAL GB10 NOTE:
#   GB10 lacks GDAKI/GPUDirect Async, so RoCEv2 NCCL must NOT run inside FULL CUDA graphs.
#   Use FULL_AND_PIECEWISE (with sparknet one-shot) or PIECEWISE CUDA graphs.

_nccl_dual_rail_apply() {
    local profile="${1:-balanced}"

    # Optimal GB10 RoCEv2 baseline (200GbE ConnectX-7, MTU 9000, RoCEv2 GID 3)
    export NCCL_NET="${NCCL_NET:-IB}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
    export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export GLOO_DISABLE_IPV6="${GLOO_DISABLE_IPV6:-1}"

    case "$profile" in
        balanced)
            export NCCL_PROTO="${NCCL_PROTO:-^LL128}"
            unset NCCL_BUFFSIZE 2>/dev/null || true
            ;;
        throughput)
            export NCCL_PROTO="Simple"
            unset NCCL_BUFFSIZE 2>/dev/null || true
            ;;
        *)
            echo "Unknown NCCL profile '$profile' (expected 'balanced' or 'throughput')" >&2
            return 2
            ;;
    esac

    # For 4-node switchless rings, enable bidirectional subnet-aware routing & patched multi-HCA libnccl
    if [ "${WORLD_SIZE:-2}" -ge 4 ]; then
        export NCCL_ALGO="${NCCL_ALGO:-Ring}"
        export NCCL_CROSS_NIC=1
        export NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-1}"
        export NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
        export NCCL_IB_SUBNET_PREFIX_LEN="${NCCL_IB_SUBNET_PREFIX_LEN:-24}"
        export NCCL_SWITCHLESS_BIDIRECTIONAL="${NCCL_SWITCHLESS_BIDIRECTIONAL:-2}"
        export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-4194304}"
        export NCCL_COLLNET_ENABLE=0
        export NCCL_NVLS_ENABLE=0
        export NCCL_PAT_ENABLE=0
        export NCCL_RMA_DISABLE=1
        export NCCL_RUNTIME_CONNECT=1
        export NCCL_RING4_BULK_OK="${NCCL_RING4_BULK_OK:-1}"
        local nccl_so=""
        if [ -n "${SPARK_NCCL_LIB:-}" ] && [ -f "$SPARK_NCCL_LIB" ]; then
            nccl_so="$SPARK_NCCL_LIB"
        elif [ -f "/opt/nccl-switchless-4hca/libnccl.so.2.30.7" ]; then
            nccl_so="/opt/nccl-switchless-4hca/libnccl.so.2.30.7"
        fi
        if [ -n "$nccl_so" ]; then
            export VLLM_NCCL_SO_PATH="$nccl_so"
            export LD_PRELOAD="${nccl_so}${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
    fi
}

_nccl_dual_rail_apply "${1:-balanced}"
