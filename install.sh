#!/usr/bin/env bash
# One-command setup & optional model download/cluster-sync installer for ernie-5.5-dgx-spark.
#
# Usage:
#   ./install.sh                                      # Install CLI & runtime overlay only
#   ./install.sh --download-model                     # Also download dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4
#   ./install.sh --hosts spark0,spark1 --download-model  # Install, discover 2x/4x cables, download & spread to all Sparks!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DO_DOWNLOAD=0
HOSTS=""
MODEL_REPO="dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4"
MODEL_DEST="${HOME}/models/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4"

while [ $# -gt 0 ]; do
    case "$1" in
        --download-model) DO_DOWNLOAD=1; shift ;;
        --hosts)          HOSTS="${2:?--hosts requires comma-separated hosts}"; shift 2 ;;
        --repo)           MODEL_REPO="${2:?--repo requires HF repo id}"; shift 2 ;;
        --dest)           MODEL_DEST="${2:?--dest requires local path}"; shift 2 ;;
        -h|--help)
            echo "Usage: ./install.sh [--hosts h0,h1[,h2,h3]] [--download-model] [--repo <hf_repo>] [--dest <dir>]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

echo "==> Installing ernie-5.5-dgx-spark (editable) with $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps -e "$SCRIPT_DIR"

if [ -n "$HOSTS" ]; then
    echo "==> Discovering 200GbE QSFP / RoCEv2 topology across: $HOSTS"
    spark-ernie discover --hosts "$HOSTS"
fi

if [ "$DO_DOWNLOAD" = "1" ]; then
    echo "==> Downloading $MODEL_REPO -> $MODEL_DEST (and syncing to discovered peers if configured)"
    spark-ernie download --repo "$MODEL_REPO" --dest "$MODEL_DEST"
else
    echo ""
    echo "Optional: Download & spread the official uncensored MXFP4 model across your 2x/4x Sparks:"
    echo "  https://huggingface.co/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4"
    echo "  spark-ernie download"
fi

echo ""
echo "Installed! Quick start:"
echo "  1. Discover your 2x or 4x 200GbE QSFP cables:"
echo "       spark-ernie discover --hosts spark0,spark1                 # 2x direct"
echo "       spark-ernie discover --hosts spark0,spark1,spark2,spark3   # 4x ring"
echo "  2. (Optional) Download & spread ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4 across all Sparks:"
echo "       spark-ernie download"
echo "  3. Configure optional API key & defaults:"
echo "       spark-ernie init --api-key auto --reasoning-effort high"
echo "  4. Start with 1M context, VRAM prefix caching & DSpark enabled by default:"
echo "       spark-ernie start"
echo "  5. Stream with high reasoning (/v1/chat/completions or /v1/responses):"
echo "       spark-ernie chat"
echo "       spark-ernie chat --api responses \"Write a lock-free SPSC ring buffer in C++20\""
echo "  6. Verify everything:"
echo "       spark-ernie test"
