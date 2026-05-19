#!/usr/bin/env bash
# =============================================================================
# start_vllm.sh
# -------------
# Start a vLLM vLLM inference server for the Clinical RAG demo.
#
# Usage:
#   bash scripts/start_vllm.sh                    # uses defaults from .env
#   bash scripts/start_vllm.sh --model <model-id> # override model
#   bash scripts/start_vllm.sh --port 8002        # override port
#   bash scripts/start_vllm.sh --help
#
# VRAM requirements (approximate, fp16):
#   mistralai/Mistral-7B-Instruct-v0.3  → 16 GB   (default)
#   mistralai/Mistral-3B-Instruct       →  8 GB
#   TinyLlama/TinyLlama-1.1B-Chat-v1.0 →  4 GB   (demo/CPU fallback)
#
# GPU check:
#   nvidia-smi --query-gpu=name,memory.total --format=csv
# =============================================================================

set -euo pipefail

# ── Load .env if present ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +o allexport
    echo "[start_vllm] Loaded .env"
fi

# ── Defaults (can be overridden by .env or CLI flags) ────────────────────────
MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-float16}"

# ── Parse CLI overrides ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";           shift 2 ;;
        --host)          HOST="$2";            shift 2 ;;
        --port)          PORT="$2";            shift 2 ;;
        --tensor-parallel) TENSOR_PARALLEL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2";   shift 2 ;;
        --gpu-util)      GPU_UTIL="$2";        shift 2 ;;
        --dtype)         DTYPE="$2";           shift 2 ;;
        --help|-h)
            sed -n '2,30p' "$0"   # Print the header comment
            exit 0
            ;;
        *)
            echo "Unknown flag: $1  (run with --help for usage)"
            exit 1
            ;;
    esac
done

# ── Environment check ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       Clinical RAG — vLLM Inference Server           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Model              : $MODEL"
echo "  Host:Port          : $HOST:$PORT"
echo "  Tensor parallel    : $TENSOR_PARALLEL"
echo "  Max model length   : $MAX_MODEL_LEN tokens"
echo "  GPU memory util    : $GPU_UTIL"
echo "  dtype              : $DTYPE"
echo ""

# Check vllm is available
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm is not installed in the current environment."
    echo ""
    echo "  Activate the project environment first:"
    echo "    conda activate vllm-clinical-rag"
    echo ""
    echo "  vllm (CPU) is on Anaconda main channel. The GPU build is in"
    echo "  progress — until it ships, the GPU version installs via pip"
    echo "  as part of environment.yml (conda env create -f environment.yml)."
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unavailable")
    echo "  GPU                : $GPU_INFO"
    echo ""
else
    echo "  ⚠  nvidia-smi not found — CPU inference will be slow."
    echo "     Set DTYPE=bfloat16 and expect reduced throughput."
    echo ""
    DTYPE="bfloat16"
fi

# ── Check if port already in use ──────────────────────────────────────────────
if lsof -ti:"$PORT" &>/dev/null 2>&1; then
    echo "  ⚠  Port $PORT is already in use."
    echo "     Stop the existing process or choose a different port with --port."
    echo ""
    read -r -p "  Kill existing process on port $PORT? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        lsof -ti:"$PORT" | xargs kill -9
        echo "  Killed process on port $PORT."
        sleep 1
    else
        exit 1
    fi
fi

# ── Download model note ───────────────────────────────────────────────────────
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SLUG="${MODEL//\//__}"
if [[ ! -d "$HF_CACHE/hub/models--$MODEL_SLUG" ]]; then
    echo "  ℹ  Model not found in cache. vLLM will download it from Hugging Face."
    echo "     This may take several minutes on first run."
    echo "     Cache location: $HF_CACHE"
    echo ""
fi

# ── Launch vLLM ───────────────────────────────────────────────────────────────
echo "  Starting vLLM server..."
echo "  vLLM endpoint will be ready at:"
echo "    http://$HOST:$PORT/v1"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model                   "$MODEL" \
    --host                    "$HOST" \
    --port                    "$PORT" \
    --tensor-parallel-size    "$TENSOR_PARALLEL" \
    --max-model-len           "$MAX_MODEL_LEN" \
    --gpu-memory-utilization  "$GPU_UTIL" \
    --dtype                   "$DTYPE" \
    --trust-remote-code \
    --served-model-name       "$(basename "$MODEL")"
