#!/usr/bin/env bash
set -euo pipefail

# Simple starter for async_cosyvoice FastAPI + vLLM
# Usage:
#   ./start_vllm_async.sh [MODEL_DIR] [PORT]
# Defaults:
#   MODEL_DIR=pretrained_models/CosyVoice2-0.5B
#   PORT=8022

MODEL_DIR="${1:-pretrained_models/CosyVoice2-0.5B}"
PORT="${2:-50000}"

echo "[vLLM] Starting Async CosyVoice FastAPI"
echo "Model dir: ${MODEL_DIR}"
echo "Port: ${PORT}"

# Optional: activate conda if available (online env)
if command -v conda >/dev/null 2>&1; then
  # Try common path; ignore if missing
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  # Activate env if exists; ignore errors
  if conda env list | awk '{print $1}' | grep -q "^cosyvoice$"; then
    conda activate cosyvoice || true
  fi
fi

# vLLM API v1
export VLLM_USE_V1=1

# Ensure we run from repo root
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Install async_cosyvoice requirements (Linux server)
echo "[vLLM] Installing requirements (this may take several minutes)..."
pip install -r async_cosyvoice/requirements.txt

# Register CosyVoice2 model with vLLM
python async_cosyvoice/copy_model_to_vllm.py || {
  echo "[vLLM] Failed to patch vLLM registry; ensure vllm==0.7.3 is installed." >&2
  exit 1
}

# Copy extra files into model dir (cosyvoice2.yaml, campplus.onnx, etc.)
if [ -d "async_cosyvoice/CosyVoice2-0.5B" ]; then
  echo "[vLLM] Copying CosyVoice2-0.5B extras into ${MODEL_DIR}..."
  mkdir -p "${MODEL_DIR}"
  cp -n async_cosyvoice/CosyVoice2-0.5B/* "${MODEL_DIR}/" || true
else
  echo "[vLLM] Warning: async_cosyvoice/CosyVoice2-0.5B not found; skip copying extras."
fi

# Sanity check model files
REQUIRED_FILES=(
  "${MODEL_DIR}/cosyvoice2.yaml"
  "${MODEL_DIR}/flow.pt"
  "${MODEL_DIR}/hift.pt"
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[vLLM] Missing required file: $f"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "[vLLM] Please download CosyVoice2-0.5B model into ${MODEL_DIR} (see async_cosyvoice README)." >&2
  exit 2
fi

# Start FastAPI compat server (JIT + TRT + FP16) on same port/routes as stream_service
cd async_cosyvoice/runtime/fastapi
python compat_stream_service.py \
  --model_dir "${REPO_ROOT}/${MODEL_DIR}" \
  --load_jit \
  --load_trt \
  --fp16 \
  --port "${PORT}"
