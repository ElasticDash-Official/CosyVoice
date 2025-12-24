#!/usr/bin/env bash
set -o pipefail

echo "=== Deep cleaning all model caches ==="
rm -rf ~/.cache/modelscope 2>/dev/null || true
rm -rf ~/.cache/huggingface 2>/dev/null || true
rm -rf ~/.cache/torch 2>/dev/null || true
rm -rf /tmp/modelscope* 2>/dev/null || true
rm -rf /tmp/huggingface* 2>/dev/null || true

echo "Cache cleaned successfully"

UPTIME_SEC=$(awk '{print int($1)}' /proc/uptime)
if [ "$UPTIME_SEC" -lt 60 ]; then
  sleep 30
fi

source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

# Completely disable DeepSpeed
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_SPARSE_ALLREDUCE=0
export DS_BUILD_UTILS=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_QUANTIZER=0
export DS_BUILD_RAGGED_DEVICE_OPS=0
export DS_BUILD_INFERENCE_CORE_OPS=0

# Allow modelscope to download
export MODELSCOPE_CACHE=~/.cache/modelscope
export HF_HOME=~/.cache/huggingface

# GPU and memory optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NO_MMAP=1
export MALLOC_ARENA_MAX=4
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDA_MEMORY_FRACTION=0.45
export COSYVOICE_FP16=true
export COSYVOICE_QUANTIZED=true
export COSYVOICE_COMPILE=false
export CUDA_LAUNCH_BLOCKING=0

cd /home/ec2-user/CosyVoice

echo "Starting CosyVoice2 service on port 50000..."

exec gunicorn stream_service:app \
  --bind 0.0.0.0:50000 \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --worker-connections 1000 \
  --max-requests 100 \
  --max-requests-jitter 10 \
  --access-logfile - \
  --error-logfile - \
  --log-level warning
