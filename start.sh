#!/usr/bin/env bash
set -o pipefail

UPTIME_SEC=$(awk '{print int($1)}' /proc/uptime)
if [ "$UPTIME_SEC" -lt 60 ]; then
  sleep 30
fi

source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NO_MMAP=1
export MALLOC_ARENA_MAX=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export COSYVOICE_FP16=true
export COSYVOICE_QUANTIZED=true
export COSYVOICE_COMPILE=false
export CUDA_LAUNCH_BLOCKING=0

cd /home/ec2-user/CosyVoice

# gunicorn 多 worker 模式：3 个 worker 并行处理请求（32GB 内存足够）
exec gunicorn stream_service:app \
  --bind 0.0.0.0:50000 \
  --workers 3 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --worker-connections 1000 \
  --max-requests 100 \
  --max-requests-jitter 10 \
  --access-logfile - \
  --error-logfile - \
  --log-level warning