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

# 单进程 uvicorn，避免主/子进程各持一份模型
exec uvicorn stream_service:app \
  --host 0.0.0.0 \
  --port 50000 \
  --timeout-keep-alive 300 \
  --log-level warning