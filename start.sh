#!/usr/bin/env bash

# 不要 set -e，避免 systemd 误判失败
set -o pipefail

# ⏳ 防止 boot 早期启动：仅在开机 < 60 秒时等待
UPTIME_SEC=$(awk '{print int($1)}' /proc/uptime)
if [ "$UPTIME_SEC" -lt 60 ]; then
  sleep 30
fi

# 激活 conda
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

# 明确 GPU
export CUDA_VISIBLE_DEVICES=0

# PyTorch / malloc 安全参数
export PYTORCH_NO_MMAP=1
export MALLOC_ARENA_MAX=4

# 性能优化环境变量
export COSYVOICE_FP16=true
export COSYVOICE_QUANTIZED=true
export COSYVOICE_COMPILE=false  # 首次推理会慢，可选启用

# CUDA优化
export CUDA_LAUNCH_BLOCKING=0

cd /home/ec2-user/CosyVoice

exec gunicorn stream_service:app \
  --bind 0.0.0.0:50000 \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --worker-connections 1000 \
  --access-logfile - \
  --error-logfile - \
  --log-level warning