#!/usr/bin/env bash

# 不要 set -e，避免 systemd 误判失败
set -o pipefail

# ⏳ 防止 boot 早期启动
sleep 30

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

# ❗ 核心：使用 gunicorn 多 worker 实现真正并行
# -w 4: 4个worker进程 (根据GPU内存调整)
# -k uvicorn.workers.UvicornWorker: 使用异步worker
# --timeout 300: 增加超时时间
# --worker-connections 1000: 每个worker的连接数
exec gunicorn stream_service:app \
  --bind 0.0.0.0:50000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --worker-connections 1000 \
  --access-logfile - \
  --error-logfile - \
  --log-level warning
