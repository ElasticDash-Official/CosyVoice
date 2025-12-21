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

cd /home/ec2-user/CosyVoice

# ❗ 核心：stdout / stderr 都交给 systemd journald 管理
# 不再手动写日志文件，避免无限增长
exec python stream_service.py
