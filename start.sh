#!/usr/bin/env bash
set -e

# ⏳ 防止 boot 早期启动
sleep 30

# 激活 conda
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

# 明确 GPU（避免 systemd 环境不一致）
export CUDA_VISIBLE_DEVICES=0

cd /home/ec2-user/CosyVoice

exec python stream_service.py
