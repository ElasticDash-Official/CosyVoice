#!/bin/bash

# CosyVoice 高性能启动脚本
# 使用 uvloop 和优化配置

set -e

echo "🚀 CosyVoice Performance Mode Starting..."

# 检查并安装 uvloop
if ! python3 -c "import uvloop" 2>/dev/null; then
    echo "📦 Installing uvloop for better performance..."
    pip install uvloop -q
fi

# 设置环境变量优化
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 如果有 GPU，设置 CUDA 优化
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, enabling CUDA optimizations..."
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_CUDNN_V8_API_ENABLED=1
fi

# 启动服务（使用 uvloop）
echo "✅ Starting service with optimizations..."
python3 -c "
import uvloop
import asyncio
uvloop.install()

from stream_service import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=50000,
        loop='uvloop',
        workers=1,
        limit_concurrency=20,
        timeout_keep_alive=30,
        backlog=2048,
        log_level='warning',
        access_log=False  # 禁用访问日志进一步提速
    )
"
