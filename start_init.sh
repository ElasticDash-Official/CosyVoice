#!/usr/bin/env bash
set -e

echo "=== Initializing CosyVoice2 environment ==="

# Clean caches
echo "Cleaning model caches..."
rm -rf ~/.cache/modelscope 2>/dev/null || true
rm -rf ~/.cache/huggingface 2>/dev/null || true
rm -rf ~/.cache/torch 2>/dev/null || true
rm -rf /tmp/modelscope* 2>/dev/null || true
rm -rf /tmp/huggingface* 2>/dev/null || true

source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

# Disable DeepSpeed
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

export MODELSCOPE_CACHE=~/.cache/modelscope
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

cd /home/ec2-user/CosyVoice

echo "Pre-downloading wetext model (this may take a few minutes on first run)..."
python3 << 'EOF'
import sys
import os
os.environ['DS_BUILD_AIO'] = '0'
os.environ['DS_BUILD_SPARSE_ATTN'] = '0'
os.environ['DS_BUILD_SPARSE_ALLREDUCE'] = '0'
os.environ['DS_BUILD_UTILS'] = '0'
os.environ['DS_BUILD_FUSED_LAMB'] = '0'

try:
    from wetext import ZhNormalizer
    print("✓ wetext model downloaded successfully")
except Exception as e:
    print(f"Warning: wetext pre-download failed: {e}")
    print("  Model will download on first request")

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print("✓ CosyVoice2 import successful")
except Exception as e:
    print(f"Warning: CosyVoice2 import test failed: {e}")

print("✓ Initialization complete")
EOF

echo ""
echo "=== Starting CosyVoice2 service ==="
echo "Service will run on http://0.0.0.0:50000"
echo "API endpoint: /synthesize (POST)"
echo ""

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NO_MMAP=1
export MALLOC_ARENA_MAX=4
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDA_MEMORY_FRACTION=0.45
export COSYVOICE_FP16=true
export COSYVOICE_QUANTIZED=true
export COSYVOICE_COMPILE=false
export CUDA_LAUNCH_BLOCKING=0

exec gunicorn stream_service:app \
  --bind 0.0.0.0:50000 \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --worker-connections 1000 \
  --max-requests 100 \
  --max-requests-jitter 10 \
  --access-logfile - \
  --error-logfile - \
  --log-level warning
