#!/usr/bin/env bash
# 检查量化模型是否存在

MODEL_DIR="/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
QUANTIZED_DIR="${MODEL_DIR}-quantized"

echo "🔍 检查模型目录..."
echo ""

if [ -d "$QUANTIZED_DIR" ]; then
    echo "✅ 量化模型已存在: $QUANTIZED_DIR"
    echo ""
    echo "📁 目录内容:"
    ls -lh "$QUANTIZED_DIR"
    echo ""
    echo "💾 目录大小:"
    du -sh "$QUANTIZED_DIR"
    echo ""
    echo "⚡ 建议: 服务将自动使用量化模型"
else
    echo "❌ 量化模型不存在: $QUANTIZED_DIR"
    echo ""
    echo "📦 原始模型:"
    if [ -d "$MODEL_DIR" ]; then
        echo "✅ 存在: $MODEL_DIR"
        du -sh "$MODEL_DIR"
        echo ""
        echo "🔧 创建量化模型的步骤:"
        echo "1. cd /home/ec2-user/CosyVoice"
        echo "2. conda activate cosyvoice"
        echo "3. python quantize_model.py --model_dir '$MODEL_DIR' --output_dir '$QUANTIZED_DIR'"
        echo ""
        echo "⏱️  预计耗时: 5-10分钟"
        echo "💾 预计节省: ~50% 磁盘空间"
        echo "⚡ 预计提速: 2-3x 推理速度"
    else
        echo "❌ 原始模型也不存在: $MODEL_DIR"
    fi
fi

echo ""
echo "🔄 当前服务状态:"
systemctl status cosyvoice-stream.service --no-pager -l | head -20
