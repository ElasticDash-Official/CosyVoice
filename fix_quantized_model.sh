#!/bin/bash
# 快速修复：复制缺失的子目录到量化模型

set -e

ORIGINAL="/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B"
QUANTIZED="/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B-quantized"

echo "🔧 修复量化模型 - 复制缺失的目录"
echo "======================================"

if [ ! -d "$ORIGINAL" ]; then
    echo "❌ 原始模型目录不存在: $ORIGINAL"
    exit 1
fi

if [ ! -d "$QUANTIZED" ]; then
    echo "❌ 量化模型目录不存在: $QUANTIZED"
    exit 1
fi

# 复制所有子目录
echo "📋 检查并复制子目录..."
for item in "$ORIGINAL"/*; do
    if [ -d "$item" ]; then
        dirname=$(basename "$item")
        if [ ! -d "$QUANTIZED/$dirname" ]; then
            echo "  ✓ 复制目录: $dirname"
            cp -r "$item" "$QUANTIZED/"
        else
            echo "  ⏭️  已存在: $dirname"
        fi
    fi
done

# 复制配置文件（如果缺失）
echo ""
echo "📋 检查配置文件..."
for file in cosyvoice.yaml configuration.json config.json; do
    if [ -f "$ORIGINAL/$file" ] && [ ! -f "$QUANTIZED/$file" ]; then
        echo "  ✓ 复制文件: $file"
        cp "$ORIGINAL/$file" "$QUANTIZED/"
    fi
done

echo ""
echo "======================================"
echo "✅ 修复完成！"
echo "======================================"
echo ""
echo "现在可以启动服务："
echo "  export COSYVOICE_FP16=true"
echo "  export COSYVOICE_QUANTIZED=true"
echo "  python stream_service.py"
