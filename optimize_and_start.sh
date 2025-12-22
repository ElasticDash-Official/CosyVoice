#!/bin/bash
# 🚀 CosyVoice 一键性能优化脚本
# 自动量化模型并启用所有优化

set -e

echo "🚀 CosyVoice 一键性能优化"
echo "======================================"
echo ""

# 配置
MODEL_DIR="/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
QUANTIZED_DIR="${MODEL_DIR}-quantized"

# 检查原始模型是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 错误: 找不到模型目录: $MODEL_DIR"
    echo "请修改脚本中的 MODEL_DIR 变量"
    exit 1
fi

# 步骤 1: 量化模型
if [ -d "$QUANTIZED_DIR" ]; then
    echo "✓ 量化模型已存在: $QUANTIZED_DIR"
    read -p "是否重新量化? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔧 开始量化模型..."
        python quantize_model.py "$MODEL_DIR" "$QUANTIZED_DIR"
    fi
else
    echo "🔧 开始量化模型..."
    python quantize_model.py "$MODEL_DIR" "$QUANTIZED_DIR"
fi

echo ""
echo "======================================"
echo "✅ 优化完成！"
echo "======================================"
echo ""
echo "📝 下一步："
echo ""
echo "方式 1 - 使用环境变量启动（推荐）："
echo "  export COSYVOICE_FP16=true"
echo "  export COSYVOICE_QUANTIZED=true"
echo "  ./start_fast.sh"
echo ""
echo "方式 2 - 直接启动（使用默认配置）："
echo "  ./start_fast.sh"
echo "  # FP16 默认开启，量化需要设置 COSYVOICE_QUANTIZED=true"
echo ""
echo "方式 3 - 测试性能对比："
echo "  python benchmark_quantized.py \\"
echo "    $MODEL_DIR \\"
echo "    $QUANTIZED_DIR"
echo ""
echo "======================================"
echo ""

# 询问是否立即启动
read -p "是否立即启动优化后的服务? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "🚀 启动服务..."
    
    # 停止旧服务
    pkill -f "stream_service.py" 2>/dev/null || true
    sleep 2
    
    # 设置环境变量并启动
    export COSYVOICE_FP16=true
    export COSYVOICE_QUANTIZED=true
    
    echo "✓ FP16: $COSYVOICE_FP16"
    echo "✓ Quantized: $COSYVOICE_QUANTIZED"
    echo ""
    
    ./start_fast.sh
fi
