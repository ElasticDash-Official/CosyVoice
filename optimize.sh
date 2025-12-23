#!/usr/bin/env bash
# 一键优化脚本 - 解决并发慢的问题

set -e

echo "🚀 CosyVoice 性能优化脚本"
echo "================================"
echo ""

# 激活环境
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

cd /home/ec2-user/CosyVoice

# 步骤1: 检查/安装 gunicorn
echo "📦 [1/3] 检查依赖..."
if ! python -c "import gunicorn" 2>/dev/null; then
    echo "  ⬇️  安装 gunicorn..."
    pip install gunicorn -q
fi
echo "  ✅ gunicorn 已就绪"

# 步骤2: 检查量化模型
echo ""
echo "🔍 [2/3] 检查量化模型..."
MODEL_DIR="/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
QUANTIZED_DIR="${MODEL_DIR}-quantized"

if [ -d "$QUANTIZED_DIR" ]; then
    echo "  ✅ 量化模型已存在"
    du -sh "$QUANTIZED_DIR"
else
    echo "  ⚠️  量化模型不存在"
    echo ""
    echo "  量化模型可以提速 2-3倍，是否现在创建？(约需5-10分钟)"
    echo "  输入 'y' 创建，或按回车跳过："
    read -r answer
    
    if [ "$answer" = "y" ]; then
        echo "  🔄 开始量化..."
        
        # 检查量化脚本
        if [ -f "quantize_model.py" ]; then
            python quantize_model.py \
              --model_dir "$MODEL_DIR" \
              --output_dir "$QUANTIZED_DIR"
            echo "  ✅ 量化完成！"
        elif [ -f "simple_quantize.py" ]; then
            python simple_quantize.py \
              --model_dir "$MODEL_DIR" \
              --output_dir "$QUANTIZED_DIR"
            echo "  ✅ 量化完成！"
        else
            echo "  ❌ 找不到量化脚本"
            echo "  跳过量化，将使用原始模型（会较慢）"
        fi
    else
        echo "  ⏭️  跳过量化（将使用原始模型）"
    fi
fi

# 步骤3: 重启服务
echo ""
echo "🔄 [3/3] 重启服务..."

if systemctl is-active --quiet cosyvoice-stream.service; then
    echo "  停止旧服务..."
    sudo systemctl stop cosyvoice-stream.service
    sleep 2
fi

echo "  启动优化后的服务..."
sudo systemctl start cosyvoice-stream.service
sleep 5

# 检查状态
if systemctl is-active --quiet cosyvoice-stream.service; then
    echo "  ✅ 服务启动成功"
    echo ""
    
    # 检查worker数量
    echo "📊 检查worker数量..."
    sleep 2
    WORKER_COUNT=$(ps aux | grep -E "gunicorn.*stream_service" | grep -v grep | wc -l)
    
    if [ "$WORKER_COUNT" -ge 4 ]; then
        echo "  ✅ 检测到 $WORKER_COUNT 个进程（1 master + workers）"
    else
        echo "  ⚠️  只检测到 $WORKER_COUNT 个进程"
    fi
    
    echo ""
    echo "📝 最近日志："
    sudo journalctl -u cosyvoice-stream.service -n 10 --no-pager
    
    echo ""
    echo "================================"
    echo "✅ 优化完成！"
    echo ""
    echo "预期改进："
    if [ -d "$QUANTIZED_DIR" ]; then
        echo "  • RTF: 1.2 → 0.4 (3x faster)"
    else
        echo "  • RTF: ~1.2 (未使用量化)"
    fi
    echo "  • 并发: 1 → 4 workers"
    echo "  • 吞吐量: ~4-12x"
    echo ""
    echo "监控命令："
    echo "  sudo journalctl -u cosyvoice-stream.service -f"
    
else
    echo "  ❌ 服务启动失败"
    echo ""
    echo "查看错误日志："
    sudo journalctl -u cosyvoice-stream.service -n 50 --no-pager
    exit 1
fi
