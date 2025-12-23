#!/usr/bin/env bash
# 安装gunicorn（如果还没安装）

source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

echo "📦 检查 gunicorn..."
if python -c "import gunicorn" 2>/dev/null; then
    echo "✅ gunicorn 已安装"
    python -c "import gunicorn; print(f'版本: {gunicorn.__version__}')"
else
    echo "⬇️  安装 gunicorn..."
    pip install gunicorn
    echo "✅ 安装完成"
fi

echo ""
echo "🔍 检查依赖..."
python -c "
try:
    import uvicorn.workers
    print('✅ uvicorn.workers 可用')
except ImportError:
    print('❌ uvicorn.workers 不可用')
    print('运行: pip install uvicorn[standard]')
"

echo ""
echo "✅ 准备就绪！现在可以重启服务："
echo "   sudo systemctl restart cosyvoice-stream.service"
