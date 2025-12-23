#!/usr/bin/env bash
# 检查多worker是否正常工作

echo "🔍 检查 CosyVoice 服务状态..."
echo ""

# 检查进程数
WORKER_COUNT=$(ps aux | grep -E "gunicorn.*stream_service" | grep -v grep | wc -l)

if [ "$WORKER_COUNT" -eq 0 ]; then
    echo "❌ 服务未运行"
    echo ""
    echo "启动服务："
    echo "  sudo systemctl start stream_service"
    exit 1
fi

echo "✅ 检测到 $WORKER_COUNT 个进程"
echo ""

# 显示进程详情
echo "📊 进程列表："
ps aux | grep -E "gunicorn.*stream_service" | grep -v grep | awk '{printf "  PID: %-6s CPU: %-5s MEM: %-5s CMD: %s\n", $2, $3"%", $4"%", substr($0, index($0,$11))}'

echo ""

# 检查端口
echo "🌐 端口监听："
if command -v netstat &> /dev/null; then
    netstat -tlnp 2>/dev/null | grep :50000 || echo "  (需要root权限查看详情)"
elif command -v ss &> /dev/null; then
    ss -tlnp 2>/dev/null | grep :50000 || echo "  (需要root权限查看详情)"
fi

echo ""

# 最近的日志
echo "📝 最近日志 (最后20行)："
sudo journalctl -u cosyvoice-stream.service -n 20 --no-pager | tail -20

echo ""
echo "💡 提示："
echo "  - Master进程: 负责管理worker"
echo "  - Worker进程: 实际处理请求 (应该有4个)"
echo "  - 总进程数 = 1个master + 4个worker = 5个"

# 测试并发
echo ""
echo "🧪 测试并发处理？(y/n)"
read -r answer
if [ "$answer" = "y" ]; then
    echo "发送3个并发请求..."
    for i in {1..3}; do
        curl -s -X POST http://localhost:50000/synthesize \
          -F "text=测试文本$i" \
          -o /dev/null &
    done
    sleep 2
    echo "✅ 请求已发送，检查日志中的时间戳判断是否并行"
fi
