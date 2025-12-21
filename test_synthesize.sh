#!/bin/bash
# CosyVoice2-0.5B 测试脚本

echo "======================================"
echo "CosyVoice2 语音合成测试"
echo "======================================"

# 测试1: 使用默认的餐馆店员instruction
echo ""
echo "测试1: 使用默认餐馆店员场景"
curl -X POST http://localhost:50000/synthesize \
  -F "text=您好,欢迎光临!请问您想吃点什么?" \
  --output test1_default_restaurant.wav

echo "✓ 输出文件: test1_default_restaurant.wav"

# 测试2: 自定义instruction - 四川话
echo ""
echo "测试2: 使用自定义instruction(四川话)"
curl -X POST http://localhost:50000/synthesize \
  -F "text=收到好友从远方寄来的生日礼物,那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐" \
  -F "instruction=用四川话说这句话<|endofprompt|>" \
  --output test2_sichuan.wav

echo "✓ 输出文件: test2_sichuan.wav"

# 测试3: 自定义instruction - 广东话
echo ""
echo "测试3: 使用自定义instruction(广东话)"
curl -X POST http://localhost:50000/synthesize \
  -F "text=今天天气真好,我们一起去喝茶吧" \
  -F "instruction=用广东话说这句话<|endofprompt|>" \
  --output test3_cantonese.wav

echo "✓ 输出文件: test3_cantonese.wav"

# 测试4: 自定义instruction - 可爱女生
echo ""
echo "测试4: 自定义instruction(可爱女生)"
curl -X POST http://localhost:50000/synthesize \
  -F "text=哇,这个蛋糕看起来好好吃呀!" \
  -F "instruction=你是一位可爱活泼的女生,说话声音甜美,语气俏皮可爱。<|endofprompt|>" \
  --output test4_cute_girl.wav

echo "✓ 输出文件: test4_cute_girl.wav"

# 测试5: 自定义instruction - 专业播音员
echo ""
echo "测试5: 自定义instruction(专业播音员)"
curl -X POST http://localhost:50000/synthesize \
  -F "text=各位观众朋友大家好,欢迎收看今天的新闻联播" \
  -F "instruction=你是一位专业的新闻播音员,声音清晰标准,语速适中,语气严肃庄重。<|endofprompt|>" \
  --output test5_broadcaster.wav

echo "✓ 输出文件: test5_broadcaster.wav"

echo ""
echo "======================================"
echo "所有测试完成!"
echo "======================================"
