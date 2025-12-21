# CosyVoice Stream Service API 使用说明

## 概述

此服务根据不同的CosyVoice模型版本自动适配API接口:
- **CosyVoice (第一代)**: 使用 speaker_id
- **CosyVoice2**: 使用 instruction + prompt_wav
- **CosyVoice3**: 使用 instruction + prompt_wav

## 查看模型信息

首先查看当前加载的模型类型:

```bash
curl http://localhost:50000/model_info
```

响应示例:
```json
{
  "model_type": "CosyVoice2",
  "model_dir": "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B",
  "endpoint": "/synthesize_instruct",
  "usage": "Use /synthesize_instruct endpoint with 'instruction' and 'prompt_wav' parameters"
}
```

## 1. CosyVoice 第一代 API (使用 speaker_id)

### 查看可用的speakers

```bash
curl http://localhost:50000/speakers
```

### 合成语音

```bash
curl -X POST http://localhost:50000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，我是通义生成式语音大模型",
    "speaker": "中文女"
  }' \
  --output output.wav
```

## 2. CosyVoice2 API (使用 instruction)

### 方式1: 使用 instruct2 模式 (推荐)

用instruction描述语言和风格需求:

```bash
curl -X POST http://localhost:50000/synthesize_instruct \
  -F "text=收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  -F "instruction=用四川话说这句话<|endofprompt|>" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  --output output.wav
```

**Instruction 示例:**
- `"用四川话说这句话<|endofprompt|>"`
- `"用广东话说这句话<|endofprompt|>"`
- `"用英文说这句话<|endofprompt|>"`

### 方式2: 使用 zero_shot 模式

用prompt_text提供参考文本:

```bash
curl -X POST http://localhost:50000/synthesize_instruct \
  -F "text=收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  --output output.wav
```

### 方式3: 使用 cross_lingual 模式 (精细控制)

仅使用prompt_wav进行细粒度控制,text中可以包含特殊标记如 `[laughter]`, `[breath]`:

```bash
curl -X POST http://localhost:50000/synthesize_instruct \
  -F "text=在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  --output output.wav
```

## 3. CosyVoice3 API (使用 instruction)

### 使用 instruct2 模式

CosyVoice3的instruction格式需要包含系统提示:

```bash
curl -X POST http://localhost:50000/synthesize_instruct \
  -F "text=好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。" \
  -F "instruction=You are a helpful assistant. 请用广东话表达。<|endofprompt|>" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  --output output.wav
```

**CosyVoice3 Instruction 示例:**
- `"You are a helpful assistant. 请用广东话表达。<|endofprompt|>"`
- `"You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>"`
- `"You are a helpful assistant. 请用英文说。<|endofprompt|>"`

### 使用 zero_shot 模式

CosyVoice3的prompt_text格式也需要包含系统提示:

```bash
curl -X POST http://localhost:50000/synthesize_instruct \
  -F "text=八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。" \
  -F "prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  --output output.wav
```

### 使用 cross_lingual 模式 (精细控制)

支持 `[breath]` 等特殊控制标记:

```bash
curl -X POST http://localhost:50000/synthesize_instruct \
  -F "text=You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  --output output.wav
```

## Python 示例

### CosyVoice 第一代

```python
import requests

response = requests.post(
    "http://localhost:50000/synthesize",
    json={
        "text": "你好，我是通义生成式语音大模型",
        "speaker": "中文女"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### CosyVoice2/3

```python
import requests

# 准备文件和数据
files = {
    'prompt_wav': open('./asset/zero_shot_prompt.wav', 'rb')
}
data = {
    'text': '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    'instruction': '用四川话说这句话<|endofprompt|>'
}

response = requests.post(
    "http://localhost:50000/synthesize_instruct",
    files=files,
    data=data
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 健康检查

```bash
curl http://localhost:50000/health
```

响应:
```json
{
  "status": "ok",
  "model_type": "CosyVoice2"
}
```

## 注意事项

1. **模型切换**: 修改 `stream_service.py` 中的 `model_dir` 变量来切换不同的模型
2. **端点选择**: 根据模型类型自动选择正确的端点
   - CosyVoice v1 → `/synthesize`
   - CosyVoice2/3 → `/synthesize_instruct`
3. **音频格式**: prompt_wav需要是有效的音频文件(支持多种格式)
4. **流式传输**: 所有接口都支持流式传输,实时返回音频块
5. **语言支持**:
   - CosyVoice2: 中文各方言、英文等
   - CosyVoice3: 支持更多语言和更精细的控制

## 错误处理

如果使用了错误的端点,服务会返回提示信息:

```bash
# 在CosyVoice2上使用/synthesize会返回:
{
  "detail": "This endpoint is only for CosyVoice v1. Current model is CosyVoice2. Please use /synthesize_instruct instead."
}
```
