# 统一的 /synthesize 接口使用说明

## 概述

现在 `/synthesize` 接口已经统一支持所有 CosyVoice 模型,会自动根据加载的模型类型选择正确的推理方式。

当前配置: **CosyVoice2-0.5B** (使用 instruction 模式)

## 默认配置

- **默认 instruction**: "你是一位热情友好的餐馆店员,说话温柔亲切,语气礼貌专业。<|endofprompt|>"
- **默认 prompt_wav**: `./asset/zero_shot_prompt.wav`

## 快速开始

### 1. 使用默认餐馆店员场景

```bash
curl -X POST http://localhost:50000/synthesize \
  -F "text=您好,欢迎光临!请问您想吃点什么?" \
  --output output.wav
```

不需要指定任何参数,会自动使用默认的餐馆店员 instruction。

### 2. 自定义 instruction

```bash
curl -X POST http://localhost:50000/synthesize \
  -F "text=收到好友从远方寄来的生日礼物,那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐" \
  -F "instruction=用四川话说这句话<|endofprompt|>" \
  --output output.wav
```

## Instruction 示例

### 方言类
- `"用四川话说这句话<|endofprompt|>"`
- `"用广东话说这句话<|endofprompt|>"`
- `"用东北话说这句话<|endofprompt|>"`
- `"用上海话说这句话<|endofprompt|>"`

### 角色/风格类
- `"你是一位热情友好的餐馆店员,说话温柔亲切,语气礼貌专业。<|endofprompt|>"`
- `"你是一位可爱活泼的女生,说话声音甜美,语气俏皮可爱。<|endofprompt|>"`
- `"你是一位专业的新闻播音员,声音清晰标准,语速适中,语气严肃庄重。<|endofprompt|>"`
- `"你是一位温柔耐心的客服人员,说话语气温和亲切,充满同理心。<|endofprompt|>"`
- `"你是一位儿童故事讲述者,声音温柔生动,富有感染力,语气充满童趣。<|endofprompt|>"`

## Python 示例

```python
import requests

# 使用默认餐馆店员场景
response = requests.post(
    "http://localhost:50000/synthesize",
    data={"text": "您好,欢迎光临!请问您需要什么帮助?"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# 使用自定义 instruction
response = requests.post(
    "http://localhost:50000/synthesize",
    data={
        "text": "今天天气真好啊!",
        "instruction": "你是一位可爱活泼的女生,说话声音甜美,语气俏皮可爱。<|endofprompt|>"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 测试脚本

我们提供了两个测试脚本:

### Bash 版本
```bash
chmod +x test_synthesize.sh
./test_synthesize.sh
```

### Python 版本
```bash
python test_synthesize.py
```

这些脚本会测试多种不同的 instruction,生成多个示例音频文件。

## 参数说明

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| text | ✓ | 要合成的文本 | "您好,欢迎光临!" |
| instruction | ✗ | 语言/风格指令 | "用四川话说这句话<|endofprompt|>" |
| prompt_wav | ✗ | 参考音频文件 | 上传音频文件 |
| prompt_text | ✗ | 参考文本 | "希望你以后能够做的比我还好呦。" |

注意:
- 如果不指定 `instruction`,会使用默认的餐馆店员 instruction
- 如果不上传 `prompt_wav`,会使用默认的音频文件
- 所有 instruction 必须以 `<|endofprompt|>` 结尾

## 查看模型信息

```bash
curl http://localhost:50000/model_info
```

返回:
```json
{
  "model_type": "CosyVoice2",
  "model_dir": "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B",
  "endpoint": "/synthesize_instruct",
  "usage": "Use /synthesize_instruct endpoint with 'instruction' and 'prompt_wav' parameters"
}
```

## 切换模型

编辑 `stream_service.py` 的第23行:

```python
# 切换到 CosyVoice 第一代
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice-300M-SFT"

# 或切换到 CosyVoice3
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
```

重启服务后,接口会自动适配新的模型。

## 注意事项

1. **CosyVoice 第一代**: 使用 `speaker` 参数而不是 `instruction`
2. **CosyVoice2**: instruction 格式为 `"描述<|endofprompt|>"`
3. **CosyVoice3**: instruction 格式为 `"You are a helpful assistant. 描述<|endofprompt|>"`
4. 接口会自动检测模型类型并使用正确的推理方法
5. 默认音频文件路径: `./asset/zero_shot_prompt.wav`
