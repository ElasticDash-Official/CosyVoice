# 🔥 关键修复: 声音克隆不匹配问题

## 问题根源

**之前使用了错误的推理模式!**

根据 CosyVoice2 官方示例 (`example.py`):
- ❌ **`inference_cross_lingual`** 是用于 "**细粒度控制**" (fine grained control)
  - 用于添加特殊标记: `[laughter]`, `[breath]` 等
  - **不是用来做声音克隆的!**

- ✅ **`inference_zero_shot`** 才是用于 "**声音克隆**"
  - 需要提供 `prompt_text` (音频文件的文字内容)
  - 会准确克隆 prompt_wav 的声音

## 官方示例证据

来自 `example.py` 第 35-57 行 (CosyVoice2 示例):

```python
def cosyvoice2_example():
    """ CosyVoice2 Usage """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

    # ✅ zero_shot usage - 用于声音克隆
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        '收到好友从远方寄来的生日礼物...',
        '希望你以后能够做的比我还好呦。',  # ← prompt_text (音频文件的文字内容)
        './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # ❌ cross_lingual usage - 用于细粒度控制 (NOT voice cloning!)
    # 注释写的很清楚: "fine grained control"
    for i, j in enumerate(cosyvoice.inference_cross_lingual(
        '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来...',  # ← 包含特殊标记
        './asset/zero_shot_prompt.wav')):
        torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # ✅ instruct2 usage - 用于指令控制风格
    for i, j in enumerate(cosyvoice.inference_instruct2(
        '收到好友从远方寄来的生日礼物...',
        '用四川话说这句话<|endofprompt|>',  # ← instruction
        './asset/zero_shot_prompt.wav')):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

## 修复内容

### 修改文件: `stream_service.py`

#### 1. 添加默认 prompt_text (第 30-31 行)

```python
# 默认的prompt音频文件路径和对应文本
default_prompt_wav = "./asset/zero_shot_prompt.wav"
default_prompt_text = "希望你以后能够做的比我还好呦。"  # zero_shot_prompt.wav 的文字内容
```

#### 2. 更新推理模式选择逻辑 (第 157-174 行)

**之前 (错误):**
```python
else:
    # 无 instruction - 使用 cross_lingual 模式 (纯声音克隆)
    inference_method = lambda: cosyvoice.inference_cross_lingual(
        text,
        temp_wav_path,
        stream=True
    )
```

**现在 (正确):**
```python
else:
    # 无 instruction - 使用 zero_shot 模式 (纯声音克隆)
    # 需要 prompt_text (音频文件的文字内容)
    actual_prompt_text = prompt_text if prompt_text else default_prompt_text

    inference_method = lambda: cosyvoice.inference_zero_shot(
        text,
        actual_prompt_text,  # ← 关键: 提供音频文件的文字内容
        temp_wav_path,
        stream=True
    )
```

## CosyVoice2 推理模式总结

| 推理模式 | 用途 | 参数 | 说明 |
|---------|------|------|------|
| **`inference_zero_shot`** | 声音克隆 | `text`, `prompt_text`, `prompt_wav` | 需要音频的文字内容,准确克隆声音 |
| **`inference_instruct2`** | 指令控制 | `text`, `instruction`, `prompt_wav` | 用 instruction 控制说话风格 |
| **`inference_cross_lingual`** | 细粒度控制 | `text`, `prompt_wav` | 用于 `[laughter]`, `[breath]` 等标记 |

## 为什么之前声音不匹配?

1. **使用了 `cross_lingual` 模式**
   - 这个模式不是为声音克隆设计的
   - 它主要处理特殊标记,不保证声音相似度

2. **没有提供 `prompt_text`**
   - `zero_shot` 模式需要知道音频里说的是什么
   - prompt_text 帮助模型更准确地提取声音特征

## 部署修复

```bash
# 1. 提交修改
git add stream_service.py CRITICAL_FIX.md
git commit -m "Fix voice cloning: use inference_zero_shot instead of cross_lingual

- CosyVoice2's inference_cross_lingual is for fine-grained control, not voice cloning
- Use inference_zero_shot with prompt_text for accurate voice cloning
- Add default_prompt_text for default audio file"

# 2. 推送到远程
git push origin main

# 3. 服务器部署
ssh ec2-user@server
cd /home/ec2-user/CosyVoice
git pull origin main
sudo systemctl restart stream_service

# 4. 查看日志 - 应该看到 "Mode: ZERO_SHOT"
journalctl -u stream_service -f
```

## 测试验证

发送测试请求:

```bash
# 不提供 instruction - 应该使用 ZERO_SHOT 模式克隆声音
curl -X POST "http://localhost:50000/synthesize" \
  -F "text=这是测试,声音应该匹配 zero_shot_prompt.wav" \
  --output test_zero_shot.wav
```

日志应该显示:

```
✓ Using DEFAULT prompt_wav: .../asset/zero_shot_prompt.wav
  - File size: 333824 bytes

✓ Verified prompt_wav audio properties:
  - Sample rate: 16000 Hz
  - Duration: 10.43 seconds

[CosyVoice2] Mode: ZERO_SHOT (voice cloning)
  → Using inference_zero_shot for voice cloning
  - Text: '这是测试,声音应该匹配 zero_shot_prompt.wav' (len=24)
  - Prompt text: '希望你以后能够做的比我还好呦。'
  - Voice reference: zero_shot_prompt.wav
  - Voice will MATCH the prompt audio
```

## 重要提示

### 使用自定义音频时

如果要上传自己的音频文件,**必须提供 prompt_text**:

```python
data = {
    "text": "你好,欢迎光临",
    "prompt_text": "这是我录制的音频内容"  # ← 必须提供!
}
files = {
    "prompt_wav": open("my_voice.wav", "rb")
}
response = requests.post(url, data=data, files=files, stream=True)
```

### 使用 instruction 控制风格时

```python
data = {
    "text": "你好,欢迎光临",
    "instruction": "你是一位专业的播音员<|endofprompt|>"
    # 不需要 prompt_text,会自动使用 instruct2 模式
}
response = requests.post(url, data=data, stream=True)
```

## 问题解决时间线

1. **问题**: 生成的声音和 prompt 音频完全不匹配
2. **调查**: 查看服务日志,确认使用了 CROSS_LINGUAL 模式
3. **发现**: 检查 `example.py` 发现 cross_lingual 是用于细粒度控制,不是声音克隆
4. **修复**: 改为使用 inference_zero_shot 并提供 prompt_text
5. **验证**: 部署后测试声音克隆效果

---

**结论**: 这是一个使用了错误 API 的问题。CosyVoice2 的声音克隆需要使用 `inference_zero_shot` 模式,并提供 `prompt_text` 参数。
