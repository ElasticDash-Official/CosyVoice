# CosyVoice2 部署和验证指南

## 📋 更新内容

### 1. 修复 torchcodec 依赖问题
- 文件: `cosyvoice/utils/file_utils.py`
- 改动: 直接使用 soundfile 读取音频,避免 torchcodec/FFmpeg 依赖

### 2. 增强日志输出
- 文件: `stream_service.py`
- 改动: 添加详细的音频文件验证和使用日志

### 3. 包含默认音频文件
- 文件: `asset/zero_shot_prompt.wav` (326 KB)
- 文件: `asset/cross_lingual_prompt.wav` (592 KB)
- 作用: 作为 CosyVoice2 的基底音色参考

---

## 🚀 部署步骤

### 步骤 1: 本地验证

```bash
# 验证音频文件
python3 verify_audio_base.py

# 应该看到:
# ✓ 文件存在: /path/to/asset/zero_shot_prompt.wav
# ✓ 音频文件属性: 采样率: 16000 Hz, 时长: X 秒, ...
```

### 步骤 2: 提交代码

```bash
# 查看修改
git status
git diff

# 提交修改
git add cosyvoice/utils/file_utils.py stream_service.py asset/*.wav
git commit -m "Fix torchcodec issue and add audio file verification

- Use soundfile directly to avoid torchcodec/FFmpeg dependency
- Add detailed logging for audio file usage verification
- Include default prompt audio files for CosyVoice2 voice reference"

# 推送到远程
git push origin main
```

### 步骤 3: 服务器部署

```bash
# SSH 到服务器
ssh ec2-user@your-server

# 进入项目目录
cd /home/ec2-user/CosyVoice

# 拉取最新代码
git pull origin main

# 验证文件存在
ls -lh asset/zero_shot_prompt.wav
python3 verify_audio_base.py

# 重启服务
sudo systemctl restart stream_service

# 实时查看日志
journalctl -u stream_service -f
```

### 步骤 4: 验证日志输出

重启后,当有请求时,你应该看到类似的日志:

```
✓ Using DEFAULT prompt_wav: /home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav
  - File size: 333824 bytes (326.0 KB)
  - This audio will be used as the BASE VOICE for synthesis

✓ Verified prompt_wav audio properties:
  - Sample rate: 16000 Hz
  - Duration: 10.43 seconds
  - Channels: 1
  - Format: WAV

→ Using inference_instruct2 (instruction + voice reference)
  - Text: '您好，我们这里是...' (len=43)
  - Instruction: '你是一位热情友好的餐馆店员...'
  - Voice reference: zero_shot_prompt.wav
```

---

## 🔍 故障排查

### 问题 1: 音频文件不存在

**症状:**
```
✗ No prompt_wav provided and default file not found!
  - Expected path: /home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav
```

**解决:**
```bash
# 检查文件是否存在
ls -lh /home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav

# 如果不存在,检查 git 状态
cd /home/ec2-user/CosyVoice
git status
git pull

# 如果仍然不存在,从本地上传
scp asset/zero_shot_prompt.wav ec2-user@server:/home/ec2-user/CosyVoice/asset/
```

### 问题 2: torchcodec 错误仍然出现

**症状:**
```
ImportError: TorchCodec is required for load_with_torchcodec
```

**解决:**
```bash
# 确认代码已更新
cd /home/ec2-user/CosyVoice
git log --oneline -1
# 应该看到最新的 commit: "Fix torchcodec issue..."

# 如果没有更新,拉取代码
git pull origin main

# 重启服务
sudo systemctl restart stream_service
```

### 问题 3: 生成的语音不像预期的声音

**原因:**
- `instruction` 控制说话风格
- `zero_shot_prompt.wav` 提供基础音色

**验证:**
1. 检查日志确认使用了正确的音频文件
2. 尝试上传自定义的 prompt_wav 来改变音色
3. 修改 instruction 来改变说话风格

**测试不同音色:**
```bash
# 使用默认音频文件
curl -X POST "http://server:50000/synthesize" \
  -F "text=你好,欢迎光临" \
  -F "instruction=你是一位温柔的客服人员。<|endofprompt|>" \
  --output output1.wav

# 使用自定义音频文件
curl -X POST "http://server:50000/synthesize" \
  -F "text=你好,欢迎光临" \
  -F "instruction=你是一位温柔的客服人员。<|endofprompt|>" \
  -F "prompt_wav=@my_voice.wav" \
  --output output2.wav
```

---

## 📝 关键要点

1. **CosyVoice2 需要音频文件作为音色参考**
   - `instruction` = 说话风格 (温柔/严肃/活泼)
   - `prompt_wav` = 声音音色 (音调/音质/声线)

2. **默认音频文件**
   - 路径: `./asset/zero_shot_prompt.wav`
   - 如果不提供 prompt_wav,自动使用这个文件

3. **日志验证**
   - 查看日志确认音频文件被正确加载
   - 检查音频属性 (采样率、时长、声道)

4. **自定义音色**
   - 上传自己的 WAV 文件来改变声音
   - 音频要求: 16kHz 或 22050Hz, 单声道, 3-10 秒清晰语音

---

## 🎯 下一步

1. 部署到服务器
2. 查看日志验证音频文件被正确使用
3. 测试语音合成
4. (可选) 准备不同的 prompt_wav 文件用于不同音色
