#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化测试：对比 stream=True 和 stream=False 的音色效果
"""
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import soundfile as sf
import os

model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B-2512'
cosyvoice = AutoModel(model_dir=model_dir)

text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
prompt_wav = './asset/zero_shot_prompt.wav'

print(f"Testing with:")
print(f"  - Text: {text[:30]}...")
print(f"  - Prompt text: {prompt_text}")
print(f"  - Prompt wav: {os.path.abspath(prompt_wav)}")
print(f"  - File exists: {os.path.exists(prompt_wav)}")
print()

# 测试1: stream=False (和 example.py 一样)
print("=" * 60)
print("Test 1: stream=False (like example.py)")
print("=" * 60)
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)):
    output_file = 'test_stream_false.wav'
    sf.write(output_file, j['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    print(f"✓ Generated: {output_file}")
    print(f"  Shape: {j['tts_speech'].shape}")

print()

# 测试2: stream=True (你的 API 使用的方式)
print("=" * 60)
print("Test 2: stream=True (like stream_service.py)")
print("=" * 60)
chunks = []
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=True)):
    chunks.append(j['tts_speech'].squeeze().cpu().numpy())
    print(f"  Chunk {i}: shape={j['tts_speech'].shape}")

if chunks:
    import numpy as np
    full_audio = np.concatenate(chunks)
    output_file = 'test_stream_true.wav'
    sf.write(output_file, full_audio, cosyvoice.sample_rate)
    print(f"✓ Generated: {output_file}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Final shape: {full_audio.shape}")

print()
print("=" * 60)
print("请比较两个音频文件的音色:")
print("  - test_stream_false.wav (应该正确)")
print("  - test_stream_true.wav (可能有问题?)")
print("=" * 60)
