#!/usr/bin/env python3
"""
验证默认音频文件是否被正确使用
"""
import soundfile as sf
import os

# 检查默认音频文件
default_prompt_wav = "./asset/zero_shot_prompt.wav"

print("=" * 60)
print("验证 CosyVoice2 默认音频文件")
print("=" * 60)

if os.path.exists(default_prompt_wav):
    abs_path = os.path.abspath(default_prompt_wav)
    file_size = os.path.getsize(default_prompt_wav)

    print(f"✓ 文件存在: {abs_path}")
    print(f"  - 文件大小: {file_size} bytes ({file_size/1024:.1f} KB)")

    try:
        # 读取音频信息
        audio_data, sample_rate = sf.read(default_prompt_wav)
        audio_info = sf.info(default_prompt_wav)

        print(f"\n✓ 音频文件属性:")
        print(f"  - 采样率: {audio_info.samplerate} Hz")
        print(f"  - 时长: {audio_info.duration:.2f} 秒")
        print(f"  - 声道数: {audio_info.channels}")
        print(f"  - 格式: {audio_info.format}")
        print(f"  - 样本数: {len(audio_data)}")

        print(f"\n✓ 这个音频文件将作为 CosyVoice2 的基底音色")
        print(f"  - instruction 参数控制说话风格")
        print(f"  - 这个音频文件提供声音音色特征")

    except Exception as e:
        print(f"\n✗ 无法读取音频文件: {e}")
else:
    print(f"✗ 文件不存在: {os.path.abspath(default_prompt_wav)}")
    print(f"  - 当前工作目录: {os.getcwd()}")

print("\n" + "=" * 60)

# 检查服务是否能找到这个文件
print("\n测试服务路径:")
print(f"当前工作目录: {os.getcwd()}")
print(f"相对路径: {default_prompt_wav}")
print(f"绝对路径: {os.path.abspath(default_prompt_wav)}")
