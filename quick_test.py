#!/usr/bin/env python3
"""快速测试 TTS 服务"""
import requests

# 测试服务是否运行
print("1. 测试服务健康状态...")
try:
    response = requests.get("http://localhost:50000/health")
    print(f"   ✓ 服务正常: {response.json()}")
except Exception as e:
    print(f"   ✗ 服务异常: {e}")
    exit(1)

# 测试合成功能 (正确的 Form Data 格式)
print("\n2. 测试语音合成 (Form Data 格式)...")
try:
    response = requests.post(
        "http://localhost:50000/synthesize",
        data={"text": "你好,这是一个测试"}  # 使用 data= 而不是 json=
    )
    response.raise_for_status()
    print(f"   ✓ 合成成功! 音频大小: {len(response.content)} bytes")
except Exception as e:
    print(f"   ✗ 合成失败: {e}")

# 测试错误的 JSON 格式 (这会导致 422 错误)
print("\n3. 测试错误的 JSON 格式 (演示 422 错误)...")
try:
    response = requests.post(
        "http://localhost:50000/synthesize",
        json={"text": "你好,这是一个测试"}  # 使用 json= 会导致 422 错误
    )
    response.raise_for_status()
    print(f"   ✓ 合成成功")
except requests.exceptions.HTTPError as e:
    print(f"   ✗ 预期的错误: {e.response.status_code} - {e.response.text}")
