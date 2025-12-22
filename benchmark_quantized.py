#!/usr/bin/env python3
"""
性能测试工具：对比原始模型和量化模型的性能
"""

import torch
import time
import sys
import os
from cosyvoice.cli.cosyvoice import AutoModel

def benchmark_model(model_dir, test_text, iterations=5, use_fp16=False):
    """
    测试模型性能
    
    Args:
        model_dir: 模型目录
        test_text: 测试文本
        iterations: 测试次数
        use_fp16: 是否使用 FP16
    """
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(model_dir)}")
    print(f"FP16: {use_fp16}")
    print(f"{'='*60}")
    
    # 加载模型
    print("Loading model...")
    start_load = time.time()
    cosyvoice = AutoModel(model_dir=model_dir, fp16=use_fp16)
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # 准备测试数据
    prompt_wav = "/home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav"
    if not os.path.exists(prompt_wav):
        # 尝试其他可能的路径
        prompt_wav = os.path.join(os.path.dirname(model_dir), "asset/zero_shot_prompt.wav")
        if not os.path.exists(prompt_wav):
            print(f"⚠️  Warning: prompt_wav not found, using model's default")
            prompt_wav = None
    
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    # 预热（避免首次推理的初始化开销）
    print("Warming up...")
    if prompt_wav:
        for _ in cosyvoice.inference_zero_shot(test_text[:20], prompt_text, prompt_wav, stream=False):
            pass
    
    # 正式测试
    print(f"Running {iterations} iterations...")
    times = []
    audio_lengths = []
    
    for i in range(iterations):
        start = time.time()
        
        audio_chunks = []
        if prompt_wav:
            for result in cosyvoice.inference_zero_shot(test_text, prompt_text, prompt_wav, stream=False):
                audio_chunks.append(result['tts_speech'])
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        # 计算生成的音频长度
        if audio_chunks:
            total_samples = sum(chunk.shape[-1] for chunk in audio_chunks)
            audio_length = total_samples / cosyvoice.sample_rate
            audio_lengths.append(audio_length)
        
        print(f"  Iteration {i+1}/{iterations}: {elapsed:.3f}s", end='')
        if audio_lengths:
            rtf = elapsed / audio_lengths[-1]
            print(f" (RTF: {rtf:.3f})", end='')
        print()
    
    # 统计结果
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    if audio_lengths:
        avg_audio_len = sum(audio_lengths) / len(audio_lengths)
        avg_rtf = avg_time / avg_audio_len
    else:
        avg_audio_len = 0
        avg_rtf = 0
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Average time:     {avg_time:.3f}s")
    print(f"Min time:         {min_time:.3f}s")
    print(f"Max time:         {max_time:.3f}s")
    if avg_audio_len > 0:
        print(f"Audio length:     {avg_audio_len:.2f}s")
        print(f"RTF (Real-Time Factor): {avg_rtf:.3f}")
        print(f"  → {'✓ Faster than real-time' if avg_rtf < 1 else '✗ Slower than real-time'}")
    print(f"{'='*60}")
    
    return {
        'load_time': load_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_rtf': avg_rtf,
        'audio_length': avg_audio_len
    }

def compare_models(original_dir, quantized_dir, test_text, iterations=5):
    """对比原始模型和量化模型"""
    
    print("\n" + "🔥"*30)
    print("CosyVoice Performance Benchmark")
    print("🔥"*30)
    print(f"\nTest text: {test_text}")
    print(f"Iterations: {iterations}")
    
    # 测试原始模型
    try:
        results_original = benchmark_model(original_dir, test_text, iterations, use_fp16=False)
    except Exception as e:
        print(f"❌ Error testing original model: {e}")
        results_original = None
    
    # 测试量化模型
    try:
        results_quantized = benchmark_model(quantized_dir, test_text, iterations, use_fp16=True)
    except Exception as e:
        print(f"❌ Error testing quantized model: {e}")
        results_quantized = None
    
    # 对比结果
    if results_original and results_quantized:
        print("\n" + "="*60)
        print("📊 COMPARISON SUMMARY")
        print("="*60)
        
        speedup = results_original['avg_time'] / results_quantized['avg_time']
        load_speedup = results_original['load_time'] / results_quantized['load_time']
        
        print(f"\n{'Metric':<25} {'Original':<15} {'Quantized':<15} {'Improvement':<15}")
        print("-"*70)
        print(f"{'Load time':<25} {results_original['load_time']:>10.2f}s    {results_quantized['load_time']:>10.2f}s    {load_speedup:>10.2f}x")
        print(f"{'Inference time':<25} {results_original['avg_time']:>10.3f}s    {results_quantized['avg_time']:>10.3f}s    {speedup:>10.2f}x")
        print(f"{'RTF':<25} {results_original['avg_rtf']:>10.3f}     {results_quantized['avg_rtf']:>10.3f}     {results_original['avg_rtf']/results_quantized['avg_rtf']:>10.2f}x")
        print("-"*70)
        
        if speedup >= 1.5:
            emoji = "🚀🚀🚀"
        elif speedup >= 1.2:
            emoji = "🚀🚀"
        elif speedup >= 1.0:
            emoji = "🚀"
        else:
            emoji = "⚠️"
        
        print(f"\n{emoji} Overall speedup: {speedup:.2f}x {emoji}")
        
        if speedup >= 1.5:
            print("✅ Excellent! Quantization provides significant speedup.")
        elif speedup >= 1.2:
            print("👍 Good speedup from quantization.")
        elif speedup >= 1.0:
            print("📊 Modest improvement. Consider other optimizations.")
        else:
            print("⚠️  Quantized model is slower. Check configuration.")
        
        print("="*60)

def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark_quantized.py <original_model_dir> <quantized_model_dir> [test_text]")
        print("\nExample:")
        print("  python benchmark_quantized.py \\")
        print("    /home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B \\")
        print("    /home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B-quantized \\")
        print('    "你好，欢迎来到我们的餐厅。"')
        sys.exit(1)
    
    original_dir = sys.argv[1]
    quantized_dir = sys.argv[2]
    test_text = sys.argv[3] if len(sys.argv) > 3 else "你好，欢迎光临我们的餐厅，今天想吃点什么呢？"
    
    # 验证路径
    if not os.path.exists(original_dir):
        print(f"❌ Error: Original model directory not found: {original_dir}")
        sys.exit(1)
    
    if not os.path.exists(quantized_dir):
        print(f"❌ Error: Quantized model directory not found: {quantized_dir}")
        sys.exit(1)
    
    # 运行对比
    compare_models(original_dir, quantized_dir, test_text, iterations=5)

if __name__ == "__main__":
    main()
