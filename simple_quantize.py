#!/usr/bin/env python3
"""
简化版模型量化工具 - 只需要 .pt 权重文件
适用于只有权重文件、没有配置文件的模型
"""

import torch
import os
import sys
import shutil
from pathlib import Path

def get_file_size_mb(path):
    """获取文件大小（MB）"""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 ** 2)
    return 0

def quantize_weight_file(input_path, output_path, verbose=True):
    """
    量化单个权重文件（FP32 -> FP16）
    
    Args:
        input_path: 输入 .pt 文件路径
        output_path: 输出 .pt 文件路径
        verbose: 是否显示详细信息
    """
    if verbose:
        print(f"  Loading: {os.path.basename(input_path)}...", end=' ')
    
    # 加载权重
    state_dict = torch.load(input_path, map_location='cpu')
    
    # 转换为 FP16
    quantized_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            quantized_dict[key] = value.half()
        else:
            quantized_dict[key] = value
    
    # 保存
    torch.save(quantized_dict, output_path)
    
    if verbose:
        original_size = get_file_size_mb(input_path)
        quantized_size = get_file_size_mb(output_path)
        print(f"✓ ({original_size:.1f}MB -> {quantized_size:.1f}MB, {original_size/quantized_size:.2f}x)")
    
    return get_file_size_mb(input_path), get_file_size_mb(output_path)

def simple_quantize(model_dir, output_dir, skip_hift=True):
    """
    简化版量化 - 直接处理 .pt 文件
    
    Args:
        model_dir: 原始模型目录
        output_dir: 输出目录
        skip_hift: 是否跳过 HiFi-GAN 量化
    """
    print(f"🔧 Simple Quantization Tool")
    print(f"=" * 60)
    print(f"Input:  {model_dir}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)
    
    # 查找所有 .pt 文件
    pt_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pt'):
            pt_files.append(file)
    
    if not pt_files:
        print(f"❌ No .pt files found in {model_dir}")
        return False
    
    print(f"\n✓ Found {len(pt_files)} .pt file(s): {', '.join(pt_files)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 量化每个文件
    print(f"\n⚡ Quantizing models (FP32 -> FP16)...")
    total_original = 0
    total_quantized = 0
    
    for pt_file in pt_files:
        # 检查是否跳过 HiFi-GAN
        if skip_hift and ('hift' in pt_file.lower() or 'hifigan' in pt_file.lower()):
            print(f"  Copying: {pt_file}... ⏭️  (skipped, preserving quality)")
            shutil.copy(f"{model_dir}/{pt_file}", f"{output_dir}/{pt_file}")
            size = get_file_size_mb(f"{model_dir}/{pt_file}")
            total_original += size
            total_quantized += size
        else:
            orig, quant = quantize_weight_file(
                f"{model_dir}/{pt_file}",
                f"{output_dir}/{pt_file}"
            )
            total_original += orig
            total_quantized += quant
    
    # 复制其他文件
    print(f"\n📋 Copying other files...")
    other_files = [f for f in os.listdir(model_dir) 
                   if not f.endswith('.pt') and os.path.isfile(f"{model_dir}/{f}")]
    
    copied = 0
    for file in other_files:
        try:
            shutil.copy(f"{model_dir}/{file}", f"{output_dir}/{file}")
            copied += 1
        except:
            pass
    
    if copied > 0:
        print(f"  ✓ Copied {copied} additional file(s)")
    else:
        print(f"  ℹ️  No additional files to copy")
    
    # 创建标记文件
    with open(f"{output_dir}/QUANTIZED_INFO.txt", 'w') as f:
        from datetime import datetime
        f.write("=" * 60 + "\n")
        f.write("CosyVoice Quantized Model (FP16)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Original model: {model_dir}\n")
        f.write(f"Quantized on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"HiFi-GAN quantized: {not skip_hift}\n")
        f.write(f"Original size: {total_original:.1f} MB\n")
        f.write(f"Quantized size: {total_quantized:.1f} MB\n")
        f.write(f"Compression ratio: {total_original/total_quantized:.2f}x\n")
        f.write("\n")
        f.write("Usage:\n")
        f.write("  export COSYVOICE_FP16=true\n")
        f.write("  export COSYVOICE_QUANTIZED=true\n")
        f.write("  python stream_service.py\n")
    
    # 显示总结
    print(f"\n" + "=" * 60)
    print(f"📊 Quantization Summary")
    print(f"=" * 60)
    print(f"Original size:  {total_original:>10.1f} MB")
    print(f"Quantized size: {total_quantized:>10.1f} MB")
    print(f"Compression:    {total_original/total_quantized:>10.2f}x")
    print(f"Saved space:    {total_original-total_quantized:>10.1f} MB")
    print(f"=" * 60)
    
    print(f"\n✅ Quantization complete!")
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n💡 To use the quantized model:")
    print(f"")
    print(f"   Option 1 - Auto-detect (Recommended):")
    print(f"      export COSYVOICE_FP16=true")
    print(f"      export COSYVOICE_QUANTIZED=true")
    print(f"      python stream_service.py")
    print(f"")
    print(f"   Option 2 - Direct path:")
    print(f"      # Edit stream_service.py:")
    print(f"      model_dir = '{output_dir}'")
    print(f"      cosyvoice = AutoModel(model_dir=model_dir, fp16=True)")
    print(f"")
    print(f"   Option 3 - High-performance:")
    print(f"      export COSYVOICE_QUANTIZED=true")
    print(f"      ./start_fast.sh")
    print(f"")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Simple CosyVoice Model Quantization (FP32 -> FP16)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python simple_quantize.py pretrained_models/model pretrained_models/model-quantized
  
  # Also quantize HiFi-GAN (may affect quality)
  python simple_quantize.py pretrained_models/model pretrained_models/model-quantized --quantize-hift
        """
    )
    
    parser.add_argument('model_dir', type=str, help='Input model directory')
    parser.add_argument('output_dir', type=str, help='Output directory for quantized model')
    parser.add_argument('--quantize-hift', action='store_true',
                       help='Also quantize HiFi-GAN (may reduce audio quality)')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.model_dir):
        print(f"❌ Error: Not a directory: {args.model_dir}")
        sys.exit(1)
    
    if os.path.exists(args.output_dir):
        response = input(f"⚠️  Output directory exists: {args.output_dir}\n   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # 执行量化
    success = simple_quantize(
        args.model_dir,
        args.output_dir,
        skip_hift=not args.quantize_hift
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
