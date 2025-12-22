# CosyVoice 模型量化完整指南

## 什么是模型量化？

模型量化是将模型参数从 **32位浮点数 (FP32)** 转换为 **8位整数 (INT8)** 或 **16位浮点数 (FP16)**，从而：
- ⚡ **推理速度提升 2-4倍**
- 💾 **模型大小减少 2-4倍**
- 🎯 **精度损失 < 5%**（通常可忽略）

## 量化方法对比

| 方法 | 速度提升 | 精度损失 | 难度 | 推荐场景 |
|-----|---------|---------|------|---------|
| **FP16 半精度** | 1.5-2x | <1% | ⭐ 简单 | GPU推理，几乎无损 |
| **动态量化** | 2-3x | 2-3% | ⭐⭐ 中等 | CPU/GPU，快速部署 |
| **静态量化** | 3-4x | 1-2% | ⭐⭐⭐ 复杂 | 生产环境，需校准 |
| **量化感知训练(QAT)** | 3-4x | <1% | ⭐⭐⭐⭐ 困难 | 最佳质量，需重训练 |

## 方法 1: FP16 半精度（最简单，推荐）

### 优点
- ✅ 几乎无精度损失
- ✅ 速度提升 50-100%
- ✅ 只需修改几行代码
- ✅ 支持 NVIDIA GPU (Compute Capability >= 7.0)

### 使用方法

**在 `stream_service.py` 中启用 FP16：**

```python
# 修改模型初始化
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
cosyvoice = AutoModel(model_dir=model_dir, fp16=True)  # 添加 fp16=True
```

**或使用环境变量：**
```bash
export COSYVOICE_FP16=1
python stream_service.py
```

### 验证 FP16 是否生效
```python
# 检查模型参数类型
print(next(cosyvoice.model.llm.parameters()).dtype)
# 输出: torch.float16 (成功) 或 torch.float32 (未启用)
```

---

## 方法 2: 动态量化（推荐，平衡性能和质量）

### 特点
- 运行时自动量化权重
- 不需要校准数据
- 适用于 Linear 层（全连接层）

### 使用脚本

创建 `quantize_model.py`：

```python
import torch
from cosyvoice.cli.cosyvoice import AutoModel
import os

def quantize_dynamic(model_dir, output_dir):
    """动态量化 CosyVoice 模型"""
    print(f"Loading model from {model_dir}...")
    cosyvoice = AutoModel(model_dir=model_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 量化 LLM 模块
    print("Quantizing LLM...")
    llm_quantized = torch.quantization.quantize_dynamic(
        cosyvoice.model.llm,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},  # 要量化的层类型
        dtype=torch.qint8
    )
    
    # 量化 Flow 模块
    print("Quantizing Flow...")
    flow_quantized = torch.quantization.quantize_dynamic(
        cosyvoice.model.flow,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # 保存量化模型
    print(f"Saving quantized models to {output_dir}...")
    torch.save(llm_quantized.state_dict(), f"{output_dir}/llm_quantized.pt")
    torch.save(flow_quantized.state_dict(), f"{output_dir}/flow_quantized.pt")
    
    # HiFi-GAN 通常不量化（对音质影响大）
    torch.save(cosyvoice.model.hift.state_dict(), f"{output_dir}/hift.pt")
    
    # 复制其他必要文件
    import shutil
    for file in ['cosyvoice.yaml', 'campplus.onnx', 'speech_tokenizer_v1.onnx', 'spk2info.pt']:
        src = f"{model_dir}/{file}"
        if os.path.exists(src):
            shutil.copy(src, f"{output_dir}/{file}")
    
    print("✅ Quantization complete!")
    print(f"Quantized model saved to: {output_dir}")
    
    # 计算模型大小
    original_size = sum(os.path.getsize(f"{model_dir}/{f}") 
                       for f in ['llm.pt', 'flow.pt', 'hift.pt'])
    quantized_size = sum(os.path.getsize(f"{output_dir}/{f}") 
                        for f in ['llm_quantized.pt', 'flow_quantized.pt', 'hift.pt'])
    
    print(f"\n📊 Size comparison:")
    print(f"  Original: {original_size / 1024**2:.1f} MB")
    print(f"  Quantized: {quantized_size / 1024**2:.1f} MB")
    print(f"  Compression: {original_size / quantized_size:.2f}x")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python quantize_model.py <model_dir> <output_dir>")
        print("Example: python quantize_model.py pretrained_models/CosyVoice2-0.5B pretrained_models/CosyVoice2-0.5B-quantized")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    quantize_dynamic(model_dir, output_dir)
```

### 运行量化
```bash
python quantize_model.py \
    /home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512 \
    /home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized
```

---

## 方法 3: 静态量化（最佳性能）

### 特点
- 需要校准数据集
- 量化激活值和权重
- 性能提升最大

### 完整脚本

创建 `quantize_static.py`：

```python
import torch
from torch.quantization import quantize_fx
from cosyvoice.cli.cosyvoice import AutoModel
import os

def calibrate_model(model, calibration_data):
    """使用校准数据集"""
    model.eval()
    with torch.no_grad():
        for text, prompt_text, prompt_wav in calibration_data:
            # 运行推理进行校准
            for _ in model.inference_zero_shot(text, prompt_text, prompt_wav, stream=False):
                pass

def quantize_static(model_dir, output_dir, calibration_texts):
    """静态量化"""
    print(f"Loading model from {model_dir}...")
    cosyvoice = AutoModel(model_dir=model_dir)
    
    # 准备校准数据
    print("Preparing calibration data...")
    calibration_data = []
    prompt_wav = f"{model_dir}/../asset/zero_shot_prompt.wav"
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    for text in calibration_texts:
        calibration_data.append((text, prompt_text, prompt_wav))
    
    # 配置量化
    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 准备模型
    cosyvoice.model.llm.qconfig = qconfig
    cosyvoice.model.flow.qconfig = qconfig
    
    # 插入观察器
    print("Inserting observers...")
    torch.quantization.prepare(cosyvoice.model.llm, inplace=True)
    torch.quantization.prepare(cosyvoice.model.flow, inplace=True)
    
    # 校准
    print("Calibrating with sample data...")
    calibrate_model(cosyvoice, calibration_data)
    
    # 转换为量化模型
    print("Converting to quantized model...")
    torch.quantization.convert(cosyvoice.model.llm, inplace=True)
    torch.quantization.convert(cosyvoice.model.flow, inplace=True)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    torch.save(cosyvoice.model.llm.state_dict(), f"{output_dir}/llm_static_quantized.pt")
    torch.save(cosyvoice.model.flow.state_dict(), f"{output_dir}/flow_static_quantized.pt")
    
    print(f"✅ Static quantization complete! Saved to {output_dir}")

if __name__ == "__main__":
    model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
    output_dir = f"{model_dir}-static-quantized"
    
    # 校准文本（使用真实场景的文本）
    calibration_texts = [
        "你好，欢迎光临我们的餐厅。",
        "今天有什么可以帮助您的吗？",
        "这是您的账单，总共是一百二十元。",
        "祝您用餐愉快，期待您的再次光临。",
        "我们今天的特色菜是红烧狮子头和清蒸鲈鱼。"
    ]
    
    quantize_static(model_dir, output_dir, calibration_texts)
```

---

## 方法 4: 使用 BetterTransformer (快速优化)

### 特点
- 使用 PyTorch 内置优化
- 无需量化，速度提升 20-40%
- 零精度损失

```python
# 安装
pip install optimum

# 在代码中使用
from optimum.bettertransformer import BetterTransformer

# 优化 Transformer 模块
if hasattr(cosyvoice.model.llm, 'text_encoder'):
    cosyvoice.model.llm.text_encoder = BetterTransformer.transform(
        cosyvoice.model.llm.text_encoder
    )
```

---

## 加载量化模型

修改 `stream_service.py` 以支持量化模型：

```python
import torch

# 在模型初始化时
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized"

# 方法1: 直接加载量化模型
cosyvoice = AutoModel(model_dir=model_dir)

# 方法2: 手动加载量化权重
cosyvoice = AutoModel(model_dir=original_model_dir)
quantized_llm = torch.load(f"{model_dir}/llm_quantized.pt")
quantized_flow = torch.load(f"{model_dir}/flow_quantized.pt")
cosyvoice.model.llm.load_state_dict(quantized_llm)
cosyvoice.model.flow.load_state_dict(quantized_flow)
```

---

## 性能测试

### 测试脚本 `benchmark_quantized.py`
```python
import torch
import time
from cosyvoice.cli.cosyvoice import AutoModel

def benchmark(model_dir, text, iterations=10):
    cosyvoice = AutoModel(model_dir=model_dir)
    prompt_wav = "/home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav"
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    # 预热
    for _ in cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False):
        pass
    
    # 测试
    times = []
    for i in range(iterations):
        start = time.time()
        for _ in cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False):
            pass
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.3f}s")
    return avg_time

if __name__ == "__main__":
    test_text = "你好，欢迎来到我们的餐厅，今天想吃点什么呢？"
    
    print("Testing original model...")
    time_original = benchmark("pretrained_models/original", test_text)
    
    print("\nTesting quantized model...")
    time_quantized = benchmark("pretrained_models/quantized", test_text)
    
    print(f"\n🚀 Speedup: {time_original/time_quantized:.2f}x")
```

---

## 注意事项

### ⚠️ 不要量化的模块
- **HiFi-GAN** (vocoder)：对音质影响大
- **Embedding 层**：维度通常不大
- **BatchNorm/LayerNorm**：量化收益小

### ✅ 推荐量化的模块
- **Linear 层**（全连接）：收益最大
- **Attention 模块**：速度提升明显
- **LLM 部分**：参数量大，适合量化

### 🔍 质量检查
```bash
# 生成对比音频
python test_quantized_quality.py

# 使用 MOS (Mean Opinion Score) 评估
# 或使用 ViSQOL 客观评价工具
```

---

## 推荐方案

### 开发/测试环境
```bash
# 使用 FP16（最简单）
export COSYVOICE_FP16=1
python stream_service.py
```

### 生产环境
```bash
# 1. 量化模型
python quantize_model.py original_model quantized_model

# 2. 测试质量
python benchmark_quantized.py

# 3. 部署
# 修改 stream_service.py 中的 model_dir 指向量化模型
```

### 极致性能
```bash
# 使用 TensorRT (需要 NVIDIA GPU)
# 参考 runtime/triton_trtllm/ 目录
cd runtime/triton_trtllm
./run.sh
```

---

## 常见问题

**Q: 量化后音质下降明显？**
A: 
1. 不要量化 HiFi-GAN
2. 使用静态量化 + 充足校准数据
3. 考虑使用 FP16 而非 INT8

**Q: CPU 上量化效果不好？**
A: INT8 量化主要优化 CPU，GPU 上建议用 FP16

**Q: 量化后报错？**
A: 某些操作不支持量化，可以使用 `QuantStub` 和 `DeQuantStub` 包裹

**Q: 内存占用没减少？**
A: 动态量化仅减少磁盘大小，运行时仍解压为 FP32。使用静态量化可减少运行时内存。
