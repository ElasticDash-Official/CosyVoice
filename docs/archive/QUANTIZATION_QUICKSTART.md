# 🚀 快速量化指南

## 一键量化（最简单）

### 步骤 1: 量化模型
```bash
# 在服务器上运行
cd /home/ec2-user/CosyVoice

python quantize_model.py \
  pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized
```

### 步骤 2: 修改服务配置
编辑 `stream_service.py`:
```python
# 修改第 18 行
# 原来：
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"

# 改为：
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized"

# 修改第 20 行，添加 fp16=True
# 原来：
cosyvoice = AutoModel(model_dir=model_dir)

# 改为：
cosyvoice = AutoModel(model_dir=model_dir, fp16=True)
```

### 步骤 3: 重启服务
```bash
# 停止旧服务
pkill -f stream_service.py

# 启动优化后的服务
./start_fast.sh
```

---

## 性能测试（可选）

```bash
# 对比原始模型和量化模型的性能
python benchmark_quantized.py \
  pretrained_models/Fun-CosyVoice3-0.5B-2512 \
  pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized \
  "你好，欢迎来到我们的餐厅，今天想吃点什么呢？"
```

---

## 预期效果

| 指标 | 原始模型 | 量化模型 | 提升 |
|-----|---------|---------|-----|
| **模型大小** | ~500 MB | ~250 MB | **2x** ⬇️ |
| **加载时间** | ~10s | ~5s | **2x** ⬆️ |
| **推理速度** | 1.0x | 1.5-2.5x | **最高 2.5x** ⬆️ |
| **内存占用** | ~2GB | ~1GB | **2x** ⬇️ |
| **音质** | 100% | 95-98% | 轻微下降 |

---

## 故障排查

### 问题 1: 量化后报错
```bash
# 确保 PyTorch 版本 >= 1.13
pip install --upgrade torch

# 重新量化
python quantize_model.py <input> <output>
```

### 问题 2: 音质明显下降
```bash
# 不量化 HiFi-GAN（默认行为）
python quantize_model.py input output

# 如果已经量化了，重新量化但跳过 HiFi-GAN
python quantize_model.py input output  # 默认就会跳过
```

### 问题 3: GPU 上没有加速
```bash
# 确保启用 FP16
cosyvoice = AutoModel(model_dir=model_dir, fp16=True)

# 检查 CUDA 版本
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 4: 内存不足
```bash
# 量化可以减少内存占用
# 如果还是不够，减少 limit_concurrency
# 在 stream_service.py 中修改：
limit_concurrency=10  # 从 20 改为 10
```

---

## 高级选项

### 同时量化 HiFi-GAN（可能影响音质）
```bash
python quantize_model.py \
  pretrained_models/original \
  pretrained_models/quantized \
  --quantize-hift
```

### 仅使用 FP16（不量化）
在 `stream_service.py` 中：
```python
# 不需要运行 quantize_model.py
# 直接添加 fp16=True
cosyvoice = AutoModel(model_dir=model_dir, fp16=True)
```
这样可以获得 **1.5-2x 加速，几乎无音质损失**。

---

## 组合优化（最佳实践）

```bash
# 1. 量化模型
python quantize_model.py pretrained_models/original pretrained_models/quantized

# 2. 修改 stream_service.py
#    - model_dir 指向量化模型
#    - 添加 fp16=True
#    - 已包含其他优化（缓存、日志等）

# 3. 使用高性能启动脚本
./start_fast.sh

# 预期总体加速：2-3x
```

---

## 最快配置（生产环境）

```python
# stream_service.py 完整优化配置

# 1. 使用量化模型 + FP16
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized"
cosyvoice = AutoModel(model_dir=model_dir, fp16=True)

# 2. 启用 CUDA 优化（在文件开头添加）
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

# 3. 使用 uvloop（通过 start_fast.sh）
# 4. 优化并发配置
uvicorn.run(
    app,
    loop='uvloop',
    workers=1,
    limit_concurrency=20,
    backlog=2048,
)
```

---

## 验证量化效果

### 1. 检查模型大小
```bash
du -h pretrained_models/Fun-CosyVoice3-0.5B-2512/*.pt
du -h pretrained_models/Fun-CosyVoice3-0.5B-2512-quantized/*.pt
```

### 2. 测试推理速度
```bash
python benchmark_quantized.py original_model quantized_model
```

### 3. 测试音质（主观评价）
```bash
# 生成测试音频
python test_synthesize.py
```

---

## 总结

✅ **推荐方案**（平衡性能和质量）：
```bash
1. python quantize_model.py <input> <output>  # 量化模型
2. 修改 stream_service.py 添加 fp16=True
3. ./start_fast.sh  # 使用优化启动脚本
```

⚡ **极致性能**（可接受轻微质量损失）：
```bash
1. 量化模型
2. fp16=True
3. 量化 HiFi-GAN (--quantize-hift)
4. 使用 TensorRT (runtime/triton_trtllm/)
```

💎 **最佳质量**（优先音质）：
```bash
1. 只用 fp16=True（不量化）
2. 不量化 HiFi-GAN（默认）
3. 其他代码级优化（已完成）
```
