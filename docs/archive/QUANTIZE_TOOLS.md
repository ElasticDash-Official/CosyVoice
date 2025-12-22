# 量化工具说明

## 两个量化工具的区别

### `simple_quantize.py` ⭐ 推荐
**适用场景：**
- ✅ 模型目录只有 `.pt` 权重文件
- ✅ 没有 `cosyvoice.yaml` 配置文件
- ✅ 快速简单，无需加载完整模型

**使用方法：**
```bash
python simple_quantize.py \
  pretrained_models/original \
  pretrained_models/quantized
```

**工作原理：**
- 直接读取 `.pt` 文件
- 将所有 FP32 参数转换为 FP16
- 复制其他文件到输出目录
- 不需要加载完整模型

---

### `quantize_model.py`
**适用场景：**
- ✅ 完整的 CosyVoice 模型（带配置文件）
- ✅ 需要通过 AutoModel 加载的模型
- ✅ 需要验证模型结构的场景

**使用方法：**
```bash
python quantize_model.py \
  pretrained_models/original \
  pretrained_models/quantized
```

**工作原理：**
- 通过 CosyVoice 的 AutoModel 加载
- 需要 `cosyvoice.yaml` 等配置文件
- 更严格的模型验证

---

## 快速决策

### 情况 1: 报错 "cosyvoice.yaml not found"
```bash
# 使用简化版
python simple_quantize.py <input> <output>
```

### 情况 2: 完整的模型包
```bash
# 两个都可以，推荐简化版（更快）
python simple_quantize.py <input> <output>
```

### 情况 3: 一键优化脚本
```bash
# optimize_and_start.sh 已更新为使用 simple_quantize.py
./optimize_and_start.sh
```

---

## 命令对比

| 特性 | simple_quantize.py | quantize_model.py |
|-----|-------------------|-------------------|
| **速度** | ⚡ 快 | 较慢 |
| **依赖** | 只需 PyTorch | 需要 CosyVoice |
| **配置文件要求** | ❌ 不需要 | ✅ 需要 |
| **验证** | 基础 | 完整 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 效果对比

两个工具产生的量化效果**完全相同**：
- ✅ 都是 FP32 → FP16 转换
- ✅ 压缩比相同（~2x）
- ✅ 推理速度提升相同（1.5-2x）
- ✅ 音质影响相同（几乎无损）

唯一区别是**执行方式**，结果是一样的。

---

## 故障排查

### 问题：simple_quantize.py 找不到 .pt 文件
```bash
# 检查目录内容
ls -la pretrained_models/your_model/

# 确保有 .pt 文件
```

### 问题：quantize_model.py 报错配置文件缺失
```bash
# 改用简化版
python simple_quantize.py <input> <output>
```

### 问题：量化后模型无法加载
```bash
# 检查原始模型目录结构
ls -la pretrained_models/original/

# 确保量化目录包含所有必要文件
ls -la pretrained_models/quantized/
```

---

## 推荐使用方式

**默认选择：**
```bash
python simple_quantize.py \
  pretrained_models/original \
  pretrained_models/quantized
```

**高级选项：**
```bash
# 也量化 HiFi-GAN（可能影响音质）
python simple_quantize.py \
  pretrained_models/original \
  pretrained_models/quantized \
  --quantize-hift
```

**一键优化：**
```bash
# 自动使用 simple_quantize.py
./optimize_and_start.sh
```
