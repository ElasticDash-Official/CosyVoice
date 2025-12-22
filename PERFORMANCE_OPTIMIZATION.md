# CosyVoice 性能优化指南

## 已实施的优化

### 1. 日志优化 ✅
- 将日志级别从 INFO 改为 WARNING，减少 I/O 开销
- 移除冗余的详细日志输出
- 仅在错误时输出关键信息

### 2. 音频文件处理优化 ✅
- **预加载机制**：启动时预加载默认音频文件信息，避免每次请求都读取
- **缓存验证**：默认音频只验证一次，上传音频快速验证
- **减少 I/O**：移除重复的 soundfile 验证检查

### 3. 数据类型优化 ✅
- 移除不必要的 `astype('float32')` 转换（numpy数组默认已是float32）
- 直接使用 `tobytes()` 减少内存拷贝

### 4. 并发配置优化 ✅
- 提高 `limit_concurrency` 从 10 到 20
- 增加 `backlog=2048` 提升连接队列

### 5. 代码简化 ✅
- 移除重复的导入语句
- 简化日志输出逻辑
- 优化错误处理流程

## 进一步优化建议

### 🚀 模型级别优化

#### 1. 使用 TorchScript / ONNX 加速
```bash
# 导出为 TorchScript (JIT)
python cosyvoice/bin/export_jit.py

# 导出为 ONNX
python cosyvoice/bin/export_onnx.py
```

#### 2. 量化模型 (减少模型大小和推理时间)
```python
import torch
# 动态量化 (8-bit)
quantized_model = torch.quantization.quantize_dynamic(
    cosyvoice.model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### 3. 使用 Flash Attention (如果模型支持)
```bash
pip install flash-attn --no-build-isolation
```

### ⚡ 推理优化

#### 4. 批处理推理 (Batch Processing)
修改 API 支持批量文本：
```python
@app.post("/synthesize_batch")
async def synthesize_batch(texts: List[str], ...):
    # 批量处理多个文本，提高GPU利用率
    pass
```

#### 5. GPU优化
```python
# 在模型初始化时设置
torch.backends.cudnn.benchmark = True  # 自动选择最优算法
torch.set_float32_matmul_precision('high')  # TF32精度加速
```

#### 6. 使用 Triton/TensorRT 加速
参考 `runtime/triton_trtllm/` 目录的配置

### 🔧 系统级别优化

#### 7. 使用更快的 ASGI 服务器
```bash
# 替换 uvicorn 为 hypercorn (使用uvloop)
pip install hypercorn uvloop

# 启动
hypercorn stream_service:app --bind 0.0.0.0:50000 --workers 1
```

或者启用 uvicorn 的 uvloop：
```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=50000,
    loop="uvloop",  # 使用更快的事件循环
    workers=1
)
```

#### 8. 内存管理
```python
import torch
import gc

# 在每次推理后清理缓存
@app.post("/synthesize")
async def synthesize_streaming(...):
    try:
        # ... 推理代码 ...
        pass
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

#### 9. 使用 Redis 缓存常用结果
```python
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(text, instruction):
    return hashlib.md5(f"{text}_{instruction}".encode()).hexdigest()

# 在推理前检查缓存
cache_key = get_cache_key(text, instruction)
cached = redis_client.get(cache_key)
if cached:
    return cached
```

### 📊 监控和调试

#### 10. 添加性能监控
```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        logger.warning(f"{func.__name__} took {time.time()-start:.3f}s")
        return result
    return wrapper

@app.post("/synthesize")
@timing_decorator
async def synthesize_streaming(...):
    pass
```

#### 11. 使用 Prometheus 监控
```bash
pip install prometheus-fastapi-instrumentator

# 在代码中添加
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

## 性能测试

### 基准测试脚本
```bash
# 使用 ab (Apache Bench)
ab -n 100 -c 10 -p request.json -T application/json http://localhost:50000/synthesize

# 使用 wrk
wrk -t4 -c10 -d30s --latency http://localhost:50000/synthesize
```

### 预期性能提升
- **日志优化**: ~5-10% 延迟减少
- **音频缓存**: ~10-15% 首次请求加速
- **并发提升**: 支持更多并发请求
- **模型量化**: ~2-3x 推理加速（轻微质量损失）
- **TensorRT**: ~3-5x 推理加速
- **uvloop**: ~10-20% 网络性能提升

## 硬件建议

### GPU 优化
- 使用 NVIDIA GPU with Tensor Cores (RTX 系列、A100等)
- 至少 8GB VRAM
- CUDA 11.8+ 和最新驱动

### CPU 优化
- 使用支持 AVX2/AVX512 的 CPU
- 至少 4 核心
- 16GB+ RAM

### 网络优化
- 使用 HTTP/2 (FastAPI 默认支持)
- 启用 gzip 压缩
- 考虑使用 CDN 分发

## 故障排查

### 如果还是慢
1. **检查 GPU 利用率**: `nvidia-smi`
2. **检查内存**: `htop` 或 `free -h`
3. **检查磁盘 I/O**: `iotop`
4. **检查网络**: `iftop`
5. **分析瓶颈**: 使用 `py-spy` 或 `cProfile`

```bash
# 使用 py-spy 分析
pip install py-spy
py-spy top --pid <process_id>
```

## 总结

**低成本优化** (已完成):
- ✅ 日志级别调整
- ✅ 音频文件预加载
- ✅ 减少重复验证
- ✅ 代码简化

**中等成本优化** (推荐):
- 🔸 使用 uvloop
- 🔸 启用 TorchScript
- 🔸 GPU 后端优化

**高成本优化** (生产环境):
- 🔹 模型量化
- 🔹 TensorRT 加速
- 🔹 负载均衡 + 多实例
