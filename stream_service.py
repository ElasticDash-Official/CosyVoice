# cosyvoice_service.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice, CosyVoice2, CosyVoice3
from typing import Optional
import os
import logging
import tempfile
import soundfile as sf
import numpy as np
import io
import time
from functools import lru_cache

# 配置日志 - 生产环境使用WARNING级别减少开销
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 抑制第三方库的警告
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*weight_norm.*')

# 初始化 FastAPI 应用
app = FastAPI()

# GPU优化（如果可用）
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 自动选择最优卷积算法
    torch.set_float32_matmul_precision('high')  # 使用TF32精度加速
    # 启用 CUDA 图优化（减少kernel启动开销）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 初始化 CosyVoice 模型
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B-quantized"
alt_model_dir = "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B-quantized-2"
# model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
# alt_model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-quantized"

# 性能优化：支持通过环境变量启用 FP16 和量化模型
USE_FP16 = os.getenv('COSYVOICE_FP16', 'true').lower() == 'true'
USE_QUANTIZED = os.getenv('COSYVOICE_QUANTIZED', 'true').lower() == 'true'  # 默认启用量化

if USE_QUANTIZED:
    # 自动查找量化模型目录（优先级：FP16 > 原始）
    fp16_dir = model_dir.rstrip('/') + '-quantized'
    
    if os.path.exists(fp16_dir):
        model_dir = fp16_dir
        print(f"✅ Using FP16 quantized model: {model_dir}")
    else:
        print(f"⚠️  Quantized model not found at {fp16_dir}")
        print(f"    Using original model (will be slower)")

print(f"📂 Loading model from: {model_dir}")
print(f"⚙️  FP16 enabled: {USE_FP16}")
print(f"⚙️  Quantized enabled: {USE_QUANTIZED}")

cosyvoice = AutoModel(model_dir=model_dir, load_trt=True, fp16=USE_FP16)

# 记录 worker 信息（用于日志追踪）
import multiprocessing as mp
current_pid = os.getpid()

# 使用更可靠的 worker 识别方法：gunicorn 设置的环境变量或进程序号
# gunicorn 会为每个 worker 按顺序分配，我们通过 PID 取模 + 启动顺序来区分
WORKER_ID = str(current_pid % 10)

# 为了实现真正的负载均衡，奇数 worker 使用 quantized-2 副本
if int(WORKER_ID) % 2 == 1:
    if os.path.exists(alt_model_dir) and alt_model_dir != model_dir:
        print(f"🔄 Worker {WORKER_ID} (PID: {current_pid}) loading alternate quantized model...")
        cosyvoice = AutoModel(model_dir=alt_model_dir, load_trt=True, fp16=USE_FP16)
        model_dir = alt_model_dir  # 更新 model_dir 用于日志
        print(f"✅ Worker {WORKER_ID} loaded: {alt_model_dir}")
    else:
        print(f"⚠️  Worker {WORKER_ID} (PID: {current_pid}): quantized-2 not found, using default")
        print(f"    Model: {model_dir}")
else:
    print(f"✅ Worker {WORKER_ID} (PID: {current_pid}) using default: {model_dir}")

# 检测模型类型
model_type = type(cosyvoice).__name__
logger.warning(f"Loaded model type: {model_type}")

# 可选：torch.compile加速（PyTorch 2.0+，首次推理会慢，后续会快）
USE_COMPILE = os.getenv('COSYVOICE_COMPILE', 'false').lower() == 'true'
if USE_COMPILE and hasattr(torch, 'compile'):
    try:
        logger.warning("Applying torch.compile optimization...")
        # 编译模型的关键部分（如果支持）
        if hasattr(cosyvoice, 'model'):
            cosyvoice.model.llm = torch.compile(cosyvoice.model.llm, mode='reduce-overhead')
        logger.warning("torch.compile applied successfully")
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")

# 默认的prompt音频文件路径和对应文本
# 使用绝对路径确保在任何工作目录下都能找到文件
# default_prompt_wav = "/home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav"
# default_prompt_text = "希望你以后能够做的比我还好呦。"  # zero_shot_prompt.wav 的文字内容

default_prompt_wav = "/home/ec2-user/CosyVoice/asset/paimon_prompt.wav"
default_prompt_text = "那种地方不太可能会有人吧？"  # paimon_prompt.wav 的文字内容

# 默认的instruction(餐馆店员场景)
default_instruction = "你是一位热情友好的餐馆店员,说话温柔亲切,语气礼貌专业。<|endofprompt|>"

# 对于CosyVoice(第一代),获取可用的speakers列表
available_speakers = []
default_speaker = None
if isinstance(cosyvoice, CosyVoice):
    available_speakers = cosyvoice.list_available_spks()
    logger.info(f"Available speakers: {available_speakers}")
    default_speaker = available_speakers[0] if available_speakers else None
    logger.info(f"Default speaker: {default_speaker}")

# 性能优化：预加载和缓存默认音频
_default_audio_cache = None
_default_audio_info = None

if os.path.exists(default_prompt_wav):
    try:
        # 预加载默认音频文件信息，避免每次请求都读取
        _default_audio_info = sf.info(default_prompt_wav)
        logger.info(f"✓ Preloaded default audio: {default_prompt_wav} ({_default_audio_info.duration:.2f}s)")
    except Exception as e:
        logger.warning(f"Failed to preload default audio: {e}")

# 统一的请求模型
class TTSRequest(BaseModel):
    text: str
    instruction: Optional[str] = None  # CosyVoice2/3使用,例如: "你是一位热情友好的餐馆店员"
    speaker: Optional[str] = None  # CosyVoice第一代使用
    prompt_text: Optional[str] = None  # 用于zero-shot的参考文本

# 统一的合成路由 - 支持所有模型
@app.post("/synthesize")
async def synthesize_streaming(
    text: str = Form(...),
    instruction: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    prompt_text: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None)
):
    """
    统一的语音合成接口,自动适配不同模型:

    CosyVoice第一代:
    - 使用 speaker 参数 (如 "中文女")

    CosyVoice2/3:
    - 使用 instruction 参数 (如 "你是一位热情友好的女餐馆店员,说话温柔亲切,语气礼貌专业。<|endofprompt|>")
    - 可选: prompt_wav 音频文件 (如果不提供,使用默认音频)
    - 可选: prompt_text 参考文本

    默认为餐馆店员场景
    """
    try:
        # # CosyVoice 第一代 - 使用 speaker_id
        # if isinstance(cosyvoice, CosyVoice):
        #     speaker = speaker if speaker else default_speaker

        #     if speaker not in available_speakers:
        #         logger.warning(f"Requested speaker '{speaker}' not available. Using default: {default_speaker}")
        #         speaker = default_speaker

        #     if not speaker:
        #         raise HTTPException(status_code=500, detail="No available speakers found")

        #     logger.info(f"[CosyVoice] Synthesizing with speaker: {speaker}")

        #     async def audio_stream():
        #         for result in cosyvoice.inference_sft(text, speaker, stream=True):
        #             audio_chunk = result["tts_speech"].numpy().tobytes()
        #             yield audio_chunk

        #     return StreamingResponse(audio_stream(), media_type="application/octet-stream")

        # CosyVoice2/3 - 支持多种推理模式
        if isinstance(cosyvoice, (CosyVoice2, CosyVoice3)):
            # 推理模式选择:
            # 有 instruction → 使用 instruct2 (指令控制风格)
            # 无 instruction → 使用 zero_shot (纯声音克隆,需要 prompt_text)
            # cross_lingual 用于细粒度控制 ([laughter], [breath] 等标记)
            instruction_text = instruction if instruction else None

            # 记录开始时间和模型信息
            start_ts = time.perf_counter()
            logger.warning(
                "[TTS] start worker=%s model=%s dir=%s text_len=%d instr=%s",
                WORKER_ID,
                model_type,
                model_dir,
                len(text),
                bool(instruction_text),
            )

            # 处理prompt_wav文件 (可选) - 优化版本
            temp_wav_path = None
            if prompt_wav:
                # 用户上传了音频文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    content = await prompt_wav.read()
                    temp_file.write(content)
                    temp_wav_path = temp_file.name
                logger.debug(f"Using uploaded audio: {temp_wav_path}")
            elif os.path.exists(default_prompt_wav):
                # 使用默认音频文件（已预加载验证）
                temp_wav_path = default_prompt_wav
                logger.debug(f"Using default audio: {default_prompt_wav}")
            else:
                # 没有音频文件
                logger.error(f"No audio file available: {default_prompt_wav}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice reference required. Place file at: {default_prompt_wav}"
                )

            # 根据参数选择推理方法 - 优化版本
            if temp_wav_path:
                # 跳过重复验证（上传的文件已在保存时验证，默认文件已预加载）
                # 仅对上传文件进行快速验证
                if prompt_wav:
                    try:
                        audio_info = sf.info(temp_wav_path)
                        if audio_info.duration < 0.1:  # 快速检查
                            raise ValueError("Audio too short")
                    except Exception as e:
                        logger.error(f"Invalid audio: {e}")
                        raise HTTPException(status_code=400, detail="Invalid audio file")
                
                # 确保使用绝对路径
                abs_wav_path = os.path.abspath(temp_wav_path)
                
                if instruction_text:
                    # INSTRUCT2 模式: instruction + voice
                    logger.debug(f"Mode: INSTRUCT2, text_len={len(text)}")
                    inference_method = lambda: cosyvoice.inference_instruct2(
                        text,
                        instruction_text,
                        abs_wav_path, 
                        stream=True
                    )
                else:
                    # ZERO_SHOT 模式: 纯声音克隆
                    actual_prompt_text = prompt_text if prompt_text else default_prompt_text
                    logger.debug(f"Mode: ZERO_SHOT, text_len={len(text)}")
                    inference_method = lambda: cosyvoice.inference_zero_shot(
                        text,
                        'You are a helpful assistant.<|endofprompt|>' + actual_prompt_text,
                        # actual_prompt_text,
                        abs_wav_path, 
                        stream=True
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Voice reference required"
                )

            async def audio_stream():
                total_samples = 0
                sample_rate = getattr(cosyvoice, "sample_rate", 24000)
                
                try:
                    for result in inference_method():
                        # 获取音频数据（Float32 格式）
                        audio_chunk = result["tts_speech"].squeeze()

                        # 转换为 Int16 PCM（标准音频格式，避免杂音）
                        # Float32 范围 [-1.0, 1.0] -> Int16 范围 [-32768, 32767]
                        audio_int16 = (audio_chunk * 32767).to(torch.int16)

                        # 直接传输 PCM 数据
                        yield audio_int16.cpu().numpy().tobytes()
                            
                finally:
                    # 只清理上传的临时文件
                    if prompt_wav and temp_wav_path and os.path.exists(temp_wav_path):
                        try:
                            os.unlink(temp_wav_path)
                        except:
                            pass

                    wall = time.perf_counter() - start_ts
                    audio_sec = total_samples / float(sample_rate) if total_samples else 0.0
                    rtf = wall / audio_sec if audio_sec > 0 else 0.0
                    logger.warning(
                        "[TTS] done worker=%s model=%s dir=%s wall=%.2fs audio=%.2fs rtf=%.3f text_len=%d",
                        WORKER_ID,
                        model_type,
                        model_dir,
                        wall,
                        audio_sec,
                        rtf,
                        len(text),
                    )

            return StreamingResponse(audio_stream(), media_type="audio/wav")

        else:
            raise HTTPException(status_code=500, detail=f"Unknown model type: {model_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during synthesis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 非流式测试端点 - 返回完整 WAV 文件
@app.post("/synthesize_complete")
async def synthesize_complete(
    text: str = Form(...),
    instruction: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None)
):
    """
    非流式合成接口，返回完整的 WAV 文件（用于测试音色）
    """
    try:
        if isinstance(cosyvoice, (CosyVoice2, CosyVoice3)):
            instruction_text = instruction if instruction else None
            
            # 处理音频文件
            temp_wav_path = None
            if prompt_wav:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    content = await prompt_wav.read()
                    temp_file.write(content)
                    temp_wav_path = temp_file.name
            elif os.path.exists(default_prompt_wav):
                temp_wav_path = default_prompt_wav
            else:
                raise HTTPException(status_code=400, detail="No voice reference audio")
            
            abs_wav_path = os.path.abspath(temp_wav_path)
            logger.debug(f"Complete synthesis - voice: {abs_wav_path}")
            
            # 收集所有音频块
            chunks = []
            
            if instruction_text:
                for result in cosyvoice.inference_instruct2(text, instruction_text, abs_wav_path, stream=False):
                    chunks.append(result["tts_speech"].squeeze().cpu().numpy())
            else:
                for result in cosyvoice.inference_zero_shot(
                    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
                    'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                    abs_wav_path,
                    stream=False
                ):
                    chunks.append(result["tts_speech"].squeeze().cpu().numpy())
            
            # 拼接并生成 WAV
            if chunks:
                full_audio = np.concatenate(chunks)
                buffer = io.BytesIO()
                sf.write(buffer, full_audio, cosyvoice.sample_rate, format='WAV')
                buffer.seek(0)
                
                # 清理临时文件
                if prompt_wav and temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                
                return StreamingResponse(buffer, media_type="audio/wav")
            else:
                raise HTTPException(status_code=500, detail="No audio generated")
        else:
            raise HTTPException(status_code=400, detail="Model not supported")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during complete synthesis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 获取模型信息的路由
@app.get("/model_info")
async def get_model_info():
    info = {
        "model_type": model_type,
        "model_dir": model_dir,
    }

    if isinstance(cosyvoice, CosyVoice):
        info["speakers"] = available_speakers
        info["default_speaker"] = default_speaker
        info["usage"] = "Use /synthesize endpoint with 'speaker' parameter (CosyVoice v1)"
    else:
        info["default_instruction"] = default_instruction
        info["usage"] = "Use /synthesize endpoint with 'instruction' parameter (CosyVoice2/3)"

    info["endpoint"] = "/synthesize"

    return info

# 获取可用speakers列表的路由(仅用于CosyVoice第一代)
@app.get("/speakers")
async def get_speakers():
    if not isinstance(cosyvoice, CosyVoice):
        raise HTTPException(
            status_code=400,
            detail=f"Speakers list is only available for CosyVoice v1. Current model is {model_type}."
        )
    return {
        "speakers": available_speakers,
        "default_speaker": default_speaker
    }

# 健康检查路由
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_type": model_type}

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    # 根据GPU数量和内存自动调整worker数 (每个worker会加载一次模型)
    # GPU内存充足时可增加到4-8个worker
    num_workers = min(4, max(1, multiprocessing.cpu_count() // 2))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50000,
        workers=num_workers,    # 多进程并行处理请求
        limit_concurrency=50,   # 增加并发连接数
        timeout_keep_alive=30,  # 30秒keepalive超时
        backlog=2048,          # 增加连接队列
        log_level="warning"     # 只记录警告和错误
    )
