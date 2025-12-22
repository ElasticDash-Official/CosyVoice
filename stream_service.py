# cosyvoice_service.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice, CosyVoice2, CosyVoice3
from typing import Optional
import os
import logging
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 CosyVoice 模型
# model_dir = "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B"
# model_dir = "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice-300M-SFT"
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"

cosyvoice = AutoModel(model_dir=model_dir)

# 检测模型类型
model_type = type(cosyvoice).__name__
logger.info(f"Loaded model type: {model_type}")

# 默认的prompt音频文件路径和对应文本
# 使用绝对路径确保在任何工作目录下都能找到文件
default_prompt_wav = "/home/ec2-user/CosyVoice/asset/zero_shot_prompt.wav"
default_prompt_text = "希望你以后能够做的比我还好呦。"  # zero_shot_prompt.wav 的文字内容

# default_prompt_wav = "/home/ec2-user/CosyVoice/asset/paimon_prompt.wav"
# default_prompt_text = "等等，算起来…今天是不是就是连续打工的第三天了？现在正是午饭时间！"  # zero_shot_prompt.wav 的文字内容

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

            # 处理prompt_wav文件 (可选)
            temp_wav_path = None
            if prompt_wav:
                # 用户上传了音频文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    content = await prompt_wav.read()
                    temp_file.write(content)
                    temp_wav_path = temp_file.name
                file_size = os.path.getsize(temp_wav_path)
                logger.info(f"✓ Using uploaded prompt_wav: {temp_wav_path} (size: {file_size} bytes)")
            elif os.path.exists(default_prompt_wav):
                # 使用默认音频文件
                temp_wav_path = default_prompt_wav
                abs_path = os.path.abspath(temp_wav_path)
                file_size = os.path.getsize(temp_wav_path)
                logger.info(f"✓ Using DEFAULT prompt_wav: {abs_path}")
                logger.info(f"  - File size: {file_size} bytes ({file_size/1024:.1f} KB)")
                logger.info(f"  - This audio will be used as the BASE VOICE for synthesis")
            else:
                # 没有音频文件,仅使用 instruction
                abs_path = os.path.abspath(default_prompt_wav)
                logger.warning(f"✗ No prompt_wav provided and default file not found!")
                logger.warning(f"  - Expected path: {abs_path}")
                logger.warning(f"  - Current working directory: {os.getcwd()}")
                logger.warning("Synthesis will fail - CosyVoice2 requires a voice reference audio")

            # 根据参数选择推理方法
            if temp_wav_path:
                # 验证音频文件可以被读取
                try:
                    import soundfile as sf
                    audio_info = sf.info(temp_wav_path)
                    logger.info(f"✓ Verified prompt_wav audio properties:")
                    logger.info(f"  - Sample rate: {audio_info.samplerate} Hz")
                    logger.info(f"  - Duration: {audio_info.duration:.2f} seconds")
                    logger.info(f"  - Channels: {audio_info.channels}")
                    logger.info(f"  - Format: {audio_info.format}")
                except Exception as e:
                    logger.error(f"✗ Failed to read prompt_wav audio file: {e}")
                    raise HTTPException(status_code=500, detail=f"Invalid audio file: {str(e)}")

                if instruction_text:
                    # 有 instruction - 使用 instruct2 模式 (instruction + voice)
                    logger.info(f"[{model_type}] Mode: INSTRUCT2 (instruction + voice reference)")
                    logger.info(f"  → Using inference_instruct2")
                    logger.info(f"  - Text: '{text[:50]}...' (len={len(text)})")
                    logger.info(f"  - Instruction: '{instruction_text[:80]}...'")
                    logger.info(f"  - Voice reference: {os.path.basename(temp_wav_path)}")
                    
                    # 确保使用绝对路径
                    abs_wav_path = os.path.abspath(temp_wav_path)
                    
                    inference_method = lambda: cosyvoice.inference_instruct2(
                        text,
                        instruction_text,
                        abs_wav_path, 
                        stream=True
                    )
                else:
                    # 无 instruction - 使用 zero_shot 模式 (纯声音克隆)
                    # 需要 prompt_text (音频文件的文字内容)
                    actual_prompt_text = prompt_text if prompt_text else default_prompt_text

                    logger.info(f"[{model_type}] Mode: ZERO_SHOT (voice cloning)")
                    logger.info(f"  → Using inference_zero_shot for voice cloning")
                    logger.info(f"  - Text: '{text[:50]}...' (len={len(text)})")
                    logger.info(f"  - Prompt text: '{actual_prompt_text}'")
                    logger.info(f"  - Voice reference: {os.path.basename(temp_wav_path)}")
                    logger.info(f"  - Voice will MATCH the prompt audio")
                    
                    # 确保使用绝对路径
                    abs_wav_path = os.path.abspath(temp_wav_path)
                    logger.info(f"  - Absolute path: {abs_wav_path}")
                    logger.info(f"  - File exists: {os.path.exists(abs_wav_path)}")

                    inference_method = lambda: cosyvoice.inference_zero_shot(
                        text,
                        'You are a helpful assistant.<|endofprompt|>' + actual_prompt_text,
                        abs_wav_path, 
                        stream=True
                    )
            else:
                # 没有音频文件
                raise HTTPException(
                    status_code=400,
                    detail=f"CosyVoice2 requires a voice reference audio file. Place default file at: {default_prompt_wav} or upload prompt_wav"
                )

            async def audio_stream():
                try:
                    for result in inference_method():
                        # 正确处理 tensor: squeeze() 去除多余维度, cpu() 移到 CPU
                        audio_chunk = result["tts_speech"].squeeze().cpu().numpy()
                        
                        # 返回原始 Float32 PCM 数据（不带任何文件头）
                        # 客户端需要知道采样率是 24000 Hz
                        yield audio_chunk.astype('float32').tobytes()
                finally:
                    # 只清理上传的临时文件,不清理默认文件
                    if prompt_wav and temp_wav_path and os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
                        logger.info(f"Cleaned up temporary file: {temp_wav_path}")

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
            logger.info(f"Complete synthesis - voice: {abs_wav_path}")
            
            # 收集所有音频块
            import numpy as np
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
                import io
                import soundfile as sf
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50000,
        workers=1,              # 单进程避免内存重复加载模型
        limit_concurrency=10,   # 最大10个并发连接
        timeout_keep_alive=30,  # 30秒keepalive超时
        log_level="warning"     # 只记录警告和错误，减少日志量
    )
