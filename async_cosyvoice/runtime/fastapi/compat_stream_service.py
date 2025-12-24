import io
import os
import sys
import time
import uuid
import logging
from typing import Optional, AsyncGenerator

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/../../..")

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from utils import load_audio_from_bytes

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI()
cosyvoice: AsyncCosyVoice2 | None = None
MODEL_DIR: Optional[str] = None

# Default prompt text fallback
DEFAULT_PROMPT_TEXT = "这是参考音频对应的文本"

async def _gen_spk_id_from_prompt(prompt_wav: UploadFile, prompt_text: str) -> str:
    data = await prompt_wav.read()
    prompt_speech_16k = load_audio_from_bytes(data, 16000)
    spk_id = f"spk:{uuid.uuid4().hex[:8]}"
    cosyvoice.frontend.generate_spk_info(
        spk_id,
        prompt_text,
        prompt_speech_16k,
        24000,
        "uploaded"
    )
    return spk_id

async def _audio_stream_from_spk(tts_text: str, instruct_text: Optional[str], spk_id: str, sample_rate: int) -> AsyncGenerator[bytes, None]:
    total_samples = 0
    start_ts = time.perf_counter()

    try:
        if instruct_text:
            gen = cosyvoice.inference_instruct2_by_spk_id(tts_text, instruct_text, spk_id, stream=True, speed=1.0, text_frontend=True)
        else:
            gen = cosyvoice.inference_zero_shot_by_spk_id(tts_text, spk_id, stream=True, speed=1.0, text_frontend=True)

        async for chunk in gen:
            audio = chunk["tts_speech"].squeeze()
            total_samples += int(audio.shape[-1])
            # Stream raw PCM int16 bytes
            audio_int16 = (audio.float().clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
            yield audio_int16.tobytes()
    finally:
        wall = time.perf_counter() - start_ts
        audio_sec = total_samples / float(sample_rate) if total_samples else 0.0
        rtf = wall / audio_sec if audio_sec > 0 else 0.0
        logging.warning("[vLLM] /synthesize done wall=%.2fs audio=%.2fs rtf=%.3f", wall, audio_sec, rtf)

@app.post("/synthesize")
async def synthesize_streaming(
    text: str = Form(...),
    instruction: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    prompt_text: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None)
):
    try:
        if cosyvoice is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        # Build spk_id
        if prompt_wav is not None:
            spk_id = await _gen_spk_id_from_prompt(prompt_wav, prompt_text or DEFAULT_PROMPT_TEXT)
        elif speaker:
            spk_id = speaker
        else:
            raise HTTPException(status_code=400, detail="Voice reference required (speaker or prompt_wav)")

        sample_rate = getattr(cosyvoice, "sample_rate", 24000)
        # Return raw PCM stream for compatibility with clients expecting octet-stream
        return StreamingResponse(_audio_stream_from_spk(text, instruction, spk_id, sample_rate), media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error during synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_complete")
async def synthesize_complete(
    text: str = Form(...),
    instruction: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None)
):
    """
    Non-streaming synthesis endpoint: returns a single WAV file
    """
    try:
        if cosyvoice is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        # Build spk_id from uploaded prompt or error
        if prompt_wav is not None:
            data = await prompt_wav.read()
            prompt_speech_16k = load_audio_from_bytes(data, 16000)
            spk_id = f"spk:{uuid.uuid4().hex[:8]}"
            cosyvoice.frontend.generate_spk_info(
                spk_id,
                DEFAULT_PROMPT_TEXT,
                prompt_speech_16k,
                24000,
                "uploaded"
            )
        else:
            raise HTTPException(status_code=400, detail="No voice reference audio")

        # Collect all audio chunks (non-stream)
        chunks: list[np.ndarray] = []
        sample_rate = getattr(cosyvoice, "sample_rate", 24000)

        if instruction:
            gen = cosyvoice.inference_instruct2_by_spk_id(text, instruction, spk_id, stream=False, speed=1.0, text_frontend=True)
        else:
            gen = cosyvoice.inference_zero_shot_by_spk_id(text, spk_id, stream=False, speed=1.0, text_frontend=True)

        async for result in gen:
            chunks.append(result["tts_speech"].squeeze().cpu().numpy())

        if not chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        full_audio = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, full_audio, sample_rate, format='WAV')
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error during complete synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/model_info")
async def model_info():
    if cosyvoice is None:
        return {"status": "uninitialized"}
    return {
        "model_type": "CosyVoice2",
        "model_dir": MODEL_DIR,
        "sample_rate": getattr(cosyvoice, "sample_rate", 24000),
        "usage": "Use /synthesize endpoint with 'instruction' parameter (CosyVoice2/3)",
        "endpoint": "/synthesize",
    }

@app.get("/speakers")
async def get_speakers():
    # CosyVoice2 does not expose v1 speaker list; mirror original behavior
    raise HTTPException(
        status_code=400,
        detail="Speakers list is only available for CosyVoice v1. Current model is CosyVoice2."
    )

if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='../../../pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--load_jit', action='store_true')
    parser.add_argument('--load_trt', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    cosyvoice = AsyncCosyVoice2(args.model_dir, load_jit=args.load_jit, load_trt=args.load_trt, fp16=args.fp16)
    MODEL_DIR = args.model_dir
    uvicorn.run(app, host=args.host, port=args.port)
