# cosyvoice_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import AutoModel
from typing import Optional
import numpy as np
import os
import json
import socket
import struct

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 CosyVoice 模型
cosyvoice = AutoModel(
    model_dir="/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
)

# 定义请求模型
class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = "中文女"

# 定义路由
@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    try:
        # 调用 CosyVoice 模型生成音频
        audio_data = b""
        for result in cosyvoice.inference_sft(request.text, request.speaker, stream=True):
            audio_chunk = result["tts_speech"].numpy().tobytes()
            audio_data += audio_chunk

        # 返回音频数据
        return {"audio": audio_data.hex()}  # 将音频数据转换为十六进制字符串
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查路由
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50000)
