from fastapi import FastAPI, WebSocket
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()

# 初始化 CosyVoice 模型
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M', load_jit=True, load_trt=True, load_vllm=True, fp16=True)

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收客户端发送的文本
            data = await websocket.receive_text()
            speaker = "中文女"  # 您可以根据需求动态设置
            # 流式生成语音
            for i, result in enumerate(cosyvoice.inference_sft(data, speaker, stream=True)):
                # 将生成的音频块发送给客户端
                await websocket.send_bytes(result['tts_speech'].numpy().tobytes())
    except Exception as e:
        print(f"连接关闭：{e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50000)