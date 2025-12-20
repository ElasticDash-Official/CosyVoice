# CosyVoice Backend Integration Guide

This document provides instructions on how to integrate your backend program with CosyVoice's streaming service. The streaming service allows real-time text-to-speech (TTS) generation and audio streaming.

---

## 1. Overview
The `stream_service.py` script sets up a WebSocket-based streaming service using FastAPI. Clients can send text data to the server, and the server will respond with audio data in real-time.

### Key Features:
- Real-time TTS generation.
- Streamed audio data sent directly to the client.
- Configurable speaker and model settings.

---

## 2. Running the Streaming Service

### Prerequisites:
1. **Python Environment**:
   - Ensure Python 3.8+ is installed.
   - Install required dependencies:
     ```bash
     pip install fastapi uvicorn
     ```
2. **CosyVoice Dependencies**:
   - Install all dependencies listed in `requirements.txt`.
   - Ensure the `pretrained_models` directory contains the required models (e.g., `CosyVoice-300M`).
3. **CUDA (Optional)**:
   - If using GPU acceleration, ensure CUDA and TensorRT are properly installed.

### Starting the Service:
Run the following command to start the WebSocket server:
```bash
uvicorn stream_service:app --host 0.0.0.0 --port 50000
```

The server will be accessible at `ws://<server_ip>:50000/stream`.

---

## 3. Client Integration

### WebSocket Communication
Clients can connect to the WebSocket endpoint `/stream` to send text data and receive audio data.

#### Example Client Code (Python):
```python
import asyncio
import websockets

async def stream_audio():
    uri = "ws://<server_ip>:50000/stream"
    async with websockets.connect(uri) as websocket:
        # Send text data to the server
        await websocket.send("你好，我是通义生成式语音大模型。")

        # Receive streamed audio data
        with open("output.wav", "wb") as f:
            while True:
                try:
                    audio_chunk = await websocket.recv()
                    f.write(audio_chunk)
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

asyncio.run(stream_audio())
```

### Notes:
- Replace `<server_ip>` with the actual IP address or domain of your server.
- The server sends audio data in chunks. Ensure the client writes the received data to a file or processes it in real-time.

---

## 4. Customization

### Changing the Speaker
The default speaker is set to `"中文女"`. You can modify this in the `stream_service.py` script:
```python
speaker = "中文女"  # Change this to your desired speaker
```

### Using a Different Model
To use a different pretrained model, update the `model_dir` parameter in the script:
```python
cosyvoice = AutoModel(model_dir='pretrained_models/YourModelName', ...)
```

---

## 5. Troubleshooting

### Common Issues:
1. **WebSocket Connection Fails**:
   - Ensure the server is running and the port (default: 50000) is open in the firewall/security group.
2. **Audio Data Not Received**:
   - Check the server logs for errors.
   - Ensure the `pretrained_models` directory contains the required models.
3. **Performance Issues**:
   - Use GPU acceleration for better performance.
   - Increase the number of workers when starting the server:
     ```bash
     uvicorn stream_service:app --host 0.0.0.0 --port 50000 --workers 4
     ```

---

## 6. Additional Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [CosyVoice GitHub Repository](https://github.com/ElasticDash-Official/CosyVoice)

---

For further assistance, please contact the CosyVoice development team.