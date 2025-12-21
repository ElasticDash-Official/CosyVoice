# CosyVoice Backend Integration Guide

This document provides instructions on how to integrate your backend program with CosyVoice's streaming service. The streaming service allows real-time text-to-speech (TTS) generation and audio streaming.

---

## 1. Overview
The `stream_service.py` script sets up a streaming service using FastAPI. Clients can send text data to the service via HTTP POST requests, and the service will respond with audio data in a hex-encoded format.

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
     pip install -r requirements.txt
     ```
2. **CosyVoice Dependencies**:
   - Ensure the `pretrained_models` directory contains the required models (e.g., `Fun-CosyVoice3-0.5B`).
3. **CUDA (Optional)**:
   - If using GPU acceleration, ensure CUDA and TensorRT are properly installed.

### Starting the Service:
Run the following command to start the service:
```bash
uvicorn stream_service:app --host 0.0.0.0 --port 50000
```

The service will be ready to receive HTTP requests at `http://<server-ip>:50000`.

---

## 3. Client Integration

### HTTP Communication
Clients can interact with the service by sending HTTP POST requests to the `/synthesize` endpoint and receiving audio data in a hex-encoded format.

#### Example Client Code (Python):
```python
import requests
import json

def send_request_to_service(text, speaker="中文女"):
    url = "http://<server-ip>:50000/synthesize"
    data = {
        "text": text,
        "speaker": speaker
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        audio_hex = response.json()["audio"]
        audio_data = bytes.fromhex(audio_hex)
        return audio_data
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
audio = send_request_to_service("你好，我是通义生成式语音大模型。")
with open("output.wav", "wb") as f:
    f.write(audio)
```

### Notes:
- Replace `<server-ip>` with the actual IP address or hostname of the server running the service.
- Ensure the service is running and accessible on port 50000.

---

## 4. Customization

### Changing the Speaker
The default speaker is set to `"中文女"`. You can modify this in the client request:
```json
{
  "text": "你好，我是通义生成式语音大模型。",
  "speaker": "中文男"
}
```

### Using a Different Model
To use a different pretrained model, update the `model_dir` parameter in the `stream_service.py` script:
```python
cosyvoice = AutoModel(model_dir='/path/to/your/model', ...)
```

---

## 5. Troubleshooting

### Common Issues:
1. **Service Not Responding**:
   - Ensure the `stream_service.py` script is running.
   - Check the service logs for errors.
2. **Audio Data Not Received**:
   - Ensure the `pretrained_models` directory contains the required models.
   - Verify that the client is correctly decoding the hex-encoded audio data.
3. **Performance Issues**:
   - Use GPU acceleration for better performance.
   - Ensure sufficient system resources are available.

---

## 6. Additional Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CosyVoice GitHub Repository](https://github.com/ElasticDash-Official/CosyVoice)

---

For further assistance, please contact the CosyVoice development team.