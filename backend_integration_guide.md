# CosyVoice Backend Integration Guide

This document provides instructions on how to integrate your backend program with CosyVoice's streaming service. The streaming service allows real-time text-to-speech (TTS) generation and audio streaming.

---

## 1. Overview
The `stream_service.py` script sets up a streaming service that communicates via standard input and output. Clients can send text data to the service through standard input, and the service will respond with audio data through standard output.

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
python stream_service.py
```

The service will be ready to receive requests via standard input.

---

## 3. Client Integration

### Standard Input/Output Communication
Clients can interact with the service by sending JSON requests through standard input and receiving audio data through standard output.

#### Example Client Code (Python):
```python
import subprocess
import json

def send_request_to_service(text, speaker="中文女"):
    # Start the service process
    process = subprocess.Popen(
        ["python", "stream_service.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Construct the request
    request = json.dumps({"text": text, "speaker": speaker}) + "\n"

    # Send the request to the service
    process.stdin.write(request)
    process.stdin.flush()

    # Receive the audio data
    audio_data = b""
    while True:
        chunk = process.stdout.buffer.read(1024)
        if b"__END__" in chunk:
            audio_data += chunk.replace(b"__END__", b"")
            break
        audio_data += chunk

    # Close the service process
    process.stdin.close()
    process.terminate()

    return audio_data

# Example usage
audio = send_request_to_service("你好，我是通义生成式语音大模型。")
with open("output.wav", "wb") as f:
    f.write(audio)
```

### Notes:
- Ensure the `stream_service.py` script is in the same directory as the client code, or adjust the path accordingly.
- The service sends audio data in chunks. Ensure the client writes the received data to a file or processes it in real-time.

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
   - Verify that the client is correctly handling the `__END__` marker.
3. **Performance Issues**:
   - Use GPU acceleration for better performance.
   - Ensure sufficient system resources are available.

---

## 6. Additional Resources
- [Python Subprocess Documentation](https://docs.python.org/3/library/subprocess.html)
- [CosyVoice GitHub Repository](https://github.com/ElasticDash-Official/CosyVoice)

---

For further assistance, please contact the CosyVoice development team.