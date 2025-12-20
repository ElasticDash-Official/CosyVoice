# CosyVoice Backend Integration Guide

This document provides instructions on how to integrate your backend program with CosyVoice's streaming service. The streaming service allows real-time text-to-speech (TTS) generation and audio streaming.

---

## 1. Overview
The `stream_service.py` script sets up a streaming service that communicates via a Unix domain socket. Clients can send text data to the service through the socket, and the service will respond with audio data in a length-prefixed binary format.

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

The service will be ready to receive requests via the Unix domain socket at `/tmp/cosyvoice.sock`.

---

## 3. Client Integration

### Unix Domain Socket Communication
Clients can interact with the service by sending JSON requests through the Unix domain socket and receiving audio data in a length-prefixed binary format.

#### Example Client Code (Python):
```python
import socket
import struct
import json

def send_request_to_service(text, speaker="中文女"):
    # Connect to the Unix domain socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/cosyvoice.sock")

    # Construct the request
    request = json.dumps({"text": text, "speaker": speaker}).encode()

    # Send the request to the service
    sock.sendall(request)

    # Receive the audio data
    audio_data = b""
    while True:
        # Read the length prefix
        length_prefix = sock.recv(4)
        if not length_prefix:
            break

        length = struct.unpack(">I", length_prefix)[0]
        if length == 0:  # End of stream
            break

        # Read the audio chunk
        chunk = sock.recv(length)
        audio_data += chunk

    sock.close()
    return audio_data

# Example usage
audio = send_request_to_service("你好，我是通义生成式语音大模型。")
with open("output.wav", "wb") as f:
    f.write(audio)
```

### Notes:
- Ensure the `stream_service.py` script is running and the socket path matches the client code.
- The service sends audio data in chunks with a length-prefix framing. Ensure the client processes the framing correctly.

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
   - Verify that the client is correctly handling the length-prefixed framing.
3. **Performance Issues**:
   - Use GPU acceleration for better performance.
   - Ensure sufficient system resources are available.

---

## 6. Additional Resources
- [Python Socket Programming Documentation](https://docs.python.org/3/library/socket.html)
- [CosyVoice GitHub Repository](https://github.com/ElasticDash-Official/CosyVoice)

---

For further assistance, please contact the CosyVoice development team.