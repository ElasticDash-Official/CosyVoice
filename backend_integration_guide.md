# CosyVoice Backend Integration Guide

This document provides instructions on how to integrate your backend program with CosyVoice's streaming service. The streaming service allows real-time text-to-speech (TTS) generation and audio streaming.

---

## 1. Overview
The `stream_service.py` script sets up a streaming service using FastAPI. Clients can send text data to the service via HTTP POST requests, and the service will respond with audio data in a streaming format.

### Key Features:
- Real-time TTS generation with CosyVoice2.
- Streamed audio data sent directly to the client.
- Instruction-based voice synthesis with customizable tone and style.
- Optional custom voice prompts via audio file upload.

---

## 2. Running the Streaming Service

### Prerequisites:
1. **Python Environment**:
   - Ensure Python 3.8+ is installed.
   - Install required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
2. **CosyVoice2 Model**:
   - Ensure the `pretrained_models` directory contains the CosyVoice2 model (e.g., `CosyVoice2-0.5B`).
   - Default model path: `/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B`
3. **Default Prompt Audio**:
   - Place a default prompt WAV file at `./asset/zero_shot_prompt.wav`
   - This audio file serves as the voice reference for synthesis
4. **CUDA (Optional)**:
   - If using GPU acceleration, ensure CUDA and TensorRT are properly installed.

### Starting the Service:
The service can be started using the system service:
```bash
systemctl start stream_service
systemctl status stream_service
```

Or manually with uvicorn:
```bash
uvicorn stream_service:app --host 0.0.0.0 --port 50000
```

The service will be ready to receive HTTP requests at `http://<server-ip>:50000`.

---

## 3. Client Integration

### HTTP Streaming Communication
Clients can interact with the service by sending HTTP POST requests to the `/synthesize` endpoint and receiving audio data as a streaming response.

#### API Endpoint: `/synthesize`

**Method:** POST
**Content-Type:** `multipart/form-data`

**Parameters:**
- `text` (required): The text to synthesize
- `instruction` (optional): Voice instruction to control tone and style
  - Format: Instruction text followed by `<|endofprompt|>` marker
  - If not provided, uses audio-only mode (cross-lingual synthesis)
- `prompt_wav` (optional): Custom voice prompt audio file (WAV format)
  - If not provided, uses the default prompt at `./asset/zero_shot_prompt.wav`
- `prompt_text` (optional): Reference text for zero-shot synthesis

**Synthesis Modes:**
CosyVoice2 supports three synthesis modes:

1. **Instruction + Audio (instruct2 mode)**: Most flexible
   - Provide both `instruction` and `prompt_wav`
   - `instruction` controls speaking style (tone, emotion)
   - `prompt_wav` provides voice timbre (pitch, voice characteristics)

2. **Audio-Only (cross-lingual mode)**: Direct voice cloning
   - Provide only `prompt_wav` (no `instruction`)
   - Voice will directly match the prompt audio
   - Best for voice cloning without style modification

3. **Zero-Shot (zero_shot mode)**: With reference text
   - Provide `prompt_text` and `prompt_wav`
   - Uses reference text for better voice matching

#### Example 1: Basic Usage with Default Settings
```python
import requests

def synthesize_with_default(text):
    """
    Uses default instruction (restaurant clerk voice)
    """
    url = "http://<server-ip>:50000/synthesize"
    data = {
        "text": text
    }

    response = requests.post(url, data=data, stream=True)
    if response.status_code == 200:
        with open("output_stream.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Audio saved to output_stream.wav")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
synthesize_with_default("欢迎光临!请问您需要什么?")
```

#### Example 2: Custom Instruction
```python
import requests

def synthesize_with_instruction(text, instruction):
    """
    Customize voice style using instruction
    """
    url = "http://<server-ip>:50000/synthesize"
    data = {
        "text": text,
        "instruction": instruction
    }

    response = requests.post(url, data=data, stream=True)
    if response.status_code == 200:
        with open("output_stream.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Audio saved to output_stream.wav")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example: News anchor style
instruction = "你是一位专业的新闻主播,声音清晰,语调平稳,充满权威感。<|endofprompt|>"
synthesize_with_instruction("今天的新闻报道到此结束。", instruction)
```

#### Example 3: Custom Voice Prompt
```python
import requests

def synthesize_with_custom_voice(text, instruction, prompt_wav_path):
    """
    Use custom voice prompt audio file
    """
    url = "http://<server-ip>:50000/synthesize"
    data = {
        "text": text,
        "instruction": instruction
    }
    files = {
        "prompt_wav": open(prompt_wav_path, "rb")
    }

    response = requests.post(url, data=data, files=files, stream=True)
    if response.status_code == 200:
        with open("output_stream.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Audio saved to output_stream.wav")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
instruction = "你是一位温柔的客服人员,说话柔和体贴。<|endofprompt|>"
synthesize_with_custom_voice(
    "感谢您的来电,我们会尽快处理。",
    instruction,
    "./my_custom_voice.wav"
)
```

#### Example 4: Audio-Only Mode (Voice Cloning)
```python
import requests

def synthesize_audio_only(text, prompt_wav_path):
    """
    Audio-only mode - voice will directly match the prompt audio
    No instruction needed - direct voice cloning
    """
    url = "http://<server-ip>:50000/synthesize"
    data = {
        "text": text
        # NO instruction parameter - will use cross-lingual mode
    }
    files = {
        "prompt_wav": open(prompt_wav_path, "rb")
    }

    response = requests.post(url, data=data, files=files, stream=True)
    if response.status_code == 200:
        with open("output_stream.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Audio saved to output_stream.wav")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage - voice will match paimon_prompt.wav exactly
synthesize_audio_only(
    "原神,启动!",
    "./asset/paimon_prompt.wav"
)
```

#### Example 5: Using cURL
```bash
# Basic usage with default settings
curl -X POST "http://<server-ip>:50000/synthesize" \
  -F "text=欢迎光临!请问您需要什么?" \
  --output output.wav

# With custom instruction
curl -X POST "http://<server-ip>:50000/synthesize" \
  -F "text=今天天气真好" \
  -F "instruction=你是一位活泼开朗的主持人,声音充满活力。<|endofprompt|>" \
  --output output.wav

# With custom voice prompt
curl -X POST "http://<server-ip>:50000/synthesize" \
  -F "text=您好,有什么可以帮您?" \
  -F "instruction=你是一位专业的客服人员。<|endofprompt|>" \
  -F "prompt_wav=@./my_voice.wav" \
  --output output.wav

# Audio-only mode (voice cloning, no instruction)
curl -X POST "http://<server-ip>:50000/synthesize" \
  -F "text=原神,启动!" \
  -F "prompt_wav=@./asset/paimon_prompt.wav" \
  --output output.wav
```

### Notes:
- Replace `<server-ip>` with the actual IP address or hostname of the server running the service.
- Ensure the service is running and accessible on port 50000.
- The response is streamed audio data in raw PCM format.
- Custom prompt WAV files should be clear voice recordings (16kHz or 22050Hz recommended).

---

## 4. Customization

### Customizing Voice Style with Instructions
CosyVoice2 uses instruction-based synthesis. You can customize the voice by providing different instructions that describe the desired tone, style, and characteristics:

**Instruction Format:**
```
[Your instruction describing the voice style]<|endofprompt|>
```

**Example Instructions:**
```python
# Professional news anchor
"你是一位专业的新闻主播,声音清晰,语调平稳,充满权威感。<|endofprompt|>"

# Friendly customer service
"你是一位温柔的客服人员,说话柔和体贴,充满耐心。<|endofprompt|>"

# Energetic host
"你是一位活泼开朗的主持人,声音充满活力和热情。<|endofprompt|>"

# Calm teacher
"你是一位温和的老师,说话清晰缓慢,语气平和亲切。<|endofprompt|>"

# Professional salesperson
"你是一位专业的销售人员,说话自信流畅,语气友好。<|endofprompt|>"
```

### Customizing Voice with Audio Prompts
You can also customize the voice timbre by providing a custom WAV audio file as a voice prompt:

```python
files = {
    "prompt_wav": open("./your_custom_voice.wav", "rb")
}
data = {
    "text": "Your text here",
    "instruction": "Your instruction<|endofprompt|>"
}
response = requests.post(url, data=data, files=files, stream=True)
```

**Audio Requirements:**
- Format: WAV file
- Recommended sample rate: 16kHz or 22050Hz
- Duration: 3-10 seconds of clear speech
- Quality: Clear voice without background noise

### Using a Different Model
To use a different pretrained model, update the `model_dir` parameter in the `stream_service.py` script:

```python
# CosyVoice2-0.5B (default)
model_dir = "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B"

# Or use CosyVoice3
# model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"

cosyvoice = AutoModel(model_dir=model_dir)
```

### Configuring Default Settings
Edit the default settings in `stream_service.py`:

```python
# Default instruction (restaurant clerk)
default_instruction = "你是一位热情友好的餐馆店员,说话温柔亲切,语气礼貌专业。<|endofprompt|>"

# Default prompt audio file
default_prompt_wav = "./asset/zero_shot_prompt.wav"
```

---

## 5. Troubleshooting

### Common Issues

1. **Service Not Responding**:
   - Check if the service is running:
     ```bash
     systemctl status stream_service
     ```
   - Check the service logs:
     ```bash
     journalctl -u stream_service -f
     ```
   - Ensure the service is listening on port 50000

2. **Audio Data Not Received**:
   - Verify the CosyVoice2 model is properly loaded (check logs for "Loaded model type: CosyVoice2")
   - Ensure the `pretrained_models/CosyVoice2-0.5B` directory contains all required files
   - Verify the client is correctly handling the streaming response
   - Check that the default prompt WAV file exists at `./asset/zero_shot_prompt.wav`

3. **Default Prompt Audio Missing**:
   - Error: "No prompt_wav provided and default file not found"
   - Solution: Place a WAV file at `./asset/zero_shot_prompt.wav` or provide a `prompt_wav` file in your request

4. **Model Loading Issues**:
   - Check if the model directory path is correct in `stream_service.py`
   - Verify all model files are downloaded and not corrupted
   - Ensure sufficient disk space for model files (~1-2GB)
   - Check logs for model loading errors during service startup

5. **Performance Issues**:
   - Use GPU acceleration for better performance (CUDA required)
   - Monitor system resources (RAM, GPU memory)
   - Consider reducing concurrent connections (default: 10)
   - Ensure sufficient GPU memory (recommended: 4GB+ VRAM)

6. **Instruction Format Errors**:
   - Ensure instructions end with `<|endofprompt|>` marker
   - Instructions should be descriptive and in Chinese for best results
   - Example: `"你是一位友好的店员。<|endofprompt|>"`

### Checking Service Status

```bash
# Check if service is running
systemctl status stream_service

# View service logs
journalctl -u stream_service -f

# Restart service
systemctl restart stream_service

# Test API endpoint
curl http://localhost:50000/health
curl http://localhost:50000/model_info
```

### Debugging Client Issues

```python
# Test basic connectivity
import requests

# Health check
response = requests.get("http://<server-ip>:50000/health")
print(response.json())  # Should return: {"status": "ok", "model_type": "CosyVoice2"}

# Get model info
response = requests.get("http://<server-ip>:50000/model_info")
print(response.json())
```

---

## 6. API Reference

### Available Endpoints

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_type": "CosyVoice2"
}
```

#### GET `/model_info`
Get information about the loaded model.

**Response:**
```json
{
  "model_type": "CosyVoice2",
  "model_dir": "/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B",
  "default_instruction": "你是一位热情友好的餐馆店员,说话温柔亲切,语气礼貌专业。<|endofprompt|>",
  "usage": "Use /synthesize endpoint with 'instruction' parameter (CosyVoice2/3)",
  "endpoint": "/synthesize"
}
```

#### POST `/synthesize`
Main synthesis endpoint. See Section 3 for detailed usage.

---

## 7. Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CosyVoice GitHub Repository](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice2 Model](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)

---

For further assistance, please contact the CosyVoice development team or open an issue on GitHub.
