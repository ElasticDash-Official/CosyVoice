import sys
import json
from cosyvoice.cli.cosyvoice import AutoModel

cosyvoice = AutoModel(
    model_dir="/home/ec2-user/CosyVoice/pretrained_models/CosyVoice-300M",
    load_jit=True,
    load_trt=True,
    load_vllm=True,
    fp16=True,
)

print("CosyVoice service ready", flush=True)

for line in sys.stdin:
    req = json.loads(line)
    text = req["text"]
    speaker = req.get("speaker", "中文女")

    for result in cosyvoice.inference_sft(text, speaker, stream=True):
        audio = result["tts_speech"].numpy().tobytes()
        sys.stdout.buffer.write(audio)
        sys.stdout.flush()

    # chunk 结束标记
    sys.stdout.buffer.write(b"__END__")
    sys.stdout.flush()
