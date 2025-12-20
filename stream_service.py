# cosyvoice_service.py
import os
import json
import socket
import struct
from cosyvoice.cli.cosyvoice import AutoModel

SOCK_PATH = "/tmp/cosyvoice.sock"

if os.path.exists(SOCK_PATH):
    os.remove(SOCK_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCK_PATH)
server.listen(1)

cosyvoice = AutoModel(
    model_dir="/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
)

print("CosyVoice service ready", flush=True)

while True:
    conn, _ = server.accept()
    with conn:
        data = conn.recv(8192)
        req = json.loads(data.decode())

        text = req["text"]
        speaker = req.get("speaker", "中文女")

        for result in cosyvoice.inference_sft(text, speaker, stream=True):
            audio = result["tts_speech"].numpy().tobytes()
            # length-prefix framing
            conn.sendall(struct.pack(">I", len(audio)))
            conn.sendall(audio)

        conn.sendall(struct.pack(">I", 0))  # END
