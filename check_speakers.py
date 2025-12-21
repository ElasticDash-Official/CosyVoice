from cosyvoice.cli.cosyvoice import AutoModel

# 初始化模型
cosyvoice = AutoModel(
    model_dir="/home/ec2-user/CosyVoice/pretrained_models/CosyVoice2-0.5B"
)

# 列出所有可用的speaker
available_spks = cosyvoice.list_available_spks()
print("Available speakers:")
for spk in available_spks:
    print(f"  - {spk}")
