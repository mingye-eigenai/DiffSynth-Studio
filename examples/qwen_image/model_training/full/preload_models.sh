#!/bin/bash
# Pre-download all required models to avoid download during training
# Run this once on each node before training

echo "Pre-downloading models for training..."

cd /home/kuan/workspace/repos/DiffSynth-Studio

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nunchaku

# Run a quick import to trigger model downloads
python3 << 'EOF'
from diffsynth import ModelManager
import os

# Set model paths
model_id_with_origin_paths = "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"

print("Downloading/verifying Qwen-Image-Edit models...")
manager = ModelManager()
manager.fetch_model(model_id="Qwen/Qwen-Image-Edit")

print("Downloading/verifying Qwen-Image models...")
manager.fetch_model(model_id="Qwen/Qwen-Image")

print("✅ All models downloaded successfully!")
print(f"Models cached in: {os.path.expanduser('~/.cache/modelscope')}")
EOF

echo "✅ Pre-loading complete on $(hostname)"

