#!/bin/bash
# Example: Fuse Lightning LoRA into Qwen-Image-Edit base model

python examples/qwen_image/model_training/utils/fuse_lora_weights.py \
  --lora_path "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors" \
  --output_dir "./models/Qwen-Image-Edit-Lightning-Fused" \
  --target_model "dit" \
  --lora_alpha 1.0 \
  --save_precision "bf16"

# After running this, you can use the fused model for:
# 1. Training a new LoRA on top
# 2. Direct inference without runtime LoRA overhead
