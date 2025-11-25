#!/bin/bash
# Example: Fuse a custom-trained LoRA into the base model

# For a standard format LoRA (e.g., one you trained yourself)
python examples/qwen_image/model_training/utils/fuse_lora_weights.py \
  --lora_path "./models/your_custom_lora.safetensors" \
  --output_dir "./models/Qwen-Image-Edit-CustomLoRA-Fused" \
  --target_model "dit" \
  --lora_alpha 0.8 \
  --save_precision "bf16"

# You can also specify custom model paths if using non-default models
# python examples/qwen_image/model_training/utils/fuse_lora_weights.py \
#   --model_id_with_origin_paths "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors" \
#   --lora_path "./models/your_lora.safetensors" \
#   --output_dir "./models/fused_output" \
#   --target_model "dit" \
#   --lora_alpha 1.0
