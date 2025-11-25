#!/bin/bash

# Example script demonstrating how to use the lora_fused argument
# This script fuses a pretrained LoRA (e.g., Lightning LoRA) into the base weights
# before training a new LoRA on top of it

accelerate launch examples/qwen_image/model_training/train.py \
  --task "sft" \
  --dataset_base_path "datasets/diffusion_db_aesthetics_6_5plus" \
  --trainable_models "" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_fused "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --output_path "./models/Qwen-Image-Lightning-Fused-NewLoRA" \
  --remove_prefix_in_ckpt "pipe.dit."

# Note: The --lora_fused parameter specifies a pretrained LoRA that will be fused
# into the base model weights BEFORE training the new LoRA.
# This allows you to build on top of existing LoRA models.
