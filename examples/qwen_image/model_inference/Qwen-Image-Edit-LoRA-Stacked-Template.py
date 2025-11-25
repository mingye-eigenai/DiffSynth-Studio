"""
Template for using LoRAs that were trained on top of other LoRAs (e.g., Lightning LoRA).
This shows how to properly stack LoRAs for inference to match the training setup.
"""

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
import torch
from PIL import Image

# Step 1: Initialize pipeline with base Qwen-Image weights
print("Loading base model...")
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# Step 2: Fuse base LoRA (e.g., Lightning LoRA) if your model was trained on top of it
# This step is ONLY needed if you used --lora_fused during training
base_lora_path = "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
print(f"\nFusing base LoRA from: {base_lora_path}")

# Load and detect LoRA format
base_lora_state_dict = load_state_dict(base_lora_path)
is_lightning_format = any("lora_down" in k or "lora_up" in k for k in base_lora_state_dict.keys())

if is_lightning_format:
    print("Detected Lightning LoRA format")
    lora_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
else:
    print("Detected standard LoRA format")
    lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")

num_updated = lora_loader.load(pipe.dit, base_lora_state_dict, alpha=1.0)
print(f"Base LoRA fused: {num_updated} tensors updated")

# Step 3: Load your custom LoRA on top
custom_lora_path = "./models/train/your_custom_lora/epoch-4.safetensors"
print(f"\nLoading custom LoRA from: {custom_lora_path}")
pipe.load_lora(pipe.dit, custom_lora_path, alpha=1.0)

# Step 4: Run inference
print("\nRunning inference...")
input_image = Image.open("path/to/your/image.jpg").convert("RGB")

image = pipe(
    prompt="Your edit prompt here",
    edit_image=input_image,
    seed=42,
    num_inference_steps=40,  # Use more steps for Lightning LoRA (e.g., 4-8 steps)
    height=1024,
    width=1024,
    edit_image_auto_resize=True,
)

image.save("output.jpg")
print("Done! Saved to output.jpg")

# Notes:
# 1. The order matters! Always fuse base LoRAs first, then load your custom LoRA
# 2. If you didn't use --lora_fused during training, skip Step 2
# 3. For Lightning LoRA, you can use fewer inference steps (4-8) for faster generation
# 4. Make sure alpha values match what you used during training
