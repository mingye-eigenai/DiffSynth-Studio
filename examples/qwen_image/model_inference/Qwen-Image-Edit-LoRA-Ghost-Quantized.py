from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
from diffsynth.lora import GeneralLoRALoader
import torch
from PIL import Image
from pathlib import Path

# Initialize pipeline WITHOUT quantization first for LoRA fusion
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

# Step 1: Fuse Lightning LoRA into base model first
lightning_lora_path = "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
print(f"Step 1: Fusing Lightning LoRA from: {lightning_lora_path}")

# Load Lightning LoRA state dict
lightning_lora_state_dict = load_state_dict(lightning_lora_path)
print(f"Lightning LoRA contains {len(lightning_lora_state_dict)} keys")

# Lightning LoRA uses the Lightning format
lightning_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
num_updated_lightning = lightning_loader.load(pipe.dit, lightning_lora_state_dict, alpha=1.0)
print(f"Lightning LoRA fused: {num_updated_lightning} tensors updated")

# Step 2: Fuse custom Ghost LoRA on top of the Lightning-fused model
# lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
# lora_path = "./models/train/Qwen-Image-Edit-Lightning-Ghost_lora/epoch-4.safetensors"
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora_1002/epoch-4.safetensors"
print(f"\nStep 2: Fusing custom Ghost LoRA from: {lora_path}")

# Load Ghost LoRA state dict
lora_state_dict = load_state_dict(lora_path)
print(f"Ghost LoRA contains {len(lora_state_dict)} keys")

# Detect LoRA format for Ghost LoRA
is_lightning_lora = any("lora_down" in k or "lora_up" in k for k in lora_state_dict.keys())
if is_lightning_lora:
    print("Detected Lightning LoRA format")
    lora_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
else:
    print("Detected standard LoRA format")
    lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")

# Fuse Ghost LoRA weights into the Lightning-fused model
num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
print(f"Ghost LoRA fused: {num_updated} tensors updated")
print("\nNote: Both Lightning and Ghost LoRA weights are now permanently merged into the model")

# NOW enable FP8 quantization AFTER LoRA is loaded for better performance
print("Enabling FP8 quantization...")
pipe.enable_vram_management(
    enable_dit_fp8_computation=True,  # Enable FP8 computation for DiT
    vram_limit=None,  # No limit for best performance
    auto_offload=False  # Disable auto offload for speed
)

"""Batch edit a folder of images using fused LoRA weights with FP8 quantization for optimal performance."""

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_ghost_fused_fp8_1002")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."
height = 1152
width = 768
num_inference_steps = 8  # Lightning LoRA is designed for 4-8 steps, now fused for optimal speed
seed = 1
edit_image_auto_resize = True

# Optional: Enable FP8 attention for additional speedup
# This requires GPUs that support FP8 operations (e.g., H200)
enable_fp8_attention = False  # Set to True if your GPU supports FP8

# Supported image extensions
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\nRunning inference with FP8 quantization")
print(f"Output directory: {output_dir}")
print(f"Inference steps: {num_inference_steps}")
print(f"FP8 attention: {enable_fp8_attention}")

for img_path in sorted(input_dir.rglob("*")):
    if img_path.suffix.lower() not in extensions:
        continue
    try:
        input_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue

    image = pipe(
        edit_prompt,
        edit_image=input_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=edit_image_auto_resize,
        enable_fp8_attention=enable_fp8_attention,
    )
    out_path = output_dir / f"{img_path.stem}_lightning_ghost_fp8.jpg"
    image.save(str(out_path))
    print(f"Saved: {out_path}")

print("\nDone! Lightning + Ghost LoRA fusion with FP8 quantization provides maximum speed:")
print(f"- Base model + Lightning LoRA + Ghost LoRA weights permanently fused")
print(f"- FP8 quantization reduces memory usage") 
print(f"- Only {num_inference_steps} steps needed for high quality results")
