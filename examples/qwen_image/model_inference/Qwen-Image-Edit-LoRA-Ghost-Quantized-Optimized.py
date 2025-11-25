from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
import torch
from PIL import Image
from pathlib import Path

# Initialize pipeline WITHOUT quantization first
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

# Load and FUSE LoRA into base weights to eliminate runtime overhead
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"\nFusing LoRA from: {lora_path}")
lora_state_dict = load_state_dict(lora_path)

# Use GeneralLoRALoader to permanently fuse LoRA into base weights
lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
print(f"LoRA fused into base weights: {num_updated} tensors updated")
print("Note: LoRA is now permanently merged - no runtime overhead!")

# Debug: Check model details
# print(f"Model: {pipe.dit}")
# print(f"Model parameters: {sum(p.numel() for p in pipe.dit.parameters()) / 1e9:.2f}B")

# NOW enable VRAM management with FP8 after LoRA is fused
print("\nEnabling FP8 quantization...")
pipe.enable_vram_management(
    enable_dit_fp8_computation=True,  # Enable FP8 computation for DiT
    vram_limit=None,  # No limit for maximum performance
    auto_offload=False  # Disable offload for speed
)

"""Optimized batch inference with fused LoRA and FP8 quantization."""

print(pipe.dit)
# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_fp8_optimized")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."
height = 1152
width = 768
num_inference_steps = 40  # Use your preferred steps
seed = 1
edit_image_auto_resize = True

# Optional: Enable FP8 attention for GPUs that support it
enable_fp8_attention = False  # Set to True for H200 or newer

# Supported image extensions
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\nRunning OPTIMIZED inference:")
print(f"- LoRA fused into weights (no runtime overhead)")
print(f"- FP8 quantization enabled")
print(f"- Auto offload disabled for speed")
print(f"Output directory: {output_dir}")

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
    out_path = output_dir / f"{img_path.stem}_lora_fp8_opt.jpg"
    image.save(str(out_path))
    print(f"Saved: {out_path}")

print("\nDone! This optimized version should be faster than the original.")