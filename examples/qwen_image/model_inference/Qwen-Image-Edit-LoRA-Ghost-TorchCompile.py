"""
Use torch.compile() for actual speedup instead of FP8 quantization.
This is often more effective on modern GPUs.
"""

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
import torch
from PIL import Image
from pathlib import Path
import warnings

# Suppress torch.compile warnings if any
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

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

# FUSE LoRA for no runtime overhead
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"\nFusing LoRA from: {lora_path}")
lora_state_dict = load_state_dict(lora_path)
lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
print(f"LoRA fused: {num_updated} tensors updated")

# COMPILE the model for speed (instead of FP8 quantization)
print("\nCompiling model with torch.compile()...")
print("First inference will be slow (compilation), then fast!")

# Different compile modes to try:
# - "reduce-overhead": Best for small batch sizes
# - "max-autotune": Most aggressive optimization (slower compile, faster runtime)
# - "default": Balanced
compile_mode = "reduce-overhead"  # Good for single image inference

try:
    pipe.dit = torch.compile(pipe.dit, mode=compile_mode, fullgraph=False)
    print(f"✅ Model compiled with mode='{compile_mode}'")
except Exception as e:
    print(f"⚠️ Compilation failed: {e}")
    print("Continuing without compilation...")

# Optional: Also compile VAE for additional speedup
try:
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode=compile_mode)
    print("✅ VAE decoder also compiled")
except:
    pass

"""Batch inference with torch.compile() optimization."""

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_compiled")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."
height = 1152
width = 768
num_inference_steps = 40
seed = 1
edit_image_auto_resize = True

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\nRunning inference with torch.compile() optimization")
print(f"Mode: {compile_mode}")
print(f"Output directory: {output_dir}")
print("\nNOTE: First image will be SLOW (compilation), then FAST!")

import time

for i, img_path in enumerate(sorted(input_dir.rglob("*"))):
    if img_path.suffix.lower() not in extensions:
        continue
    
    try:
        input_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue
    
    start_time = time.time()
    
    image = pipe(
        edit_prompt,
        edit_image=input_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=edit_image_auto_resize,
    )
    
    elapsed = time.time() - start_time
    
    out_path = output_dir / f"{img_path.stem}_compiled.jpg"
    image.save(str(out_path))
    
    if i == 0:
        print(f"First image (with compilation): {elapsed:.1f}s")
        print("Subsequent images will be much faster!")
    else:
        print(f"Saved: {out_path} ({elapsed:.1f}s)")

print("\n✅ Done! torch.compile() should provide better speedup than FP8 on most GPUs.")
