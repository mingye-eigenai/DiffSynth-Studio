from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
import torch
from PIL import Image
from pathlib import Path

# Initialize pipeline WITHOUT quantization first for faster LoRA fusion
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

# FUSE LoRA into base weights to eliminate runtime overhead
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"\nFusing LoRA from: {lora_path}")
lora_state_dict = load_state_dict(lora_path)

# Use GeneralLoRALoader to permanently fuse LoRA into base weights
lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
print(f"LoRA fused into base weights: {num_updated} tensors updated")
print("Note: LoRA is now permanently merged - no runtime overhead!")

# NOW enable aggressive VRAM management AFTER LoRA is fused
print("\nEnabling LOW VRAM mode with FP8 quantization and offloading...")
pipe.enable_vram_management(
    vram_limit=6.0,  # Even more aggressive limit for low-end GPUs
    vram_buffer=0.5,  # Smaller buffer for maximum savings
    auto_offload=True,  # Enable automatic offloading
    enable_dit_fp8_computation=True  # Enable FP8 computation
)

"""Optimized Low VRAM batch inference with fused LoRA, FP8 quantization and aggressive offloading."""

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_lowvram_optimized")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."

# Reduced resolution for lower VRAM usage
height = 768  # Reduced from 1152
width = 512   # Reduced from 768
num_inference_steps = 40
seed = 1
edit_image_auto_resize = True
enable_fp8_attention = True  # Enable FP8 attention for additional memory savings

# Process images with memory-efficient approach
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\nRunning OPTIMIZED LOW VRAM inference:")
print(f"- LoRA fused into weights (no runtime overhead)")
print(f"- FP8 quantization + aggressive offloading")
print(f"- VRAM limit: 6GB (works on RTX 3060 and up)")
print(f"- Resolution: {width}x{height}")
print(f"Output directory: {output_dir}")

# Optional: Force garbage collection before starting
torch.cuda.empty_cache()
import gc
gc.collect()

for img_path in sorted(input_dir.rglob("*")):
    if img_path.suffix.lower() not in extensions:
        continue
    
    try:
        input_image = Image.open(img_path).convert("RGB")
        
        # Optional: Resize input image to reduce memory usage
        if input_image.width > 1024 or input_image.height > 1024:
            input_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            print(f"  Resized input image to fit memory constraints")
            
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue

    # Generate image with memory-efficient settings
    try:
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
        
        out_path = output_dir / f"{img_path.stem}_lora_lowvram_opt.jpg"
        image.save(str(out_path))
        print(f"Saved: {out_path}")
        
    except torch.cuda.OutOfMemoryError:
        print(f"OOM for {img_path.name} - clearing cache and retrying...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Retry with even lower resolution
        try:
            image = pipe(
                edit_prompt,
                edit_image=input_image,
                seed=seed,
                num_inference_steps=num_inference_steps,
                height=512,  # Further reduced
                width=384,   # Further reduced
                edit_image_auto_resize=edit_image_auto_resize,
                enable_fp8_attention=enable_fp8_attention,
            )
            
            out_path = output_dir / f"{img_path.stem}_lora_lowvram_opt_small.jpg"
            image.save(str(out_path))
            print(f"Saved (reduced res): {out_path}")
        except Exception as e:
            print(f"Failed even with reduced resolution: {e}")
            continue
    
    # Clear CUDA cache after each image to prevent memory buildup
    torch.cuda.empty_cache()

print("\nDone! Optimized low VRAM mode should work on 6GB+ GPUs with better speed.")
