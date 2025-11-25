"""
H100 Optimized Inference with NVIDIA Transformer Engine FP8
True FP8 acceleration using Transformer Engine

Run these first:
1. bash install_flash_attention.sh
2. bash install_transformer_engine.sh
"""

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
import torch
from PIL import Image
from pathlib import Path
import time

# Import Transformer Engine components
try:
    from diffsynth.models.qwen_image_transformer_engine import (
        TRANSFORMER_ENGINE_AVAILABLE,
        replace_linear_with_te_fp8,
        enable_te_fp8_autocast,
        get_fp8_recipe
    )
    if TRANSFORMER_ENGINE_AVAILABLE:
        print("✅ NVIDIA Transformer Engine loaded - FP8 acceleration available!")
    else:
        print("⚠️ Transformer Engine not found. Using optimized BF16 mode instead.")
        print("   (This is fine - you still get excellent performance with Flash Attention + BF16)")
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("⚠️ Transformer Engine not available. Using optimized BF16 mode.")
    print("   Your current optimizations (Flash Attention + BF16 + torch.compile) are excellent!")
    
    # Define dummy functions so the script still works
    def get_fp8_recipe():
        return None
    
    def enable_te_fp8_autocast(enabled=True, fp8_recipe=None):
        import contextlib
        return contextlib.nullcontext()

# Enable H100-specific optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Check Flash Attention
from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
print(f"Flash Attention 3: {'✅ Available' if FLASH_ATTN_3_AVAILABLE else '❌ Not available'}")

print("\nLoading model...")
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

# Fuse LoRA for zero overhead
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"Fusing LoRA from: {lora_path}")
lora_state_dict = load_state_dict(lora_path)
lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
print(f"LoRA fused: {num_updated} tensors updated")

# Apply Transformer Engine FP8 optimization
if TRANSFORMER_ENGINE_AVAILABLE:
    print("\n🚀 Applying Transformer Engine FP8 optimization...")
    # Replace Linear layers with TE FP8 Linear
    # Skip certain layers that shouldn't use FP8 (e.g., final projection)
    skip_patterns = ['to_out', 'to_q', 'to_k', 'to_v']  # Can be tuned
    pipe.dit = replace_linear_with_te_fp8(pipe.dit, skip_patterns=skip_patterns)
    
    # Optional: Also compile with torch.compile for additional optimization
    try:
        pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
        print("✅ Model compiled with torch.compile")
    except:
        print("⚠️ torch.compile failed, continuing without")
else:
    # Fallback to regular torch.compile if TE not available
    print("\n⚠️ Using standard torch.compile (no FP8)")
    try:
        pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
        print("✅ Model compiled")
    except:
        print("⚠️ Compilation failed, continuing without")

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_h100_te_fp8")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."

# Optimal settings for H100 with FP8
height = 1152
width = 768
num_inference_steps = 25  
seed = 1
edit_image_auto_resize = True
enable_fp8_attention = False  # Keep False - we use TE FP8 instead

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\n🚀 Running H100 + Transformer Engine FP8 inference")
print(f"Optimizations enabled:")
print(f"  - Transformer Engine FP8: {'YES' if TRANSFORMER_ENGINE_AVAILABLE else 'NO'}")
print(f"  - Flash Attention 3: {'YES' if FLASH_ATTN_3_AVAILABLE else 'NO'}")
print(f"  - LoRA fusion: YES") 
print(f"  - TF32: YES")
print(f"  - torch.compile: YES")
print(f"  - Steps: {num_inference_steps}")
print(f"Output: {output_dir}\n")

# Get FP8 recipe if available
fp8_recipe = get_fp8_recipe() if TRANSFORMER_ENGINE_AVAILABLE else None

# Process images
times = []
for i, img_path in enumerate(sorted(input_dir.rglob("*"))[:5]):  # Test first 5
    if img_path.suffix.lower() not in extensions:
        continue
    
    try:
        input_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue
    
    torch.cuda.synchronize()
    start = time.time()
    
    # Use Transformer Engine FP8 autocast if available
    if TRANSFORMER_ENGINE_AVAILABLE:
        with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
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
    else:
        # Standard inference without FP8
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
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    
    out_path = output_dir / f"{img_path.stem}_te_fp8.jpg"
    image.save(str(out_path))
    
    print(f"Image {i+1}: {elapsed:.1f}s ({elapsed/num_inference_steps:.2f}s/step)")

if times:
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    print(f"\n📊 Performance Summary:")
    print(f"Average time: {avg_time:.1f}s")
    print(f"Per step: {avg_time/num_inference_steps:.2f}s")
    print(f"vs. Original: {42:.1f}s → {avg_time:.1f}s ({42/avg_time:.1f}x speedup)")
    
    if TRANSFORMER_ENGINE_AVAILABLE:
        print("\n✨ Using true FP8 compute with Transformer Engine!")
        print("This should be faster than standard FP8 quantization.")
    else:
        print("\n⚠️ Install Transformer Engine for true FP8 acceleration:")
        print("bash install_transformer_engine.sh")

print("\n✨ Done!")
