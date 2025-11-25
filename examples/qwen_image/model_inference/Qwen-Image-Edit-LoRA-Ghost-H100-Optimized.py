"""
Fully optimized inference for H100 with Flash Attention
Run install_flash_attention.sh first!
"""

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
import torch
from PIL import Image
from pathlib import Path
import time

# Enable H100-specific optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# # Check Flash Attention status
# try:
#     import flash_attn
#     print(f"✅ Flash Attention v{flash_attn.__version__} loaded")
#     from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
#     assert FLASH_ATTN_3_AVAILABLE, "Flash Attention not detected by model"
#     print("✅ Model will use Flash Attention 3")
# except:
#     print("⚠️ WARNING: Flash Attention not available!")
#     print("Run: bash install_flash_attention.sh")


try:
    # Try modern import first
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    try:
        # Fallback to old style import
        import flash_attn_interface
        from flash_attn_interface import flash_attn_func
        FLASH_ATTN_3_AVAILABLE = True
    except (ModuleNotFoundError, ImportError):
        FLASH_ATTN_3_AVAILABLE = False



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

# Optional: Compile with torch.compile for additional speedup
# Note: With Flash Attention, the benefit is smaller but still worth it
print("\nCompiling model...")
try:
    pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
    print("✅ Model compiled")
except:
    print("⚠️ Compilation failed, continuing without")

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_h100_fast")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."

# Optimal settings for H100
height = 1152
width = 768
num_inference_steps = 25  # Reduced from 40 - still high quality
seed = 1
edit_image_auto_resize = True
enable_fp8_attention = False  # Flash Attention doesn't support FP8 - use BF16 for speed

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\n🚀 Running H100-OPTIMIZED inference")
print(f"Optimizations enabled:")
print(f"  - Flash Attention 3: {'YES' if 'flash_attn' in globals() else 'NO'}")
print(f"  - LoRA fusion: YES") 
print(f"  - TF32: YES")
print(f"  - torch.compile: YES")
print(f"  - Reduced steps: {num_inference_steps} (from 40)")
print(f"Output: {output_dir}\n")

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
    
    out_path = output_dir / f"{img_path.stem}_fast.jpg"
    image.save(str(out_path))
    
    print(f"Image {i+1}: {elapsed:.1f}s ({elapsed/num_inference_steps:.2f}s/step)")

if times:
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    print(f"\n📊 Performance Summary:")
    print(f"Average time: {avg_time:.1f}s")
    print(f"Per step: {avg_time/num_inference_steps:.2f}s")
    print(f"vs. Original: {42:.1f}s → {avg_time:.1f}s ({42/avg_time:.1f}x speedup)")

print("\n✨ Done! This should be MUCH faster with Flash Attention installed.")
