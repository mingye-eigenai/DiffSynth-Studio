"""
Optimization strategies specifically for H100 GPU
"""

import torch
import os
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
from PIL import Image
from pathlib import Path

# H100-specific optimizations
print("Applying H100-specific optimizations...")

# 1. Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Set CUDA environment variables for H100
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for async execution
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # H100 compute capability

# 3. Use larger CUDA graphs if available
if hasattr(torch.cuda, 'graphs'):
    torch.cuda.set_sync_debug_mode(0)

print("Loading model with optimizations...")
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

# Fuse LoRA
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"Fusing LoRA from: {lora_path}")
lora_state_dict = load_state_dict(lora_path)
lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)

# 4. Try different compile modes for H100
print("\nTrying aggressive torch.compile() optimization...")
try:
    # Use inductor backend with specific H100 optimizations
    pipe.dit = torch.compile(
        pipe.dit,
        mode="max-autotune",  # Most aggressive optimization
        fullgraph=True,  # Try to compile the entire graph
        backend="inductor",  # Explicitly use inductor
    )
    print("✅ Compiled with max-autotune mode")
except Exception as e:
    print(f"⚠️ Max-autotune failed: {e}")
    try:
        pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead")
        print("✅ Fallback to reduce-overhead mode")
    except:
        print("❌ Compilation failed completely")

# 5. Enable CUDA graphs for the model
print("\nTrying to enable CUDA graphs...")
try:
    # Warm up the model
    dummy_image = Image.new('RGB', (512, 512), color='white')
    _ = pipe("test", edit_image=dummy_image, num_inference_steps=2, height=512, width=512)
    
    # Try to capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # This would capture the graph, but might not work with dynamic shapes
        pass
    print("✅ CUDA graphs available (but may not be used due to dynamic shapes)")
except:
    print("❌ CUDA graphs not applicable to this model")

# 6. Check and suggest Flash Attention installation
try:
    import flash_attn
    print(f"\n✅ Flash Attention is installed: v{flash_attn.__version__}")
except ImportError:
    print("\n❌ CRITICAL: Flash Attention not installed!")
    print("This is likely why you're not seeing speedup!")
    print("\nTo install Flash Attention for H100:")
    print("pip install flash-attn --no-build-isolation")
    print("or")
    print("pip install flash-attn[hopper]")

# Run optimized inference
print("\n" + "="*60)
print("Running OPTIMIZED inference on H100")
print("="*60)

input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_h100_optimized")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."

# Test with different settings
print("\nTesting different configurations:")

# 1. Fewer steps (since model quality is good)
print("\n1. Testing with 20 steps instead of 40...")
img_path = list(input_dir.glob("*.jpg"))[0]
image = Image.open(img_path).convert("RGB")

import time
start = time.time()
result = pipe(
    edit_prompt,
    edit_image=image,
    num_inference_steps=20,  # Half the steps
    height=768,
    width=512,
    seed=42,
)
elapsed = time.time() - start
print(f"20 steps: {elapsed:.1f}s (vs ~42s for 40 steps)")

# 2. Lower resolution
print("\n2. Testing with lower resolution...")
start = time.time()
result = pipe(
    edit_prompt,
    edit_image=image,
    num_inference_steps=20,
    height=512,  # Lower resolution
    width=384,
    seed=42,
)
elapsed = time.time() - start
print(f"512x384 @ 20 steps: {elapsed:.1f}s")

print("\n" + "="*60)
print("RECOMMENDATIONS FOR H100:")
print("="*60)
print("1. INSTALL FLASH ATTENTION! (most important)")
print("   pip install flash-attn --no-build-isolation")
print("")
print("2. Use fewer diffusion steps (20-25 instead of 40)")
print("   The model quality is good enough")
print("")
print("3. Consider batch processing multiple images")
print("   H100 is optimized for larger batches")
print("")
print("4. Use torch.compile() with max-autotune mode")
print("   (after installing Flash Attention)")
print("")
print("5. Monitor with: nvidia-smi dmon -s pucvmet")
print("   Check if GPU utilization is actually high")
