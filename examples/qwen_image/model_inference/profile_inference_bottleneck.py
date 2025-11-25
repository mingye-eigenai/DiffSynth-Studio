"""
Profile the inference pipeline to identify bottlenecks
"""

import torch
import time
import numpy as np
from contextlib import contextmanager
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
from PIL import Image

@contextmanager
def timer(name):
    """Simple timer context manager"""
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")

def profile_inference():
    """Profile different parts of the inference pipeline"""
    
    print("Loading model for profiling...")
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
    
    # Load and fuse LoRA
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    lora_state_dict = load_state_dict(lora_path)
    lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
    lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
    
    # Test data
    test_image = Image.fromarray(np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8))
    prompt = "Chibi-style 3D cartoon ghost"
    
    print("\n" + "="*60)
    print("PROFILING INFERENCE PIPELINE")
    print("="*60)
    
    # Full inference timing
    with timer("Full inference (10 steps)"):
        _ = pipe(prompt, edit_image=test_image, num_inference_steps=10, height=768, width=512)
    
    # Now profile individual components
    print("\nDetailed component profiling (1 step each):")
    
    # Test text encoding
    with timer("Text encoding"):
        pipe.encode_prompt(prompt, negative_prompt="")
    
    # Test VAE encoding
    with timer("VAE encode (image -> latent)"):
        image_tensor = torch.randn(1, 3, 768, 512).to(pipe.device, pipe.torch_dtype)
        with torch.no_grad():
            _ = pipe.vae.encode(image_tensor)
    
    # Test single DiT forward pass
    print("\nDiT forward pass breakdown:")
    # Create dummy inputs
    latents = torch.randn(1, 16, 96, 64).to(pipe.device, pipe.torch_dtype)
    timesteps = torch.tensor([500]).to(pipe.device)
    prompt_embeds = torch.randn(1, 300, 3584).to(pipe.device, pipe.torch_dtype)
    prompt_embeds_mask = torch.ones(1, 300).to(pipe.device, torch.bool)
    img_rope = torch.randn(1, 96, 64, 128).to(pipe.device, pipe.torch_dtype)
    txt_rope = torch.randn(1, 300, 128).to(pipe.device, pipe.torch_dtype)
    
    # Single step
    with timer("Single DiT forward pass"):
        with torch.no_grad():
            _ = pipe.dit(
                latents, 
                timesteps,
                prompt_embeds,
                prompt_embeds_mask,
                img_rope,
                txt_rope,
            )
    
    # Test VAE decode
    with timer("VAE decode (latent -> image)"):
        latents = torch.randn(1, 16, 96, 64).to(pipe.device, pipe.torch_dtype)
        with torch.no_grad():
            _ = pipe.vae.decode(latents)
    
    # Memory info
    print(f"\nMemory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Check if Flash Attention is being used
    print("\nChecking optimizations:")
    try:
        import flash_attn
        print(f"✅ Flash Attention installed: v{flash_attn.__version__}")
    except:
        print("❌ Flash Attention NOT installed - this is likely the bottleneck!")
    
    # Check model size
    total_params = sum(p.numel() for p in pipe.dit.parameters())
    print(f"\nDiT model size: {total_params / 1e9:.2f}B parameters")
    
    # Estimate FLOPS
    print("\nPerformance analysis:")
    dit_time_per_step = 1.0  # Approximate from your results
    tflops = (total_params * 2 * 768 * 512) / (dit_time_per_step * 1e12)
    print(f"Estimated utilization: {tflops:.1f} TFLOPS")
    print(f"H100 peak: ~1000 TFLOPS (BF16)")
    print(f"Efficiency: {tflops/1000*100:.1f}%")
    
    if tflops < 100:
        print("\n⚠️ LOW GPU UTILIZATION DETECTED!")
        print("Possible causes:")
        print("1. Memory bandwidth bottleneck")
        print("2. CPU-GPU communication overhead")
        print("3. Inefficient attention implementation")
        print("4. Small batch size (1)")

if __name__ == "__main__":
    profile_inference()
