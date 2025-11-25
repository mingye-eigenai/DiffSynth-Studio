"""
Diagnose why FP8 isn't providing speedup on H100
"""

import torch
import time
import numpy as np
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
from PIL import Image

def benchmark_single_inference(pipe, prompt, image, steps=10, warmup=2):
    """Benchmark a single inference"""
    # Warmup
    for _ in range(warmup):
        _ = pipe(prompt, edit_image=image, num_inference_steps=4, height=512, width=512)
    
    # Measure
    torch.cuda.synchronize()
    start = time.time()
    
    result = pipe(
        prompt,
        edit_image=image,
        num_inference_steps=steps,
        height=768,
        width=512,
        seed=42,
        enable_fp8_attention=True  # Make sure this is enabled!
    )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed, result

def test_configurations():
    """Test different FP8 configurations"""
    
    # Test image and prompt
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    prompt = "A cute ghost character"
    
    results = {}
    
    print("Testing different configurations on H100...")
    print("="*60)
    
    # Test 1: Baseline BF16
    print("\n1. Baseline (BF16, no quantization):")
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
    
    # Load LoRA
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    
    time_bf16, _ = benchmark_single_inference(pipe, prompt, test_image)
    results["BF16"] = time_bf16
    print(f"Time: {time_bf16:.2f}s")
    
    del pipe
    torch.cuda.empty_cache()
    
    # Test 2: FP8 with current implementation
    print("\n2. FP8 (current implementation):")
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
    
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
    
    time_fp8_current, _ = benchmark_single_inference(pipe, prompt, test_image)
    results["FP8_current"] = time_fp8_current
    print(f"Time: {time_fp8_current:.2f}s")
    
    del pipe
    torch.cuda.empty_cache()
    
    # Test 3: FP8 with offload dtype (might be the issue)
    print("\n3. FP8 with explicit offload dtype:")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                offload_dtype=torch.float8_e4m3fn,  # Explicit FP8
                offload_device="cuda"  # Keep on GPU
            ),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
    
    time_fp8_explicit, _ = benchmark_single_inference(pipe, prompt, test_image)
    results["FP8_explicit"] = time_fp8_explicit
    print(f"Time: {time_fp8_explicit:.2f}s")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY:")
    print("="*60)
    baseline = results["BF16"]
    for name, time_val in results.items():
        speedup = baseline / time_val
        print(f"{name:<20}: {time_val:>6.2f}s (speedup: {speedup:.2f}x)")
    
    print("\nDIAGNOSIS:")
    if results["FP8_current"] >= baseline * 0.95:  # Less than 5% improvement
        print("❌ FP8 is NOT providing speedup. Possible reasons:")
        print("   1. FP8 tensor cores not being utilized")
        print("   2. Implementation using FP8 storage but BF16 compute")
        print("   3. Missing Flash Attention 3 for FP8 attention")
        print("   4. Overhead from mixed precision")
        
        print("\nRECOMMENDATIONS:")
        print("1. Check if Flash Attention 3 is installed:")
        print("   pip install flash-attn --no-build-isolation")
        print("2. Use torch.compile() for better optimization:")
        print("   pipe.dit = torch.compile(pipe.dit)")
        print("3. For now, stick with BF16 + LoRA fusion for best speed")
    else:
        print("✅ FP8 is providing speedup!")
    
    # Check for Flash Attention
    print("\nChecking Flash Attention availability...")
    try:
        import flash_attn
        print(f"✅ Flash Attention version: {flash_attn.__version__}")
    except ImportError:
        print("❌ Flash Attention not installed")
        print("   This is likely why FP8 attention isn't working!")

if __name__ == "__main__":
    test_configurations()
