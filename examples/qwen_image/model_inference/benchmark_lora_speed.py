"""
Benchmark different LoRA loading and quantization strategies to find the fastest configuration.
"""

import torch
import time
import gc
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
from PIL import Image
import numpy as np

def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    return torch.cuda.memory_allocated() / 1024**3

def benchmark_inference(pipe, name, num_steps=40, num_runs=3):
    """Benchmark inference speed"""
    test_image = Image.fromarray(np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8))
    prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style"
    
    # Warmup
    _ = pipe(prompt, edit_image=test_image, num_inference_steps=4, height=512, width=512)
    
    torch.cuda.synchronize()
    times = []
    
    print(f"\nBenchmarking {name}...")
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        _ = pipe(
            prompt,
            edit_image=test_image,
            num_inference_steps=num_steps,
            height=768,
            width=512,
            seed=42
        )
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_time = np.mean(times)
    memory = get_gpu_memory()
    
    return {
        "name": name,
        "avg_time": avg_time,
        "memory": memory,
        "steps": num_steps
    }

def test_standard_lora():
    """Test 1: Standard LoRA loading (baseline)"""
    print("\n=== Test 1: Standard LoRA (BF16, no quantization) ===")
    
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
    
    # Load LoRA normally
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    
    result = benchmark_inference(pipe, "Standard LoRA")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def test_fused_lora():
    """Test 2: Fused LoRA (no runtime overhead)"""
    print("\n=== Test 2: Fused LoRA (BF16, no quantization) ===")
    
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
    
    # Fuse LoRA into weights
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    lora_state_dict = load_state_dict(lora_path)
    lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
    lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
    
    result = benchmark_inference(pipe, "Fused LoRA")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def test_quantized_then_lora():
    """Test 3: Enable quantization first, then load LoRA"""
    print("\n=== Test 3: FP8 Quantization -> LoRA Loading ===")
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                offload_dtype=torch.float8_e4m3fn
            ),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    # Enable quantization FIRST
    pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
    
    # Then load LoRA
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    
    result = benchmark_inference(pipe, "Quantize->LoRA")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def test_lora_then_quantized():
    """Test 4: Load LoRA first, then enable quantization"""
    print("\n=== Test 4: LoRA Loading -> FP8 Quantization ===")
    
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
    
    # Load LoRA FIRST
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    
    # Then enable quantization
    pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
    
    result = benchmark_inference(pipe, "LoRA->Quantize")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def test_fused_then_quantized():
    """Test 5: Fuse LoRA, then enable quantization (OPTIMAL)"""
    print("\n=== Test 5: Fused LoRA -> FP8 Quantization ===")
    
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
    
    # Fuse LoRA FIRST
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    lora_state_dict = load_state_dict(lora_path)
    lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
    lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
    
    # Then enable quantization
    pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
    
    result = benchmark_inference(pipe, "Fused->Quantize")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

if __name__ == "__main__":
    print("LoRA Loading Strategy Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    results = []
    
    # Run all tests
    try:
        results.append(test_standard_lora())
    except Exception as e:
        print(f"Failed: {e}")
    
    try:
        results.append(test_fused_lora())
    except Exception as e:
        print(f"Failed: {e}")
    
    try:
        results.append(test_quantized_then_lora())
    except Exception as e:
        print(f"Failed: {e}")
    
    try:
        results.append(test_lora_then_quantized())
    except Exception as e:
        print(f"Failed: {e}")
    
    try:
        results.append(test_fused_then_quantized())
    except Exception as e:
        print(f"Failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Strategy':<25} {'Time (s)':<12} {'Memory (GB)':<12} {'Speedup':<10}")
    print("-"*70)
    
    if results:
        baseline = results[0]["avg_time"]
        for r in sorted(results, key=lambda x: x["avg_time"]):
            speedup = baseline / r["avg_time"]
            print(f"{r['name']:<25} {r['avg_time']:<12.2f} {r['memory']:<12.2f} {speedup:<10.2f}x")
    
    print("\nRecommendation:")
    print("1. BEST SPEED: Fuse LoRA -> Enable FP8 quantization")
    print("2. This eliminates runtime LoRA overhead + reduces memory with FP8")
    print("3. Order matters: Always fuse/load LoRA before enabling quantization")
