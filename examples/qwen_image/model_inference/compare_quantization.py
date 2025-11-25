"""
Script to compare memory usage and inference speed between different quantization modes.
"""

import torch
import time
import gc
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
from PIL import Image
import numpy as np

def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    return torch.cuda.memory_allocated() / 1024**3

def measure_inference(pipe, prompt, num_steps=8, num_runs=3):
    """Measure inference time and memory"""
    # Warmup
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    _ = pipe(prompt, edit_image=test_image, num_inference_steps=4, height=512, width=512)
    
    # Measure
    torch.cuda.synchronize()
    start_mem = get_gpu_memory()
    times = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        _ = pipe(
            prompt, 
            edit_image=test_image,
            num_inference_steps=num_steps,
            height=768,
            width=512,
            seed=42
        )
        
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    
    peak_mem = get_gpu_memory()
    avg_time = np.mean(times)
    
    return {
        "avg_time": avg_time,
        "memory_used": peak_mem - start_mem,
        "peak_memory": peak_mem
    }

def test_standard_mode():
    """Test standard BF16 mode"""
    print("\n=== Testing Standard BF16 Mode ===")
    
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
    
    print(f"Model loaded. Memory: {get_gpu_memory():.2f} GB")
    
    results = measure_inference(pipe, "A cute ghost character")
    print(f"Inference time: {results['avg_time']:.2f}s")
    print(f"Peak memory: {results['peak_memory']:.2f} GB")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def test_fp8_mode():
    """Test FP8 quantized mode"""
    print("\n=== Testing FP8 Quantized Mode ===")
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                offload_dtype=torch.float8_e4m3fn
            ),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
    
    print(f"Model loaded with FP8. Memory: {get_gpu_memory():.2f} GB")
    
    results = measure_inference(pipe, "A cute ghost character")
    print(f"Inference time: {results['avg_time']:.2f}s")
    print(f"Peak memory: {results['peak_memory']:.2f} GB")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def test_low_vram_mode():
    """Test low VRAM mode with aggressive offloading"""
    print("\n=== Testing Low VRAM Mode ===")
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                offload_dtype=torch.float8_e4m3fn
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image", 
                origin_file_pattern="text_encoder/model*.safetensors",
                offload_device="cpu",
                offload_dtype=torch.float8_e4m3fn
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image", 
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                offload_device="cpu",
                offload_dtype=torch.float8_e4m3fn
            ),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    pipe.enable_vram_management(
        vram_limit=8.0,
        auto_offload=True,
        enable_dit_fp8_computation=True
    )
    
    print(f"Model loaded with aggressive offloading. Memory: {get_gpu_memory():.2f} GB")
    
    results = measure_inference(pipe, "A cute ghost character")
    print(f"Inference time: {results['avg_time']:.2f}s")
    print(f"Peak memory: {results['peak_memory']:.2f} GB")
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

if __name__ == "__main__":
    print("Quantization Comparison for Qwen-Image-Edit")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    results = {}
    
    # Test each mode
    try:
        results["bf16"] = test_standard_mode()
    except torch.cuda.OutOfMemoryError:
        print("OOM in BF16 mode - skipping")
        results["bf16"] = {"avg_time": -1, "peak_memory": -1}
    
    try:
        results["fp8"] = test_fp8_mode()
    except torch.cuda.OutOfMemoryError:
        print("OOM in FP8 mode - skipping")
        results["fp8"] = {"avg_time": -1, "peak_memory": -1}
    
    try:
        results["low_vram"] = test_low_vram_mode()
    except torch.cuda.OutOfMemoryError:
        print("OOM in Low VRAM mode - skipping")
        results["low_vram"] = {"avg_time": -1, "peak_memory": -1}
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Mode':<15} {'Time (s)':<10} {'Memory (GB)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    baseline_time = results.get("bf16", {}).get("avg_time", 1)
    for mode, result in results.items():
        if result["avg_time"] > 0:
            speedup = baseline_time / result["avg_time"] if baseline_time > 0 else 0
            print(f"{mode:<15} {result['avg_time']:<10.2f} {result['peak_memory']:<12.2f} {speedup:<10.2f}x")
        else:
            print(f"{mode:<15} {'OOM':<10} {'OOM':<12} {'-':<10}")
