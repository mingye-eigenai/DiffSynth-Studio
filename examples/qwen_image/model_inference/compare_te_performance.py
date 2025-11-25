"""
Compare Transformer Engine FP8 vs Standard Implementation Performance
"""

import torch
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Try importing Transformer Engine
try:
    import transformer_engine.pytorch as te
    TRANSFORMER_ENGINE_AVAILABLE = True
    print("✅ Transformer Engine available")
except:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("❌ Transformer Engine not installed")

def benchmark_implementation(implementation: str, num_steps: int = 10):
    """Benchmark a specific implementation"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {implementation}")
    print(f"{'='*60}")
    
    # Import pipeline
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
    from diffsynth import load_state_dict
    from diffsynth.lora import GeneralLoRALoader
    
    # Load model
    print("Loading model...")
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
    num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
    print(f"LoRA fused: {num_updated} tensors updated")
    
    # Apply optimizations based on implementation
    if implementation == "transformer_engine_fp8":
        if not TRANSFORMER_ENGINE_AVAILABLE:
            print("⚠️ Skipping - Transformer Engine not available")
            return None
            
        from diffsynth.models.qwen_image_transformer_engine import (
            replace_linear_with_te_fp8,
            enable_te_fp8_autocast,
            get_fp8_recipe
        )
        
        print("Applying Transformer Engine FP8...")
        pipe.dit = replace_linear_with_te_fp8(pipe.dit)
        fp8_recipe = get_fp8_recipe()
        use_te_fp8 = True
        
    elif implementation == "standard_fp8":
        print("Using standard FP8 quantization...")
        pipe.enable_vram_management(enable_dit_fp8_computation=True, auto_offload=False)
        use_te_fp8 = False
        fp8_recipe = None
        
    elif implementation == "bf16_baseline":
        print("Using BF16 baseline (no FP8)...")
        use_te_fp8 = False
        fp8_recipe = None
        
    elif implementation == "bf16_compiled":
        print("Using BF16 + torch.compile...")
        try:
            pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
            print("✅ Model compiled")
        except:
            print("⚠️ Compilation failed")
        use_te_fp8 = False
        fp8_recipe = None
    
    # Test settings
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    prompt = "Test prompt for benchmarking"
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        if use_te_fp8:
            with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _ = pipe(prompt, edit_image=test_image, num_inference_steps=2, 
                        height=512, width=512, edit_image_auto_resize=True)
        else:
            _ = pipe(prompt, edit_image=test_image, num_inference_steps=2,
                    height=512, width=512, edit_image_auto_resize=True)
    
    # Benchmark
    print(f"Benchmarking {num_steps} steps...")
    torch.cuda.synchronize()
    start = time.time()
    
    if use_te_fp8:
        with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            _ = pipe(prompt, edit_image=test_image, num_inference_steps=num_steps,
                    height=768, width=512, edit_image_auto_resize=True)
    else:
        _ = pipe(prompt, edit_image=test_image, num_inference_steps=num_steps,
                height=768, width=512, edit_image_auto_resize=True)
    
    torch.cuda.synchronize()
    total_time = time.time() - start
    
    # Memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    return {
        "total_time": total_time,
        "time_per_step": total_time / num_steps,
        "memory_allocated": allocated,
        "memory_reserved": reserved,
    }

def main():
    print("="*80)
    print("H100 Performance Comparison: Transformer Engine vs Standard Implementations")
    print("="*80)
    
    # Check environment
    print(f"\nEnvironment:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Implementations to test
    implementations = [
        ("bf16_baseline", "BF16 Baseline"),
        ("bf16_compiled", "BF16 + torch.compile"),
        ("standard_fp8", "Standard FP8 Quantization"),
    ]
    
    if TRANSFORMER_ENGINE_AVAILABLE:
        implementations.append(("transformer_engine_fp8", "Transformer Engine FP8"))
    
    # Run benchmarks
    results = {}
    for impl_key, impl_name in implementations:
        torch.cuda.empty_cache()
        result = benchmark_implementation(impl_key, num_steps=10)
        if result:
            results[impl_key] = result
            results[impl_key]["name"] = impl_name
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Implementation':<30} {'Time (s)':<12} {'Per Step':<12} {'Memory (GB)':<12}")
    print("-"*80)
    
    baseline_time = results.get("bf16_baseline", {}).get("total_time", 1.0)
    
    for impl_key in ["bf16_baseline", "bf16_compiled", "standard_fp8", "transformer_engine_fp8"]:
        if impl_key in results:
            r = results[impl_key]
            speedup = baseline_time / r["total_time"]
            print(f"{r['name']:<30} {r['total_time']:<12.2f} "
                  f"{r['time_per_step']:<12.3f} {r['memory_allocated']:<12.1f}")
            print(f"{'  → Speedup:':<30} {speedup:.2f}x")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if "transformer_engine_fp8" in results:
        te_speedup = baseline_time / results["transformer_engine_fp8"]["total_time"]
        if te_speedup > 1.1:
            print("✅ Transformer Engine FP8 provides speedup!")
            print(f"   {te_speedup:.2f}x faster than baseline")
        else:
            print("❌ Transformer Engine FP8 not providing expected speedup")
            print("   Possible reasons:")
            print("   - Model might be too small to benefit")
            print("   - Memory bandwidth limited")
            print("   - Need to tune FP8 recipe parameters")
    
    if "standard_fp8" in results:
        std_fp8_speedup = baseline_time / results["standard_fp8"]["total_time"]
        if std_fp8_speedup < 1.0:
            print("\n⚠️ Standard FP8 is slower than BF16")
            print("   This confirms our earlier findings about conversion overhead")
    
    print("\nRecommendations:")
    print("1. For maximum speed: BF16 + torch.compile + Flash Attention")
    print("2. For memory savings: Consider Transformer Engine FP8")
    print("3. Avoid standard FP8 quantization on current PyTorch")

if __name__ == "__main__":
    main()
