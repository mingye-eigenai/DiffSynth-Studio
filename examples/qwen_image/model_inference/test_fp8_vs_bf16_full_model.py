"""
Test FP8 vs BF16 performance for full Qwen Image model inference
Compares memory usage, speed, and quality
"""

import torch
import torch.nn as nn
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora import GeneralLoRALoader
from PIL import Image
from pathlib import Path
import time
import numpy as np
from typing import Dict, List
import gc
import json

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, skipping plots")

# Try to import Transformer Engine
try:
    from diffsynth.models.qwen_image_transformer_engine import (
        replace_linear_with_te_fp8,
        enable_te_fp8_autocast,
        get_fp8_recipe
    )
    TRANSFORMER_ENGINE_AVAILABLE = True
    print("✅ Transformer Engine available")
except:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("⚠️ Transformer Engine not available")


def setup_model(mode: str) -> QwenImagePipeline:
    """Setup model in specified mode: bf16, fp8_quantized, or fp8_te"""
    
    print(f"\nSetting up model in {mode} mode...")
    
    # Load base model
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", 
                       origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", 
                       origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", 
                       origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", 
                                   origin_file_pattern="processor/"),
    )
    
    # Load and fuse LoRA
    lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
    if Path(lora_path).exists():
        print(f"Fusing LoRA from: {lora_path}")
        lora_state_dict = load_state_dict(lora_path)
        lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
        num_updated = lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
        print(f"LoRA fused: {num_updated} tensors updated")
    
    # Apply mode-specific optimizations
    if mode == "fp8_quantized":
        print("Enabling standard FP8 quantization...")
        pipe.enable_vram_management(
            enable_dit_fp8_computation=True,
            auto_offload=False
        )
    elif mode == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
        print("Applying Transformer Engine FP8...")
        skip_patterns = ["proj_out", "final"]
        pipe.dit = replace_linear_with_te_fp8(pipe.dit, skip_patterns=skip_patterns)
    
    # Optional: compile model
    try:
        pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
        print("✅ Model compiled")
    except:
        print("⚠️ Compilation failed, continuing without")
    
    return pipe


def benchmark_inference(
    pipe: QwenImagePipeline,
    input_image: Image.Image,
    mode: str,
    num_steps: int = 25,
    num_runs: int = 5
) -> Dict:
    """Benchmark model inference"""
    
    edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style"
    height, width = 1152, 768
    
    # Get FP8 recipe if using TE
    fp8_recipe = get_fp8_recipe() if mode == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE else None
    
    # Warmup
    print(f"Warming up {mode}...")
    for _ in range(2):
        if mode == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
            with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _ = pipe(
                    edit_prompt,
                    edit_image=input_image,
                    seed=42,
                    num_inference_steps=5,  # Fewer steps for warmup
                    height=height,
                    width=width,
                )
        else:
            _ = pipe(
                edit_prompt,
                edit_image=input_image,
                seed=42,
                num_inference_steps=5,
                height=height,
                width=width,
                enable_fp8_attention=(mode == "fp8_quantized"),
            )
    
    # Benchmark
    times = []
    torch.cuda.reset_peak_memory_stats()
    
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        if mode == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
            with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                output_image = pipe(
                    edit_prompt,
                    edit_image=input_image,
                    seed=42,
                    num_inference_steps=num_steps,
                    height=height,
                    width=width,
                )
        else:
            output_image = pipe(
                edit_prompt,
                edit_image=input_image,
                seed=42,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                enable_fp8_attention=(mode == "fp8_quantized"),
            )
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  Run {i+1}: {elapsed:.2f}s ({elapsed/num_steps:.3f}s/step)")
    
    # Get memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Save last output for quality comparison
    output_path = Path(f"./test_outputs/{mode}_output.jpg")
    output_path.parent.mkdir(exist_ok=True)
    output_image.save(output_path)
    
    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "per_step_time": np.mean(times) / num_steps,
        "peak_memory_mb": peak_memory_mb,
        "output_path": str(output_path),
    }


def profile_memory_timeline(pipe: QwenImagePipeline, input_image: Image.Image, mode: str):
    """Profile memory usage over time during inference"""
    
    print(f"\nProfiling memory for {mode}...")
    
    edit_prompt = "Chibi-style 3D cartoon ghost"
    height, width = 1152, 768
    num_steps = 10  # Fewer steps for profiling
    
    memory_samples = []
    sample_interval = 0.1  # seconds
    
    # Start memory monitoring in background
    import threading
    stop_monitoring = threading.Event()
    
    def monitor_memory():
        while not stop_monitoring.is_set():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            memory_samples.append(memory_mb)
            time.sleep(sample_interval)
    
    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()
    
    # Run inference
    fp8_recipe = get_fp8_recipe() if mode == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE else None
    
    if mode == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
        with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            _ = pipe(edit_prompt, edit_image=input_image, seed=42,
                    num_inference_steps=num_steps, height=height, width=width)
    else:
        _ = pipe(edit_prompt, edit_image=input_image, seed=42,
                num_inference_steps=num_steps, height=height, width=width,
                enable_fp8_attention=(mode == "fp8_quantized"))
    
    # Stop monitoring
    stop_monitoring.set()
    monitor_thread.join()
    
    return memory_samples


def run_comprehensive_test():
    """Run comprehensive FP8 vs BF16 tests"""
    
    # Enable H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Test configurations
    modes = ["bf16", "fp8_quantized"]
    if TRANSFORMER_ENGINE_AVAILABLE:
        modes.append("fp8_te")
    
    # Load test image
    test_image_path = Path("/home/kuan/workspace/repos/avatar/fg_special/asian/001.jpg")
    if test_image_path.exists():
        test_image = Image.open(test_image_path).convert("RGB")
    else:
        # Create dummy test image
        test_image = Image.new("RGB", (512, 512), color="white")
    
    results = {}
    memory_profiles = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing {mode.upper()}")
        print(f"{'='*60}")
        
        # Setup model
        pipe = setup_model(mode)
        
        # Benchmark
        result = benchmark_inference(pipe, test_image, mode, num_steps=25, num_runs=3)
        results[mode] = result
        
        # Memory profile
        memory_profile = profile_memory_timeline(pipe, test_image, mode)
        memory_profiles[mode] = memory_profile
        
        # Cleanup
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
    
    # Generate report
    generate_report(results, memory_profiles)
    
    return results, memory_profiles


def generate_report(results: Dict, memory_profiles: Dict):
    """Generate comprehensive comparison report"""
    
    output_dir = Path("./fp8_vs_bf16_results")
    output_dir.mkdir(exist_ok=True)
    
    modes = list(results.keys())
    
    if MATPLOTLIB_AVAILABLE:
        # Create performance comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Speed comparison
        mean_times = [results[mode]["mean_time"] for mode in modes]
        ax1.bar(modes, mean_times, color=['blue', 'orange', 'green'][:len(modes)])
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Inference Speed Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add speedup labels
        bf16_time = results["bf16"]["mean_time"]
        for i, (mode, time) in enumerate(zip(modes, mean_times)):
            speedup = bf16_time / time
            ax1.text(i, time + 0.5, f"{speedup:.2f}x", ha='center')
        
        # Memory comparison
        peak_memories = [results[mode]["peak_memory_mb"] for mode in modes]
        ax2.bar(modes, peak_memories, color=['blue', 'orange', 'green'][:len(modes)])
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Memory reduction labels
        bf16_memory = results["bf16"]["peak_memory_mb"]
        for i, (mode, memory) in enumerate(zip(modes, peak_memories)):
            reduction = (1 - memory / bf16_memory) * 100
            ax2.text(i, memory + 50, f"-{reduction:.0f}%", ha='center')
        
        # Memory timeline
        for mode, profile in memory_profiles.items():
            time_points = np.arange(len(profile)) * 0.1  # sample_interval
            ax3.plot(time_points, profile, label=mode, linewidth=2)
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fp8_vs_bf16_full_comparison.png', dpi=150)
        plt.close()
    else:
        print("\nNote: matplotlib not available, skipping plots")
        print("Install with: pip install matplotlib")
    
    # Generate text report
    report_lines = [
        "FP8 vs BF16 Performance Report",
        "=" * 50,
        "",
        "Configuration:",
        f"- Image size: 1152x768",
        f"- Steps: 25",
        f"- Model: Qwen Image Edit with LoRA",
        "",
        "Results Summary:",
        ""
    ]
    
    # Performance table
    report_lines.append("Performance Comparison:")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Mode':<15} {'Time (s)':<10} {'Speedup':<10} {'Memory (MB)':<12} {'Reduction':<10}")
    report_lines.append("-" * 50)
    
    bf16_result = results["bf16"]
    for mode, result in results.items():
        speedup = bf16_result["mean_time"] / result["mean_time"]
        mem_reduction = (1 - result["peak_memory_mb"] / bf16_result["peak_memory_mb"]) * 100
        
        report_lines.append(
            f"{mode:<15} {result['mean_time']:<10.2f} {speedup:<10.2f}x "
            f"{result['peak_memory_mb']:<12.0f} {mem_reduction:<10.1f}%"
        )
    
    report_lines.extend([
        "",
        "Key Findings:",
        ""
    ])
    
    # Analysis
    if "fp8_quantized" in results:
        fp8q_speedup = bf16_result["mean_time"] / results["fp8_quantized"]["mean_time"]
        if fp8q_speedup < 1.0:
            report_lines.append(f"⚠️ Standard FP8 quantization is {1/fp8q_speedup:.1f}x SLOWER than BF16")
            report_lines.append("   This is expected - quantization overhead without hardware support")
        else:
            report_lines.append(f"✅ Standard FP8 quantization provides {fp8q_speedup:.1f}x speedup")
    
    if "fp8_te" in results:
        te_speedup = bf16_result["mean_time"] / results["fp8_te"]["mean_time"]
        report_lines.append(f"✅ Transformer Engine FP8 provides {te_speedup:.1f}x speedup")
        report_lines.append("   True FP8 compute on H100 tensor cores")
    
    report_lines.extend([
        "",
        "Recommendations:",
        "1. For A100/V100: Use BF16 (FP8 will be slower)",
        "2. For H100/H200: Use Transformer Engine FP8 for speedup",
        "3. Use standard FP8 only if memory is critical",
        "",
        f"Report generated: {output_dir}",
    ])
    
    # Save report
    with open(output_dir / "report.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print report
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    run_comprehensive_test()
