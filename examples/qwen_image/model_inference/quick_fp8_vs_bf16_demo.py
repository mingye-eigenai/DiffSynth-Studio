"""
Quick demo showing FP8 vs BF16 differences
Run this to understand the key concepts
"""

import torch
import torch.nn as nn
import numpy as np

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, skipping visualizations")

def visualize_fp8_vs_bf16():
    """Visualize the difference between FP8 and BF16 representations"""
    
    print("\n🔍 FP8 vs BF16 Comparison Demo")
    print("=" * 50)
    
    # 1. Precision comparison
    print("\n1. PRECISION COMPARISON")
    print("-" * 30)
    
    # BF16: 1 sign, 8 exponent, 7 mantissa = 16 bits
    # FP8 E4M3: 1 sign, 4 exponent, 3 mantissa = 8 bits
    # FP8 E5M2: 1 sign, 5 exponent, 2 mantissa = 8 bits
    
    print("BF16:     [S][EEEEEEEE][MMMMMMM]     = 16 bits")
    print("FP8 E4M3: [S][EEEE][MMM]              = 8 bits")
    print("FP8 E5M2: [S][EEEEE][MM]              = 8 bits")
    print("\nS=Sign, E=Exponent, M=Mantissa")
    
    # 2. Range comparison
    print("\n2. NUMERICAL RANGE")
    print("-" * 30)
    
    bf16_max = torch.finfo(torch.bfloat16).max
    bf16_min = torch.finfo(torch.bfloat16).min
    bf16_eps = torch.finfo(torch.bfloat16).eps
    
    # FP8 ranges (approximate)
    fp8_e4m3_max = 448.0  # 2^8 * (1 + 7/8)
    fp8_e5m2_max = 57344.0  # 2^15 * (1 + 3/4)
    
    print(f"BF16:     Range [{bf16_min:.2e}, {bf16_max:.2e}], Epsilon: {bf16_eps:.2e}")
    print(f"FP8 E4M3: Range [-{fp8_e4m3_max}, {fp8_e4m3_max}], Better precision")
    print(f"FP8 E5M2: Range [-{fp8_e5m2_max}, {fp8_e5m2_max}], Better range")
    
    # 3. Memory usage example
    print("\n3. MEMORY USAGE FOR QWEN IMAGE MODEL")
    print("-" * 30)
    
    # Calculate for a typical transformer block
    dim = 3072
    num_heads = 24
    num_layers = 60
    
    # Linear layers in one transformer block
    linear_params = {
        "img_mod": dim * (6 * dim),
        "txt_mod": dim * (6 * dim),
        "mlp_hidden": dim * (4 * dim),
        "mlp_out": (4 * dim) * dim,
        "attention": 4 * dim * dim,  # Q, K, V, O projections
    }
    
    total_params_per_block = sum(linear_params.values())
    total_params = total_params_per_block * num_layers
    
    bf16_memory_gb = (total_params * 2) / (1024**3)  # 2 bytes per BF16
    fp8_memory_gb = (total_params * 1) / (1024**3)   # 1 byte per FP8
    
    print(f"Model Parameters: {total_params/1e9:.2f}B")
    print(f"BF16 Memory: {bf16_memory_gb:.2f} GB")
    print(f"FP8 Memory:  {fp8_memory_gb:.2f} GB")
    print(f"Savings:     {bf16_memory_gb - fp8_memory_gb:.2f} GB ({(1 - fp8_memory_gb/bf16_memory_gb)*100:.0f}%)")
    
    # 4. Quantization error visualization
    print("\n4. QUANTIZATION ERROR VISUALIZATION")
    print("-" * 30)
    
    # Create sample weights
    torch.manual_seed(42)
    weights = torch.randn(1000, dtype=torch.bfloat16) * 0.02
    
    # Simulate FP8 quantization
    def quantize_to_fp8_e4m3(tensor):
        # Simple simulation - in reality, this is more complex
        scale = tensor.abs().max() / 224.0  # FP8 E4M3 max positive value
        quantized = torch.round(tensor / scale).clamp(-224, 224)
        dequantized = quantized * scale
        return dequantized, scale
    
    weights_fp8, scale = quantize_to_fp8_e4m3(weights)
    errors = (weights - weights_fp8).abs()
    
    # Statistics
    print(f"Quantization Error Statistics:")
    print(f"  Mean Error: {errors.mean():.2e}")
    print(f"  Max Error:  {errors.max():.2e}")
    print(f"  Std Error:  {errors.std():.2e}")
    
    if MATPLOTLIB_AVAILABLE:
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution comparison
        ax1.hist(weights.float().cpu().numpy(), bins=50, alpha=0.5, label='BF16', density=True)
        ax1.hist(weights_fp8.float().cpu().numpy(), bins=50, alpha=0.5, label='FP8 (simulated)', density=True)
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Weight Distribution: BF16 vs FP8')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        ax2.hist(errors.float().cpu().numpy(), bins=50, color='red', alpha=0.7)
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Quantization Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        ax2.text(0.95, 0.95, f'Mean Error: {errors.mean():.2e}\nMax Error: {errors.max():.2e}',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('fp8_vs_bf16_visualization.png', dpi=150)
        print(f"Saved visualization to fp8_vs_bf16_visualization.png")
    else:
        print("(Install matplotlib to see visualization plots)")
    
    # 5. Performance implications
    print("\n5. PERFORMANCE IMPLICATIONS")
    print("-" * 30)
    
    print("\n✅ When FP8 is FASTER:")
    print("  - H100/H200 with Transformer Engine (native FP8 compute)")
    print("  - Large batch sizes (better tensor core utilization)")
    print("  - Memory-bandwidth limited operations")
    
    print("\n❌ When FP8 is SLOWER:")
    print("  - GPUs without FP8 tensor cores (A100, V100, RTX)")
    print("  - Small batch sizes or operations")
    print("  - When using quantization without hardware support")
    
    print("\n💡 Key Insight:")
    print("FP8 saves memory ALWAYS, but is faster ONLY with proper hardware support!")


def benchmark_simple_operation():
    """Simple benchmark to show conversion overhead"""
    
    print("\n\n6. SIMPLE BENCHMARK: Matrix Multiplication")
    print("=" * 50)
    
    sizes = [(1024, 1024), (4096, 4096)]
    
    for M, N in sizes:
        print(f"\nMatrix size: {M}x{N}")
        print("-" * 30)
        
        # Create matrices
        A_bf16 = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
        B_bf16 = torch.randn(N, M, dtype=torch.bfloat16, device='cuda')
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A_bf16, B_bf16)
        
        # Benchmark BF16
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            C_bf16 = torch.matmul(A_bf16, B_bf16)
        end.record()
        
        torch.cuda.synchronize()
        bf16_time = start.elapsed_time(end) / 100
        
        # Simulate FP8 with conversion overhead
        # This represents what happens without hardware FP8 support
        start.record()
        for _ in range(100):
            # Simulate quantization overhead
            A_fp8 = A_bf16.to(torch.int8).to(torch.bfloat16) * 0.01
            B_fp8 = B_bf16.to(torch.int8).to(torch.bfloat16) * 0.01
            C_fp8 = torch.matmul(A_fp8, B_fp8)
        end.record()
        
        torch.cuda.synchronize()
        fp8_simulated_time = start.elapsed_time(end) / 100
        
        print(f"BF16 time:          {bf16_time:.3f} ms")
        print(f"FP8 (simulated):    {fp8_simulated_time:.3f} ms")
        print(f"Overhead:           {fp8_simulated_time/bf16_time:.2f}x slower")
        
        # Memory usage
        bf16_memory = (2 * M * N * 2) / (1024**2)  # 2 matrices, 2 bytes each
        fp8_memory = (2 * M * N * 1) / (1024**2)   # 2 matrices, 1 byte each
        
        print(f"BF16 memory:        {bf16_memory:.1f} MB")
        print(f"FP8 memory:         {fp8_memory:.1f} MB")
        print(f"Memory saved:       {bf16_memory - fp8_memory:.1f} MB ({(1-fp8_memory/bf16_memory)*100:.0f}%)")


if __name__ == "__main__":
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        
        if "H100" in gpu_name or "H200" in gpu_name:
            print("✅ Your GPU supports native FP8 compute!")
        else:
            print("⚠️  Your GPU does not have FP8 tensor cores")
    
    # Run demos
    visualize_fp8_vs_bf16()
    
    if torch.cuda.is_available():
        benchmark_simple_operation()
    
    print("\n\n🎯 SUMMARY")
    print("=" * 50)
    print("1. FP8 uses half the memory of BF16 (8 bits vs 16 bits)")
    print("2. FP8 is only faster with hardware support (H100/H200)")
    print("3. Without hardware support, FP8 is slower due to conversion")
    print("4. Accuracy loss is minimal for most deep learning tasks")
    print("5. Choose based on your hardware and priorities (speed vs memory)")
