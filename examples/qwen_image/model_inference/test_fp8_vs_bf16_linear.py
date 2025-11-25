"""
Test FP8 vs BF16 performance for Linear layers
This script isolates and benchmarks linear layer operations
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import gc

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, skipping plots")

# Try to import Transformer Engine for true FP8
try:
    import transformer_engine.pytorch as te
    from diffsynth.models.qwen_image_transformer_engine import (
        TransformerEngineFP8Linear,
        get_fp8_recipe,
        enable_te_fp8_autocast
    )
    TRANSFORMER_ENGINE_AVAILABLE = True
    print("✅ Transformer Engine available for true FP8")
except:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("⚠️ Transformer Engine not available - will test quantized FP8 only")


class FP8QuantizedLinear(nn.Module):
    """Simulated FP8 linear layer using quantization"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights in FP8 format (simulated with int8 + scale)
        self.register_buffer('weight_fp8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize with random weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize as BF16 then quantize
        weight = torch.randn(self.out_features, self.in_features, dtype=torch.bfloat16) * 0.02
        self._quantize_weight(weight)
    
    def _quantize_weight(self, weight):
        # Simple FP8 E4M3 quantization simulation
        abs_max = weight.abs().max()
        scale = abs_max / 127.0  # int8 range
        self.weight_scale.copy_(scale)
        self.weight_fp8.copy_((weight / scale).round().clamp(-128, 127).to(torch.int8))
    
    def forward(self, x):
        # Dequantize weight to BF16 for computation
        weight_bf16 = self.weight_fp8.to(torch.bfloat16) * self.weight_scale
        return torch.nn.functional.linear(x, weight_bf16, self.bias)


def create_linear_layer(layer_type: str, in_features: int, out_features: int, bias: bool = True):
    """Create linear layer of specified type"""
    if layer_type == "bf16":
        return nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16).cuda()
    elif layer_type == "fp8_quantized":
        return FP8QuantizedLinear(in_features, out_features, bias=bias).cuda()
    elif layer_type == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
        return TransformerEngineFP8Linear(in_features, out_features, bias=bias).cuda()
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def benchmark_linear_layer(
    layer_type: str,
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    test_backward: bool = True
) -> Dict[str, float]:
    """Benchmark a single linear layer configuration"""
    
    # Create layer
    layer = create_linear_layer(layer_type, in_features, out_features)
    
    # Create input
    input_tensor = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16, device="cuda")
    
    # Get FP8 recipe if using TE
    fp8_recipe = get_fp8_recipe() if layer_type == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE else None
    
    # Warmup
    for _ in range(num_warmup):
        if layer_type == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
            with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                output = layer(input_tensor)
        else:
            output = layer(input_tensor)
        
        if test_backward:
            loss = output.sum()
            loss.backward()
    
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    forward_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        if layer_type == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
            with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                output = layer(input_tensor)
        else:
            output = layer(input_tensor)
            
        torch.cuda.synchronize()
        forward_times.append(time.time() - start)
    
    # Benchmark backward pass if requested
    backward_times = []
    if test_backward:
        for _ in range(num_iterations):
            if layer_type == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
                with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    output = layer(input_tensor)
            else:
                output = layer(input_tensor)
            
            torch.cuda.synchronize()
            start = time.time()
            
            loss = output.sum()
            loss.backward()
            
            torch.cuda.synchronize()
            backward_times.append(time.time() - start)
    
    # Calculate memory usage
    torch.cuda.reset_peak_memory_stats()
    if layer_type == "fp8_te" and TRANSFORMER_ENGINE_AVAILABLE:
        with enable_te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            _ = layer(input_tensor)
    else:
        _ = layer(input_tensor)
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    results = {
        "forward_mean": np.mean(forward_times) * 1000,  # ms
        "forward_std": np.std(forward_times) * 1000,
        "memory_mb": memory_mb,
    }
    
    if test_backward:
        results["backward_mean"] = np.mean(backward_times) * 1000
        results["backward_std"] = np.std(backward_times) * 1000
    
    # Cleanup
    del layer, input_tensor, output
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def test_accuracy_difference(in_features: int, out_features: int, num_samples: int = 1000):
    """Test accuracy difference between FP8 and BF16"""
    
    # Create reference BF16 layer
    bf16_layer = nn.Linear(in_features, out_features, bias=True).to(torch.bfloat16).cuda()
    
    # Create FP8 layers with same weights
    fp8_quant_layer = FP8QuantizedLinear(in_features, out_features, bias=True).cuda()
    fp8_quant_layer._quantize_weight(bf16_layer.weight.data)
    fp8_quant_layer.bias.data.copy_(bf16_layer.bias.data)
    
    # Test on random inputs
    test_inputs = torch.randn(num_samples, in_features, dtype=torch.bfloat16, device="cuda")
    
    with torch.no_grad():
        bf16_outputs = bf16_layer(test_inputs)
        fp8_outputs = fp8_quant_layer(test_inputs)
    
    # Calculate metrics
    mse = torch.nn.functional.mse_loss(fp8_outputs, bf16_outputs).item()
    relative_error = (fp8_outputs - bf16_outputs).abs() / (bf16_outputs.abs() + 1e-8)
    max_relative_error = relative_error.max().item()
    mean_relative_error = relative_error.mean().item()
    
    return {
        "mse": mse,
        "max_relative_error": max_relative_error,
        "mean_relative_error": mean_relative_error,
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks for Qwen Image model dimensions"""
    
    print("\n🔬 Testing FP8 vs BF16 for Linear Layers")
    print("=" * 60)
    
    # Qwen Image model dimensions
    configs = [
        # (name, batch_size, seq_len, in_features, out_features)
        ("img_in", 1, 576, 64, 3072),        # Image input projection
        ("txt_in", 1, 300, 3584, 3072),      # Text input projection
        ("img_mod", 1, 576, 3072, 18432),    # Image modulation (6 * 3072)
        ("txt_mod", 1, 300, 3072, 18432),    # Text modulation
        ("mlp_hidden", 1, 576, 3072, 12288), # MLP hidden (4 * 3072)
        ("mlp_out", 1, 576, 12288, 3072),    # MLP output
        ("proj_out", 1, 576, 3072, 64),      # Final projection
    ]
    
    layer_types = ["bf16", "fp8_quantized"]
    if TRANSFORMER_ENGINE_AVAILABLE:
        layer_types.append("fp8_te")
    
    results = {}
    
    for name, batch_size, seq_len, in_features, out_features in configs:
        print(f"\n📊 Testing {name}: {in_features} → {out_features}, seq_len={seq_len}")
        results[name] = {}
        
        for layer_type in layer_types:
            print(f"  - {layer_type}...", end="", flush=True)
            try:
                result = benchmark_linear_layer(
                    layer_type, batch_size, seq_len, in_features, out_features,
                    num_warmup=20, num_iterations=100, test_backward=True
                )
                results[name][layer_type] = result
                print(f" ✓ (fwd: {result['forward_mean']:.2f}ms, bwd: {result['backward_mean']:.2f}ms)")
            except Exception as e:
                print(f" ✗ Error: {e}")
                results[name][layer_type] = None
    
    # Test accuracy
    print("\n🎯 Testing Accuracy Differences")
    accuracy_results = {}
    for name, _, _, in_features, out_features in configs[:3]:  # Test first 3 configs
        print(f"  - {name}...", end="", flush=True)
        acc_result = test_accuracy_difference(in_features, out_features)
        accuracy_results[name] = acc_result
        print(f" MSE: {acc_result['mse']:.2e}, Mean Rel Error: {acc_result['mean_relative_error']:.2%}")
    
    return results, accuracy_results


def plot_results(results: Dict, output_dir: Path):
    """Plot benchmark results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\nNote: matplotlib not available, skipping plots")
        print("Install with: pip install matplotlib")
        return
    
    # Prepare data for plotting
    configs = list(results.keys())
    layer_types = list(next(iter(results.values())).keys())
    
    # Forward pass performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(configs))
    width = 0.25
    
    for i, layer_type in enumerate(layer_types):
        forward_times = [results[cfg][layer_type]['forward_mean'] 
                        if results[cfg][layer_type] else 0 
                        for cfg in configs]
        ax1.bar(x + i*width - width, forward_times, width, label=layer_type)
    
    ax1.set_xlabel('Layer Configuration')
    ax1.set_ylabel('Forward Pass Time (ms)')
    ax1.set_title('Forward Pass Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Memory usage
    for i, layer_type in enumerate(layer_types):
        memory_usage = [results[cfg][layer_type]['memory_mb'] 
                       if results[cfg][layer_type] else 0 
                       for cfg in configs]
        ax2.bar(x + i*width - width, memory_usage, width, label=layer_type)
    
    ax2.set_xlabel('Layer Configuration')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Consumption')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fp8_vs_bf16_comparison.png', dpi=150)
    plt.close()
    
    # Create speedup summary
    print("\n📈 Performance Summary")
    print("=" * 60)
    
    for config in configs:
        if results[config]['bf16'] is None:
            continue
            
        bf16_time = results[config]['bf16']['forward_mean']
        print(f"\n{config}:")
        
        for layer_type in layer_types:
            if layer_type == 'bf16' or results[config][layer_type] is None:
                continue
            
            time = results[config][layer_type]['forward_mean']
            speedup = bf16_time / time
            memory_reduction = 1 - (results[config][layer_type]['memory_mb'] / 
                                  results[config]['bf16']['memory_mb'])
            
            print(f"  {layer_type}:")
            print(f"    - Speedup: {speedup:.2f}x")
            print(f"    - Memory reduction: {memory_reduction:.1%}")


if __name__ == "__main__":
    # Enable H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Run benchmarks
    results, accuracy_results = run_comprehensive_benchmark()
    
    # Plot results
    output_dir = Path("./fp8_benchmark_results")
    plot_results(results, output_dir)
    
    # Save detailed results
    import json
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump({
            "performance": results,
            "accuracy": accuracy_results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    print("\n💡 Key Insights:")
    print("1. FP8 quantized (simulated) typically shows memory savings but may be slower")
    print("2. True FP8 with Transformer Engine can provide actual speedups on H100")
    print("3. Accuracy loss is generally acceptable for inference tasks")
    print("4. Larger layers benefit more from FP8 optimization")
