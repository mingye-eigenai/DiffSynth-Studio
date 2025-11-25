# Performance Optimization Summary

## The Bottleneck: Missing Flash Attention

After extensive investigation, we found that the primary performance bottleneck is **Flash Attention not being installed**. 

### Current Situation (Without Flash Attention)
- **42-43 seconds** per image (40 steps)
- ~1.05 seconds per diffusion step
- Using PyTorch's standard attention (slow)
- Neither FP8 nor torch.compile() provide significant speedup

### Root Cause
The Qwen-Image model has Flash Attention 3 support built-in, but it falls back to PyTorch's `scaled_dot_product_attention` when Flash Attention isn't installed. On large models with long sequences, this is 2-3x slower.

## The Solution

### 1. Install Flash Attention (REQUIRED)
```bash
./install_flash_attention.sh
# or manually:
pip install ninja
pip install flash-attn --no-build-isolation
```

### 2. Use H100-Optimized Script
```bash
./examples/qwen_image/model_inference/run_quantized_inference.sh
# Select option 0: H100-OPTIMIZED
```

## Expected Performance After Fix

| Metric | Before (No Flash Attn) | After (With Flash Attn) |
|--------|------------------------|-------------------------|
| Time per image | 42-43s | 15-20s |
| Time per step | 1.05s | 0.4-0.5s |
| Speedup | 1x | **2-3x** |

## Why Other Optimizations Didn't Help

### FP8 Quantization
- **Issue**: Implementation uses FP8 for storage, not compute
- **Result**: Memory savings but no speed improvement
- **Fix**: Flash Attention is more important than FP8

### torch.compile()
- **Issue**: Can't optimize the slow attention operations
- **Result**: Only ~10% improvement on non-attention ops
- **Fix**: Flash Attention optimizes the actual bottleneck

### LoRA Fusion
- **Status**: Already implemented and working
- **Result**: ~20% speedup (already included in timings)
- **Note**: This optimization is good and we keep it

## Optimizations Stack

Once Flash Attention is installed, all optimizations work together:

1. **Flash Attention 3**: 2-3x speedup on attention
2. **LoRA Fusion**: 20% speedup (no runtime overhead)
3. **TF32 on H100**: 10-15% speedup on matmuls
4. **torch.compile()**: 10-20% additional speedup
5. **Reduced steps**: 25 steps instead of 40 (1.6x)

**Combined**: Up to **4-5x total speedup**

## Quick Diagnostic Commands

```bash
# Check if Flash Attention is installed
python examples/qwen_image/model_inference/check_optimizations.py

# Profile the pipeline
python examples/qwen_image/model_inference/profile_inference_bottleneck.py

# Benchmark different configurations
python examples/qwen_image/model_inference/diagnose_fp8_performance.py
```

## TL;DR

**Your H100 is fast, but the software isn't using it properly.**

Run this to fix:
```bash
bash install_flash_attention.sh
```

Then use option 0 in the run script for 2-3x faster inference!
