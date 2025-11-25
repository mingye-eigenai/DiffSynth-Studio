# Quantization for Faster Inference

This guide explains how to use quantization to reduce memory usage and potentially improve inference speed for Qwen-Image models.

## What is Quantization?

Quantization reduces the precision of model weights from 16-bit or 32-bit to 8-bit (FP8 or INT8), which:
- Reduces memory usage by ~50%
- Can improve inference speed on supported hardware
- May slightly reduce quality (usually negligible)

## Available Options

### 1. FP8 Quantization (Standard)

**File**: `Qwen-Image-Edit-LoRA-Ghost-Quantized.py`

Features:
- Uses FP8 (8-bit floating point) for DiT model
- Reduces VRAM usage by ~40-50%
- Minimal quality loss
- Works on all modern GPUs

### 2. FP8 Quantization (Optimized)

**File**: `Qwen-Image-Edit-LoRA-Ghost-Quantized-Optimized.py`

Features:
- **Fuses LoRA weights** to eliminate runtime overhead
- Then applies FP8 quantization
- Fastest inference speed
- Same memory savings as standard FP8

### 3. Low VRAM Mode (Standard)

**File**: `Qwen-Image-Edit-LoRA-Ghost-LowVRAM.py`

Features:
- FP8 quantization + CPU offloading
- Works on 8GB GPUs
- Slower but memory efficient

### 4. Low VRAM Mode (Optimized)

**File**: `Qwen-Image-Edit-LoRA-Ghost-LowVRAM-Optimized.py`

Features:
- **Fuses LoRA weights** first
- Aggressive memory management (6GB limit)
- Works on RTX 3060 and up
- Faster than standard low VRAM mode

## Performance Comparison

| Mode | VRAM Usage | Speed | Quality | Best For |
|------|------------|-------|---------|----------|
| Original (BF16) | ~24GB | Baseline | Best | A100/A6000 |
| FP8 Standard | ~14GB | Good | Very Good | RTX 4090 |
| FP8 Optimized | ~14GB | **Best** | Very Good | RTX 4090/4080 |
| Low VRAM Standard | ~8GB | Slow | Good | RTX 3080/4070 |
| Low VRAM Optimized | ~6GB | Better | Good | RTX 3060/3070 |

## GPU Recommendations

| GPU | VRAM | Recommended Mode |
|-----|------|------------------|
| RTX 3060 | 12GB | Low VRAM Optimized |
| RTX 3070/3070Ti | 8GB | Low VRAM Optimized |
| RTX 3080/3080Ti | 10-12GB | FP8 Optimized |
| RTX 4070/4070Ti | 12GB | FP8 Optimized |
| RTX 4080 | 16GB | FP8 Optimized |
| RTX 4090 | 24GB | FP8 Optimized or Original |
| A100/A6000 | 40GB+ | Original (for quality) |

## When to Use Which Mode

### Use FP8 Optimized when:
- You have 12-24GB VRAM
- You want the fastest inference
- Quality is important
- Running batch processing

### Use Low VRAM Optimized when:
- You have 6-12GB VRAM
- Need balance of speed and memory
- Running on mid-range GPUs (RTX 3060, 3070)

### Use Standard versions when:
- You need to keep LoRA separate (not fused)
- Testing different LoRA weights
- Need flexibility to swap LoRAs

## Advanced Options

### Enable FP8 Attention (Experimental)
For GPUs with native FP8 support:
```python
enable_fp8_attention=True  # In pipe() call
```

### Custom VRAM Limits
Adjust based on your GPU:
```python
pipe.enable_vram_management(
    vram_limit=10.0,    # 10GB limit
    vram_buffer=1.5,    # 1.5GB safety buffer
)
```

### Mixed Precision
You can mix different precisions:
```python
# FP8 for DiT, BF16 for others
ModelConfig("transformer/*.safetensors", offload_dtype=torch.float8_e4m3fn),
ModelConfig("text_encoder/*.safetensors", offload_dtype=torch.bfloat16),
ModelConfig("vae/*.safetensors", offload_dtype=torch.bfloat16),
```

## Tips for Best Performance

1. **Lightning LoRA Benefits**: When using Lightning LoRA, reduce steps to 4-8 for faster inference
2. **Resolution**: Lower resolution = less memory and faster inference
3. **Batch Size**: Process one image at a time for lowest memory usage
4. **GPU Cache**: Call `torch.cuda.empty_cache()` between images if running out of memory

## Troubleshooting

### "CUDA out of memory"
- Use Low VRAM mode
- Reduce image resolution
- Decrease vram_limit parameter

### "Slow inference"
- Disable auto_offload if you have enough VRAM
- Use FP8 mode instead of Low VRAM mode
- Check if enable_fp8_attention helps (GPU dependent)

### "Quality degradation"
- FP8 usually has minimal impact
- If noticeable, use BF16 for VAE/text encoder
- Increase inference steps slightly
