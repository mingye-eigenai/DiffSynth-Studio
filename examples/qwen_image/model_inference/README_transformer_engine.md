# NVIDIA Transformer Engine FP8 Integration for H100

This guide explains how to use NVIDIA Transformer Engine for true FP8 acceleration on H100 GPUs.

## What's Been Added

1. **Installation Script**: `install_transformer_engine.sh`
   - Installs NVIDIA Transformer Engine for H100 FP8 support
   - Verifies installation and GPU compatibility

2. **FP8 Wrapper Module**: `diffsynth/models/qwen_image_transformer_engine.py`
   - `TransformerEngineFP8Linear`: Drop-in replacement for nn.Linear with FP8
   - `TransformerEngineFP8Attention`: FP8-accelerated attention
   - Helper functions for easy integration

3. **H100 TE-FP8 Script**: `Qwen-Image-Edit-LoRA-Ghost-H100-TE-FP8.py`
   - Uses Transformer Engine for true FP8 compute
   - Combines with Flash Attention and torch.compile
   - Should be faster than standard FP8 quantization

4. **Performance Comparison**: `compare_te_performance.py`
   - Benchmarks different implementations
   - Shows real performance differences

## Quick Start

```bash
# 1. Install Transformer Engine (requires H100/H200)
bash install_transformer_engine.sh

# 2. Run optimized inference with TE FP8
python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-H100-TE-FP8.py

# 3. Compare performance
python examples/qwen_image/model_inference/compare_te_performance.py
```

## Key Differences: Standard FP8 vs Transformer Engine FP8

### Standard FP8 (Current DiffSynth Implementation)
- **Storage only**: Weights stored in FP8 format
- **Compute in BF16**: FP8 → BF16 conversion for computation
- **Result**: Memory savings but SLOWER than BF16

### Transformer Engine FP8
- **True FP8 compute**: Uses H100's FP8 tensor cores
- **Optimized kernels**: Custom CUDA kernels for FP8 operations
- **Result**: Memory savings AND potential speedup

## Performance Expectations

On H100:
- **BF16 baseline**: 1.0x
- **Standard FP8**: 0.6-0.8x (slower!)
- **Transformer Engine FP8**: 1.1-1.5x (faster!)
- **Best combo**: TE FP8 + Flash Attention + torch.compile

## Important Notes

1. **H100/H200 Only**: Transformer Engine FP8 requires Hopper architecture
2. **PyTorch 2.1+**: Need recent PyTorch version
3. **CUDA 11.8+**: Minimum CUDA version required
4. **Model Size**: Larger models benefit more from FP8

## Updated Settings

In the original H100 script, I've changed:
```python
enable_fp8_attention = False  # Flash Attention doesn't support FP8
```

This is because Flash Attention 3 doesn't support FP8 inputs yet. The TE-FP8 script handles this properly.

## Troubleshooting

If Transformer Engine installation fails:
```bash
# Try manual installation
pip install git+https://github.com/NVIDIA/TransformerEngine.git

# Or with specific PyTorch version
pip install transformer-engine[pytorch]
```

## Future Improvements

1. **Flash Attention 4**: May support FP8 inputs directly
2. **PyTorch native FP8**: Coming in future PyTorch versions
3. **Custom kernels**: Could write specific FP8 kernels for this model

For now, Transformer Engine provides the best path to true FP8 acceleration on H100.
