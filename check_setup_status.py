#!/usr/bin/env python3
"""
Quick status check for optimal Qwen-Image inference setup
"""

import torch
import sys

print("🔍 Checking your setup for optimal performance...")
print("="*60)

# GPU Check
gpu_name = torch.cuda.get_device_name(0)
compute_cap = torch.cuda.get_device_capability(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"GPU: {gpu_name}")
print(f"VRAM: {vram:.1f} GB")
print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

status = []

# Check Flash Attention
try:
    import flash_attn
    # Force reload the module to get the updated version
    import importlib
    if 'diffsynth.models.qwen_image_dit' in sys.modules:
        importlib.reload(sys.modules['diffsynth.models.qwen_image_dit'])
    from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
    if FLASH_ATTN_3_AVAILABLE:
        status.append("✅ Flash Attention 3: INSTALLED & WORKING")
    else:
        status.append("⚠️ Flash Attention 3: INSTALLED but NOT detected by model")
except ImportError:
    status.append("❌ Flash Attention 3: NOT INSTALLED (2-3x slower!)")

# Check TF32
if torch.backends.cuda.matmul.allow_tf32:
    status.append("✅ TF32: ENABLED")
else:
    status.append("⚠️ TF32: DISABLED (enable for 10-15% speedup)")

# Check PyTorch version
if "2.8" in torch.__version__ or "3." in torch.__version__:
    status.append("✅ PyTorch: Modern version")
else:
    status.append("⚠️ PyTorch: Consider upgrading to 2.8+")

# GPU-specific recommendations
if compute_cap[0] >= 9:  # H100/H200
    status.append("✅ GPU: H100/H200 with FP8 support")
    optimal_script = "Qwen-Image-Edit-LoRA-Ghost-H100-Optimized.py"
elif compute_cap[0] == 8:  # A100/RTX 30xx/40xx
    status.append("⚠️ GPU: No native FP8 compute (use BF16)")
    optimal_script = "Qwen-Image-Edit-LoRA-Ghost-TorchCompile.py"
else:
    status.append("⚠️ GPU: Older generation")
    optimal_script = "Qwen-Image-Edit-LoRA-Ghost-Quantized-Optimized.py"

print("\n📊 Status:")
for s in status:
    print(f"  {s}")

# Recommendations
print("\n💡 Recommendations:")
if "NOT INSTALLED" in str(status):
    print("  1. INSTALL FLASH ATTENTION (most important!):")
    print("     bash install_flash_attention.sh")
    print("")

if "TF32: DISABLED" in str(status):
    print("  2. Enable TF32 in your scripts:")
    print("     torch.backends.cuda.matmul.allow_tf32 = True")
    print("")

print(f"  Best script for your setup: {optimal_script}")
print(f"  Run: ./examples/qwen_image/model_inference/run_quantized_inference.sh")

# Performance estimate
if "Flash Attention 3: INSTALLED & WORKING" in str(status):
    print("\n🚀 Expected performance: ~15-20s per image")
else:
    print("\n🐌 Current performance: ~40-45s per image")
    print("   (Install Flash Attention for 2-3x speedup!)")

print("\n" + "="*60)
