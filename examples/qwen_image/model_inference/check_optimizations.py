"""
Check which optimizations are actually being used in the model
"""

import torch
import sys

print("Checking optimization status...")
print("="*60)

# 1. Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

# 2. Check Flash Attention
print("\n" + "-"*60)
print("Flash Attention Status:")
try:
    import flash_attn
    print(f"✅ flash_attn package installed: v{flash_attn.__version__}")
except ImportError:
    print("❌ flash_attn package NOT installed")

try:
    import flash_attn_interface
    print("✅ flash_attn_interface available (used by model)")
except ImportError:
    print("❌ flash_attn_interface NOT available")
    print("   This means the model is using SLOW attention!")

# Check if it's actually being used in the model
try:
    from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
    if FLASH_ATTN_3_AVAILABLE:
        print("✅ Model WILL use Flash Attention 3")
    else:
        print("❌ Model will NOT use Flash Attention 3")
        print("   Using torch.nn.functional.scaled_dot_product_attention instead")
except:
    print("⚠️ Could not check model's Flash Attention status")

# 3. Check other optimizations
print("\n" + "-"*60)
print("Other Optimizations:")
print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
print(f"CuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"CuDNN benchmark: {torch.backends.cudnn.benchmark}")

# 4. Installation instructions
print("\n" + "="*60)
print("TO FIX SLOW INFERENCE:")
print("="*60)

if 'flash_attn' not in sys.modules:
    print("\n1. Install Flash Attention 3 for H100:")
    print("   conda install -c nvidia cuda-toolkit")  # Ensure CUDA toolkit
    print("   pip install ninja  # Required for building")
    print("   pip install flash-attn --no-build-isolation")
    print("")
    print("   Or if that fails:")
    print("   git clone https://github.com/Dao-AILab/flash-attention.git")
    print("   cd flash-attention")
    print("   pip install .")
    print("")
    print("2. After installation, the model will automatically use Flash Attention")
    print("   Expected speedup: 2-3x for attention operations")
    print("")
    print("3. Then use torch.compile() for additional speedup")

# 5. Quick benchmark
print("\n" + "-"*60)
print("Quick attention benchmark:")

# Test attention speed
import time
import torch.nn.functional as F

B, H, N, D = 1, 24, 6144, 128  # Typical sizes for Qwen
device = "cuda"

q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)

# Warmup
for _ in range(5):
    _ = F.scaled_dot_product_attention(q, k, v)

# Measure
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    _ = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Attention speed: {elapsed/10*1000:.1f}ms per call")
print(f"TFLOPS: {4 * B * H * N * N * D / (elapsed/10) / 1e12:.1f}")

if elapsed/10 > 0.1:  # More than 100ms
    print("\n⚠️ ATTENTION IS VERY SLOW!")
    print("This is your bottleneck. Install Flash Attention!")
