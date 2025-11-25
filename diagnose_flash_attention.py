#!/usr/bin/env python3
"""
Diagnose Flash Attention installation issues
"""

import os
import sys
import subprocess
import torch

print("Flash Attention Installation Diagnostics")
print("="*50)

# 1. Python & PyTorch info
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")

# 2. Check CUDA installation
print("\n" + "-"*50)
print("CUDA Installation Check:")

cuda_paths = [
    "/usr/local/cuda",
    "/usr/local/cuda-12.8", 
    "/usr/local/cuda-12",
    "/usr/local/cuda-11.8",
    os.environ.get("CUDA_HOME", ""),
]

found_cuda = None
for path in cuda_paths:
    if path and os.path.exists(path):
        print(f"✅ Found CUDA at: {path}")
        if os.path.exists(f"{path}/bin/nvcc"):
            print(f"   nvcc: {path}/bin/nvcc")
            found_cuda = path
            break

if not found_cuda:
    print("❌ No CUDA installation found!")
else:
    # Check nvcc version
    try:
        result = subprocess.run([f"{found_cuda}/bin/nvcc", "--version"], 
                              capture_output=True, text=True)
        print(f"   nvcc version: {result.stdout.split('release')[1].split(',')[0].strip()}")
    except:
        print("   Could not get nvcc version")

# 3. Environment variables
print("\n" + "-"*50)
print("Environment Variables:")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"PATH includes CUDA: {'cuda' in os.environ.get('PATH', '').lower()}")

# 4. Compiler check
print("\n" + "-"*50)
print("Compiler Check:")
try:
    gcc_result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    print(f"✅ GCC found: {gcc_result.stdout.split('\\n')[0]}")
except:
    print("❌ GCC not found - needed for compilation")

try:
    g_result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    print(f"✅ G++ found: {g_result.stdout.split('\\n')[0]}")
except:
    print("❌ G++ not found - needed for compilation")

# 5. Recommended installation command
print("\n" + "="*50)
print("RECOMMENDED INSTALLATION COMMAND:")
print("="*50)

if found_cuda:
    print(f"\nCUDA_HOME={found_cuda} pip install flash-attn --no-build-isolation\n")
    
    print("Or run the automated installer:")
    print("bash install_flash_attention_nunchaku.sh")
else:
    print("\n⚠️  No CUDA found! You need to:")
    print("1. Install CUDA toolkit: conda install -c nvidia cuda-toolkit")
    print("2. Or set CUDA_HOME to your CUDA installation")

# 6. Check if already installed but not working
print("\n" + "-"*50)
try:
    import flash_attn
    print(f"✅ flash_attn is installed: v{flash_attn.__version__}")
    try:
        import flash_attn_interface
        print("✅ flash_attn_interface is available")
    except:
        print("❌ flash_attn_interface not available - partial installation?")
except ImportError:
    print("❌ flash_attn not installed")
    
# 7. PyTorch built-in optimizations
print("\n" + "-"*50)
print("Alternative: PyTorch Built-in Optimizations")
print(f"Torch SDPA available: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")
print(f"Torch compile available: {hasattr(torch, 'compile')}")
print("\nIf Flash Attention won't install, these provide some speedup.")
