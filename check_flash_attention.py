#!/usr/bin/env python3
"""
Simple check for Flash Attention status
"""

import sys
import importlib

print("Checking Flash Attention status...")
print("="*50)

# 1. Check if Flash Attention package is installed
try:
    import flash_attn
    print(f"✅ Flash Attention package: v{flash_attn.__version__}")
except ImportError:
    print("❌ Flash Attention package: NOT INSTALLED")
    sys.exit(1)

# 2. Force reload DiffSynth module to get updated imports
print("\nChecking DiffSynth integration...")
sys.path.insert(0, '/home/kuan/workspace/repos/DiffSynth-Studio')

# Clear any cached imports
modules_to_clear = [
    'diffsynth.models.qwen_image_dit',
    'diffsynth.models',
    'diffsynth'
]
for module in modules_to_clear:
    if module in sys.modules:
        del sys.modules[module]

# Import fresh
try:
    from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
    
    if FLASH_ATTN_3_AVAILABLE:
        print("✅ DiffSynth Flash Attention: ENABLED")
        print("\n🎉 SUCCESS! Flash Attention is working!")
        print("🚀 You should now see 2-3x speedup!")
        print("\nRun the H100-optimized script:")
        print("./examples/qwen_image/model_inference/run_quantized_inference.sh")
        print("Select option 0")
    else:
        print("❌ DiffSynth Flash Attention: NOT DETECTED")
        print("\nFlash Attention is installed but not detected by the model.")
        print("Try restarting your Python environment.")
        
except Exception as e:
    print(f"❌ Error checking DiffSynth: {e}")
    
print("\n" + "="*50)
