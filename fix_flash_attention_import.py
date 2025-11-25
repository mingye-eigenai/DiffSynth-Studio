#!/usr/bin/env python3
"""
Fix Flash Attention import issue
"""

import sys

print("Checking Flash Attention installation...")

# 1. Check what's actually installed
try:
    import flash_attn
    print(f"✅ flash_attn v{flash_attn.__version__} is installed")
    
    # List available modules
    print("\nAvailable flash_attn modules:")
    import pkgutil
    for importer, modname, ispkg in pkgutil.iter_modules(flash_attn.__path__, flash_attn.__name__ + "."):
        print(f"  - {modname}")
    
except ImportError as e:
    print(f"❌ flash_attn import failed: {e}")
    sys.exit(1)

# 2. Try different import methods
print("\nTrying different imports:")

# Method 1: flash_attn_interface (old style)
try:
    import flash_attn_interface
    print("✅ flash_attn_interface imported successfully")
except ImportError:
    print("❌ flash_attn_interface not found (this is OK for newer versions)")

# Method 2: flash_attn.flash_attn_interface
try:
    from flash_attn import flash_attn_interface
    print("✅ flash_attn.flash_attn_interface imported successfully")
except ImportError:
    print("❌ flash_attn.flash_attn_interface not found")

# Method 3: Direct function imports (modern way)
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    print("✅ flash_attn functions imported successfully")
    print("   This is the modern import method!")
except ImportError as e:
    print(f"❌ flash_attn functions not found: {e}")

# Method 4: Check for flash_attn_2_cuda
try:
    import flash_attn_2_cuda
    print("✅ flash_attn_2_cuda module found")
except ImportError:
    print("❌ flash_attn_2_cuda not found")

# 3. Check model compatibility
print("\n" + "-"*50)
print("Checking DiffSynth model compatibility:")

try:
    # Check if the model is looking for the right import
    import diffsynth.models.qwen_image_dit as qwen_dit
    
    # Show the import section
    with open(qwen_dit.__file__, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:20]):
            if 'flash_attn' in line:
                print(f"Line {i+1}: {line.strip()}")
    
    print(f"\nFLASH_ATTN_3_AVAILABLE = {qwen_dit.FLASH_ATTN_3_AVAILABLE}")
    
except Exception as e:
    print(f"Could not check model: {e}")

# 4. Solution
print("\n" + "="*50)
print("SOLUTION:")
print("="*50)

print("""
The issue is that DiffSynth is looking for 'flash_attn_interface' which is the old import style.
Flash Attention 2.8.3 uses a different import structure.

To fix this, we need to modify the import in the model file.

Run this command to fix it:

sed -i 's/import flash_attn_interface/from flash_attn import flash_attn_interface/g' /home/kuan/workspace/repos/DiffSynth-Studio/diffsynth/models/qwen_image_dit.py

Or manually edit /home/kuan/workspace/repos/DiffSynth-Studio/diffsynth/models/qwen_image_dit.py:
Change:
    import flash_attn_interface
To:
    from flash_attn import flash_attn_interface

Alternative: If that doesn't work, change the imports to:
    from flash_attn import flash_attn_func
    # Then replace flash_attn_interface.flash_attn_func with flash_attn_func
""")
