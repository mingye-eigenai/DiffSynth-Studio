"""
Check if your GPU supports native FP8 computation and explain FP8 performance.
"""

import torch

def check_fp8_support():
    """Check GPU capabilities for FP8"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    compute_capability = torch.cuda.get_device_capability(device)
    
    print(f"GPU: {device_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print()
    
    # Check FP8 support
    # Native FP8 computation requires compute capability 9.0+ (Hopper architecture)
    has_native_fp8 = compute_capability[0] >= 9
    
    if has_native_fp8:
        print("✅ Native FP8 Support: YES")
        print("   Your GPU can perform FP8 computations natively.")
        print("   You should see speed improvements with FP8.")
    else:
        print("❌ Native FP8 Support: NO")
        print("   Your GPU does NOT support native FP8 computation.")
        print()
        print("   What happens with FP8 on your GPU:")
        print("   1. Weights are stored in FP8 format (saves memory)")
        print("   2. For computation: FP8 → FP16/FP32 conversion")
        print("   3. After computation: FP16/FP32 → FP8 conversion")
        print("   4. Result: Memory savings but NO speed improvement")
        print("   5. Actually SLOWER due to conversion overhead!")
    
    print("\n" + "="*60)
    print("FP8 Performance Summary:")
    print("="*60)
    
    if compute_capability[0] >= 9:
        print("Hopper (H100, H200) and newer:")
        print("- Native FP8 tensor cores")
        print("- 2-4x faster than FP16")
        print("- 50% memory reduction")
    elif compute_capability[0] == 8:
        print("Ampere (RTX 30xx, A100) and Ada Lovelace (RTX 40xx):")
        print("- NO native FP8 computation")
        print("- FP8 storage only (memory savings)")
        print("- Conversion overhead makes it SLOWER")
        print("- Use FP16/BF16 for best speed")
    else:
        print("Older GPUs:")
        print("- No FP8 support at all")
        print("- Use FP16 for best performance")
    
    print("\nRecommendation for your GPU:")
    if has_native_fp8:
        print("✅ Use FP8 quantization for both memory and speed benefits")
    else:
        print("⚠️  FP8 will reduce memory but SLOW DOWN inference")
        print("💡 For speed on your GPU:")
        print("   1. Use BF16/FP16 (no quantization)")
        print("   2. Fuse LoRA weights to reduce overhead")
        print("   3. Use torch.compile() if supported")
        print("   4. Consider INT8 quantization instead")

if __name__ == "__main__":
    check_fp8_support()
    
    print("\n" + "="*60)
    print("Why FP8 is slow on consumer GPUs:")
    print("="*60)
    print("1. Consumer GPUs (RTX 3xxx, 4xxx) lack FP8 tensor cores")
    print("2. FP8 is only used for storage, not computation")
    print("3. Constant conversion FP8↔FP16 adds overhead")
    print("4. Memory bandwidth savings don't offset conversion cost")
    print("5. Only H100/H200 have true FP8 acceleration")
    
    print("\nFor RTX 30xx/40xx GPUs, better options:")
    print("- INT8 quantization (if supported)")
    print("- Model pruning/distillation")
    print("- LoRA fusion (what we did)")
    print("- Flash Attention")
    print("- torch.compile()")
