"""
Debug LoRA loading for merged QKV model.
"""

import torch
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth.models import ModelManager

print("="*60)
print("Debug LoRA Loading")
print("="*60)

# Load merged QKV LoRA checkpoint
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora_merged_qkv_1012/epoch-4.safetensors"
print(f"\n1. Loading LoRA checkpoint: {lora_path}")

lora_state_dict = load_state_dict(lora_path)
print(f"   Loaded {len(lora_state_dict)} tensors")

# Show first 20 keys
print("\n2. First 20 LoRA keys:")
for i, key in enumerate(sorted(lora_state_dict.keys())[:20]):
    shape = lora_state_dict[key].shape
    print(f"   {i+1}. {key}: {shape}")

# Check if merged QKV keys exist
print("\n3. Checking for merged QKV LoRA keys...")
merged_qkv_keys = [k for k in lora_state_dict.keys() if 'to_qkv' in k or 'add_qkv_proj' in k]
print(f"   Found {len(merged_qkv_keys)} merged QKV keys")

if merged_qkv_keys:
    print("   Sample merged QKV LoRA keys:")
    for key in sorted(merged_qkv_keys)[:5]:
        print(f"     - {key}")
else:
    print("   ⚠ NO merged QKV keys found!")

# Check what keys the LoRA has
print("\n4. Analyzing LoRA structure...")
lora_a_keys = [k for k in lora_state_dict.keys() if 'lora_A' in k]
lora_b_keys = [k for k in lora_state_dict.keys() if 'lora_B' in k]

print(f"   LoRA A matrices: {len(lora_a_keys)}")
print(f"   LoRA B matrices: {len(lora_b_keys)}")

# Check attn keys
attn_keys = [k for k in lora_state_dict.keys() if 'attn' in k]
print(f"   Attention LoRA keys: {len(attn_keys)}")

if len(attn_keys) > 0:
    print("   Sample attention LoRA keys:")
    for key in sorted(attn_keys)[:10]:
        print(f"     - {key}")

# Now load the model and try to apply LoRA
print("\n5. Loading merged QKV model...")
pipe = QwenImagePipeline(device="cpu", torch_dtype=torch.bfloat16)

dit_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
dit_config.download_if_necessary()

dit_path = dit_config.path
if isinstance(dit_path, list):
    state_dict = {}
    for shard_path in dit_path:
        state_dict.update(load_state_dict(shard_path))
else:
    state_dict = load_state_dict(dit_path)

dit_merged = QwenImageDiTMergedQKV(num_layers=60)
converter = dit_merged.state_dict_converter()
merged_state_dict = converter.from_diffusers(state_dict)

dit_merged = dit_merged.to(dtype=torch.bfloat16, device="cpu")
dit_merged.load_state_dict(merged_state_dict, strict=False)

pipe.dit = dit_merged

print("   ✓ Model loaded")

# Get model module names
print("\n6. Checking model module names...")
model_modules = {}
for name, module in pipe.dit.named_modules():
    if name:
        model_modules[name] = type(module).__name__

# Check block 0 attention modules
block0_attn_modules = {k: v for k, v in model_modules.items() if k.startswith('transformer_blocks.0.attn.')}
print(f"   Block 0 attention modules ({len(block0_attn_modules)}):")
for name, typ in sorted(block0_attn_modules.items())[:10]:
    print(f"     {name}: {typ}")

# Check if to_qkv exists
if 'transformer_blocks.0.attn.to_qkv' in model_modules:
    print("   ✓ to_qkv module exists in model")
else:
    print("   ✗ to_qkv module NOT FOUND in model!")

# Now try to load LoRA
print("\n7. Attempting to load LoRA...")
try:
    result = pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    print(f"   ✓ load_lora returned: {result}")
except Exception as e:
    print(f"   ✗ load_lora failed: {e}")
    import traceback
    traceback.print_exc()

# Check if LoRA was actually applied
print("\n8. Verifying LoRA application...")

# Check if Linear layers were wrapped
from diffsynth.vram_management import AutoWrappedLinear
for name, module in pipe.dit.named_modules():
    if 'transformer_blocks.0.attn.to_qkv' in name:
        print(f"   {name}: {type(module).__name__}")
        if isinstance(module, AutoWrappedLinear):
            print(f"     → Wrapped as AutoWrappedLinear ✓")
            # Check for LoRA attributes
            if hasattr(module, 'lora_A'):
                print(f"     → Has lora_A: {module.lora_A}")
            if hasattr(module, 'lora_B'):
                print(f"     → Has lora_B: {module.lora_B}")
        else:
            print(f"     → NOT wrapped (LoRA may not apply)")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if len(merged_qkv_keys) == 0:
    print("\n❌ PROBLEM: LoRA checkpoint doesn't have merged QKV keys!")
    print("   Your LoRA was probably saved with standard (non-merged) format.")
    print("   This happens if the training didn't use merged QKV properly.")
else:
    print("\n✓ LoRA has merged QKV keys")
    print("  Need to check if module names match for loading.")

print("="*60)

