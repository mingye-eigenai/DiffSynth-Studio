"""
Check if LoRA is actually changing the model weights.
"""

import torch
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth.models import ModelManager

print("="*60)
print("Verifying LoRA Actually Changes Weights")
print("="*60)

# Load model
print("\n1. Loading merged QKV model...")
pipe = QwenImagePipeline(device="cuda", torch_dtype=torch.bfloat16)

text_encoder_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")

for config in [text_encoder_config, vae_config]:
    config.download_if_necessary()

model_manager = ModelManager()
model_manager.load_model(text_encoder_config.path, device="cuda", torch_dtype=torch.bfloat16)
model_manager.load_model(vae_config.path, device="cuda", torch_dtype=torch.bfloat16)

pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
pipe.vae = model_manager.fetch_model("qwen_image_vae")

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

dit_merged = dit_merged.to(dtype=torch.bfloat16, device="cuda")
dit_merged.load_state_dict(merged_state_dict, strict=False)

pipe.dit = dit_merged
print("✓ Model loaded")

# Get a weight BEFORE LoRA
print("\n2. Checking weight BEFORE LoRA...")
weight_before = pipe.dit.transformer_blocks[0].attn.to_qkv.weight.clone()
print(f"   Block 0 to_qkv.weight:")
print(f"     Mean: {weight_before.mean().item():.8f}")
print(f"     Std: {weight_before.std().item():.8f}")
print(f"     Device: {weight_before.device}")
print(f"     Dtype: {weight_before.dtype}")

# Load LoRA
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora_merged_qkv_1012/epoch-4.safetensors"
print(f"\n3. Loading LoRA from: {lora_path}")

pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

# Get weight AFTER LoRA
print("\n4. Checking weight AFTER LoRA...")
weight_after = pipe.dit.transformer_blocks[0].attn.to_qkv.weight
print(f"   Block 0 to_qkv.weight:")
print(f"     Mean: {weight_after.mean().item():.8f}")
print(f"     Std: {weight_after.std().item():.8f}")
print(f"     Device: {weight_after.device}")
print(f"     Dtype: {weight_after.dtype}")

# Compare
print("\n5. Comparison...")
if torch.equal(weight_before, weight_after):
    print("   ❌ WEIGHTS DID NOT CHANGE!")
    print("      LoRA was NOT applied.")
else:
    diff = (weight_after - weight_before).abs()
    print(f"   ✓ Weights changed!")
    print(f"     Max diff: {diff.max().item():.8f}")
    print(f"     Mean diff: {diff.mean().item():.8f}")
    print(f"     LoRA WAS applied successfully!")

# Also check a LoRA key directly
print("\n6. Checking LoRA state dict...")
lora_state_dict = load_state_dict(lora_path)
lora_key_a = "transformer_blocks.0.attn.to_qkv.lora_A.default.weight"
lora_key_b = "transformer_blocks.0.attn.to_qkv.lora_B.default.weight"

if lora_key_a in lora_state_dict and lora_key_b in lora_state_dict:
    lora_a = lora_state_dict[lora_key_a]
    lora_b = lora_state_dict[lora_key_b]
    print(f"   LoRA A shape: {lora_a.shape}, dtype: {lora_a.dtype}")
    print(f"   LoRA B shape: {lora_b.shape}, dtype: {lora_b.dtype}")
    
    # Compute what the LoRA update should be
    lora_update = torch.mm(lora_b, lora_a)  # alpha=1.0
    print(f"   LoRA update shape: {lora_update.shape}")
    print(f"   LoRA update mean: {lora_update.mean().item():.8f}")
    print(f"   LoRA update std: {lora_update.std().item():.8f}")
else:
    print(f"   ✗ LoRA keys not found in checkpoint!")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if torch.equal(weight_before, weight_after):
    print("\n❌ Problem: LoRA is NOT being applied to weights")
    print("\nPossible causes:")
    print("  1. Device mismatch (LoRA on CPU, model on CUDA)")
    print("  2. Module name mismatch")
    print("  3. LoRA loader not finding the modules")
else:
    print("\n✓ LoRA IS being applied to weights correctly")
    print("  If inference still produces same output, the issue is elsewhere.")

print("="*60)

