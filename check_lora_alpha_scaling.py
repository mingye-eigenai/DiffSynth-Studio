"""
Check if alpha is being scaled correctly with rank.
Standard LoRA: weight_update = (alpha / rank) * (B @ A)
"""

import torch
from diffsynth import load_state_dict

print("="*60)
print("Checking LoRA Alpha Scaling")
print("="*60)

# Load LoRA
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora_merged_qkv_1012/epoch-4.safetensors"
print(f"\nLoading LoRA: {lora_path}")
lora_state_dict = load_state_dict(lora_path)

# Check a specific LoRA layer
lora_a_key = "transformer_blocks.0.attn.to_qkv.lora_A.default.weight"
lora_b_key = "transformer_blocks.0.attn.to_qkv.lora_B.default.weight"

lora_a = lora_state_dict[lora_a_key]
lora_b = lora_state_dict[lora_b_key]

print(f"\nLoRA matrices for block 0 to_qkv:")
print(f"  LoRA A shape: {lora_a.shape}")
print(f"  LoRA B shape: {lora_b.shape}")

# The rank is the middle dimension
rank = lora_a.shape[0]
print(f"  Rank: {rank}")

# Check for alpha parameter in checkpoint
alpha_keys = [k for k in lora_state_dict.keys() if 'alpha' in k.lower()]
print(f"\nAlpha keys in checkpoint: {len(alpha_keys)}")
if alpha_keys:
    for key in alpha_keys[:5]:
        print(f"  - {key}: {lora_state_dict[key]}")

# Calculate LoRA update without scaling
lora_update_no_scale = torch.mm(lora_b, lora_a)
print(f"\nLoRA update (B @ A, no scaling):")
print(f"  Mean: {lora_update_no_scale.mean().item():.8f}")
print(f"  Std: {lora_update_no_scale.std().item():.8f}")
print(f"  Max: {lora_update_no_scale.abs().max().item():.8f}")

# Calculate with alpha=1 (what we're using)
alpha = 1.0
lora_update_alpha1 = alpha * lora_update_no_scale
print(f"\nWith alpha=1.0:")
print(f"  Mean: {lora_update_alpha1.mean().item():.8f}")
print(f"  Std: {lora_update_alpha1.std().item():.8f}")
print(f"  Max: {lora_update_alpha1.abs().max().item():.8f}")

# Calculate with alpha=rank (standard practice)
alpha_rank = float(rank)
lora_update_alpha_rank = alpha_rank * lora_update_no_scale
print(f"\nWith alpha=rank ({rank}):")
print(f"  Mean: {lora_update_alpha_rank.mean().item():.8f}")
print(f"  Std: {lora_update_alpha_rank.std().item():.8f}")
print(f"  Max: {lora_update_alpha_rank.abs().max().item():.8f}")

# Load a base weight to compare magnitude
from diffsynth.pipelines.qwen_image import ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV

dit_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
dit_config.download_if_necessary()

dit_path = dit_config.path
if isinstance(dit_path, list):
    state_dict = {}
    for shard_path in dit_path[:1]:  # Just load first shard
        state_dict.update(load_state_dict(shard_path))
else:
    state_dict = load_state_dict(dit_path)

# Get base to_qkv weight (if exists in first shard)
base_qkv_keys = [k for k in state_dict.keys() if 'transformer_blocks.0.attn.to_q.weight' in k]
if base_qkv_keys:
    base_q = state_dict['transformer_blocks.0.attn.to_q.weight']
    print(f"\nBase to_q.weight magnitude:")
    print(f"  Mean: {base_q.mean().item():.8f}")
    print(f"  Std: {base_q.std().item():.8f}")
    print(f"  Max: {base_q.abs().max().item():.8f}")
    
    print(f"\nLoRA update as % of base weight:")
    print(f"  With alpha=1: {(lora_update_alpha1.std() / base_q.std() * 100).item():.2f}%")
    print(f"  With alpha=rank: {(lora_update_alpha_rank.std() / base_q.std() * 100).item():.2f}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

print(f"\nStandard LoRA practice:")
print(f"  alpha = rank ({rank}) for 1:1 scaling")
print(f"  This gives LoRA update ~{(lora_update_alpha_rank.std() / base_q.std() * 100).item():.2f}% of base weight magnitude")

print(f"\nCurrent setting:")
print(f"  alpha = 1.0")
print(f"  This gives LoRA update ~{(lora_update_alpha1.std() / base_q.std() * 100).item():.2f}% of base weight magnitude")
print(f"  This is {rank}x smaller than standard!")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

print(f"\n💡 Try loading LoRA with alpha={rank} instead of alpha=1.0:")
print(f"   pipe.load_lora(pipe.dit, lora_path, alpha={rank})")
print("\nThis will give you the full LoRA effect!")

print("="*60)

