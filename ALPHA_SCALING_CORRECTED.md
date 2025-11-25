# LoRA Alpha Scaling - CORRECTED Understanding

## The Real Issue

I was wrong about needing alpha=32. Your observation was correct!

## Why alpha=1.0 is Correct

### During Training (PEFT Library)

Your LoRA was trained using PEFT's `inject_adapter_in_model`, which internally applies:

```python
# In diffsynth/trainers/utils.py line 388-391:
lora_alpha = lora_rank  # Sets to 32
lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha)

# PEFT internally computes:
weight_update_during_training = (lora_alpha / lora_rank) * (B @ A)
                               = (32 / 32) * (B @ A)
                               = 1.0 * (B @ A)
```

The LoRA adapts with this scaling built-in.

### During Inference (GeneralLoRALoader)

The `GeneralLoRALoader` does NOT divide by rank:

```python
# In diffsynth/lora/__init__.py line 40:
weight_lora = alpha * torch.mm(weight_up, weight_down)
```

### The Result

Your saved LoRA checkpoint has weights that expect **alpha=1.0**:

- **alpha=1.0**: ✅ Correct - applies LoRA as trained
- **alpha=4.0**: ⚠️ Too strong - 4x overtrained effect
- **alpha=32.0**: ❌ Way too strong - 32x overtrained effect → noise!

## Why alpha=1 Seems to Have No Effect

If alpha=1 shows minimal effect, it could mean:

1. **LoRA needs more training** - 4-5 epochs might not be enough
2. **Learning rate too low** - LoRA didn't adapt much
3. **LoRA is working but subtle** - check outputs more carefully

## Recommendations

### Option 1: Use Current LoRA with alpha=1
```python
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
```

The effect should be subtle but correct.

### Option 2: Increase Alpha Slightly (if you want stronger effect)
```python
pipe.load_lora(pipe.dit, lora_path, alpha=2.0)  # 2x stronger
pipe.load_lora(pipe.dit, lora_path, alpha=4.0)  # 4x stronger
```

But be careful - too high causes noise!

### Option 3: Train Longer or With Higher LR

If you want stronger LoRA effect without artifacts:
- Train for more epochs (10-20)
- Or increase learning rate (2e-4 instead of 1e-4)
- Or increase LoRA rank (64 instead of 32)

## Test Results Explained

Your observations now make sense:
- **alpha=1**: Correct scaling, subtle effect ✅
- **alpha=4**: 4x overtrained → slight noise ⚠️
- **alpha=32**: 32x overtrained → complete noise ❌

## Updated Scripts

I've corrected both scripts to use **alpha=1.0** (the correct value):
- `Qwen-Image-Edit-LoRA-Ghost-MergedQKV.py`
- `Qwen-Image-Edit-LoRA-Ghost-MergedQKV-Single.py`

## Summary

**Use alpha=1.0** - this is the correct value for your PEFT-trained LoRA! 🎯

If the effect is too subtle, train longer or with higher learning rate rather than increasing alpha.

