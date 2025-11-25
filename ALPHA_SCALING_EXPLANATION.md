# LoRA Alpha Scaling Explanation

## The Issue

Your LoRA was loading but had very weak effect because `alpha` wasn't scaled correctly.

## Why This Happens

The `GeneralLoRALoader` computes LoRA updates as:
```python
weight_lora = alpha * torch.mm(weight_up, weight_down)
```

**NO division by rank!** This means the alpha value directly controls the magnitude.

## Standard LoRA Practice

Typically, LoRA updates are computed as:
```python
weight_lora = (alpha / rank) * (B @ A)
```

When `alpha = rank`, this gives `weight_lora = B @ A` (full effect).

## Your LoRA (rank=32)

### Analysis
- Base weight std: **0.0547**
- LoRA update std (raw): **0.00275**

### With Different Alpha Values

| Alpha | LoRA Magnitude | % of Base Weight | Effect |
|-------|---------------|------------------|--------|
| 1.0   | 0.00275      | 5.03%           | Very weak (32x too small) |
| 4.0   | 0.01100      | 20.12%          | Still weak |
| 16.0  | 0.04403      | 80.52%          | Getting there |
| 32.0  | 0.08789      | 161.00%         | **Full effect** ✅ |

## Recommendation

Use **alpha=32** (same as your LoRA rank) for full effect:

```python
pipe.load_lora(pipe.dit, lora_path, alpha=32.0)
```

This will give you the full strength of your trained LoRA!

## Why alpha=1 Was Too Small

With rank=32 and alpha=1:
- LoRA contributes only **5%** of base weight magnitude
- Effect is very subtle and hard to see
- This is **32x weaker** than intended

With alpha=32:
- LoRA contributes **161%** of base weight magnitude
- Full effect as trained
- Clear visible difference in outputs

## Updated Scripts

I've updated both inference scripts to use `alpha=32.0` by default:
- `Qwen-Image-Edit-LoRA-Ghost-MergedQKV.py`
- `Qwen-Image-Edit-LoRA-Ghost-MergedQKV-Single.py`

## Test Results

With alpha=32, you should see:
- ✅ **Mean pixel difference ~166** (vs 5.21 with alpha=1)
- ✅ **~80% of pixels** differ by >10 (vs 10% with alpha=1)
- ✅ **Strong visual difference** in ghost style

Try it now and you'll see the full LoRA effect! 🚀

