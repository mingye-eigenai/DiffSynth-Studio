# LoRA Stacking for Inference

When using LoRAs for inference, it's crucial to recreate the same model state that was used during training.

## Key Principle

**Your inference setup must match your training setup!**

## Scenarios

### Scenario 1: LoRA trained on base model

If you trained your LoRA directly on the base Qwen-Image-Edit model:

```python
# Training command used:
# accelerate launch train.py --lora_base_model "dit" ...
# (WITHOUT --lora_fused)

# Inference:
pipe = QwenImagePipeline.from_pretrained(...)
pipe.load_lora(pipe.dit, "your_lora.safetensors", alpha=1.0)
# That's it! No base LoRA fusion needed
```

### Scenario 2: LoRA trained on Lightning-fused model

If you trained your LoRA on top of Lightning (or any other base LoRA):

```python
# Training command used:
# accelerate launch train.py \
#   --lora_fused "./models/Lightning-LoRA.safetensors" \
#   --lora_base_model "dit" ...

# Inference (must fuse Lightning first!):
pipe = QwenImagePipeline.from_pretrained(...)

# Step 1: Fuse Lightning LoRA
lightning_lora = load_state_dict("./models/Lightning-LoRA.safetensors")
loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
loader.load(pipe.dit, lightning_lora, alpha=1.0)

# Step 2: Load your LoRA on top
pipe.load_lora(pipe.dit, "your_lora.safetensors", alpha=1.0)
```

## Common Mistakes

1. **Wrong order**: Loading your LoRA before fusing the base LoRA
2. **Missing base fusion**: Forgetting to fuse the base LoRA when your model was trained on it
3. **Unnecessary fusion**: Fusing a base LoRA when your model was trained on vanilla weights

## How to Check

Look at your training logs or script:
- If you used `--lora_fused`, you MUST fuse that same LoRA during inference
- If you didn't use `--lora_fused`, you should NOT fuse any base LoRA during inference

## Performance Tips

- Lightning LoRA allows 4-8 step inference instead of 20-50 steps
- When stacking LoRAs, the base LoRA's benefits (like faster inference) are preserved
- You can adjust alpha values for both LoRAs independently if needed
