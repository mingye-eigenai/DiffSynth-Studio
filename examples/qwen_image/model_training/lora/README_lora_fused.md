# Using the `lora_fused` Feature

The `lora_fused` feature allows you to fuse a pretrained LoRA into the base model weights before training a new LoRA. This is useful for:
- Building on top of existing LoRA models (e.g., Lightning LoRA)
- Creating specialized LoRAs that extend existing style LoRAs
- Chaining multiple LoRA adaptations

## How It Works

1. The pretrained LoRA weights are loaded and permanently merged into the base model
2. A new LoRA is then trained on top of these fused weights
3. The system automatically detects Lightning LoRA format (uses `lora_down`/`lora_up`) vs standard format (uses `lora_A`/`lora_B`)

## Usage Example

```bash
accelerate launch examples/qwen_image/model_training/train.py \
  --task "sft" \
  --dataset_base_path "datasets/your_dataset" \
  --trainable_models "" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_fused "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --output_path "./models/your_new_lora"
```

## Supported LoRA Formats

### Lightning LoRA Format
- Uses naming convention: `module.lora_down.weight`, `module.lora_up.weight`
- May include per-layer alpha values
- Automatically detected and handled

### Standard LoRA Format
- Uses naming convention: `module.lora_A.default.weight`, `module.lora_B.default.weight`
- Common in diffusers-style LoRAs

## Important Notes

1. The fused LoRA permanently modifies the base model weights in memory
2. The original model files on disk are not modified
3. The new LoRA is trained on top of the fused weights
4. Make sure the `--lora_base_model` matches the model the pretrained LoRA was trained for (usually "dit")

## Troubleshooting

If you see "0 tensors are updated by LoRA":
- Check that the LoRA file path is correct
- Verify the LoRA was trained for the same model architecture
- Ensure the `--lora_base_model` parameter matches the LoRA's target model
