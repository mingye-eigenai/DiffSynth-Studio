# Qwen-Image-Edit LoRA Training Scripts

This directory contains training scripts for Qwen-Image-Edit LoRA fine-tuning.

## Available Scripts

### 1. Standard Training
- **File**: `Qwen-Image-Edit-Ghost.sh`
- **Use case**: Standard training with separate Q, K, V layers
- **Speed**: Baseline
- **Compatibility**: Direct HuggingFace compatibility

### 2. Merged QKV Training (Recommended) ⚡
- **File**: `Qwen-Image-Edit-Ghost-qkv-merged-efficient.sh`
- **Use case**: Efficient training with merged QKV layers
- **Speed**: 10-15% faster than standard
- **Compatibility**: Requires conversion for HuggingFace (converter provided)

### 3. LoRA Fusion Example
- **File**: `Qwen-Image-LoRA-Fused-Example.sh`
- **Use case**: Fine-tune on top of existing LoRA (e.g., Lightning LoRA)

## Quick Comparison

| Feature | Standard | Merged QKV |
|---------|----------|------------|
| Training Speed | Baseline | **+10-15%** |
| Memory Usage | Baseline | Similar |
| Model Quality | ✓ | ✓ (Identical) |
| Setup Complexity | Simple | Simple |
| HF Compatibility | Direct | Needs conversion |
| **Recommended for** | Quick experiments | Production training |

## How to Choose

### Use Standard Training if:
- You need direct HuggingFace compatibility
- You're doing quick experiments
- You want maximum compatibility

### Use Merged QKV Training if: ⭐
- You want faster training (10-15% speedup)
- You're training for production
- You don't mind a simple conversion step

## Quick Start

### Standard Training
```bash
bash Qwen-Image-Edit-Ghost.sh
```

### Merged QKV Training (Faster)
```bash
bash Qwen-Image-Edit-Ghost-qkv-merged-efficient.sh
```

### With Pretrained LoRA
```bash
# Edit the script and uncomment:
# --lora_fused "./path/to/pretrained_lora.safetensors"

bash Qwen-Image-Edit-Ghost-qkv-merged-efficient.sh
```

## Configuration

All scripts share similar parameters:

```bash
--dataset_base_path          # Where your data is stored
--dataset_metadata_path      # CSV with image paths and prompts
--data_file_keys            # Columns to use: "image,edit_image"
--extra_inputs              # Additional inputs: "edit_image"
--learning_rate             # Learning rate (e.g., 1e-4)
--num_epochs               # Number of training epochs
--lora_rank                # LoRA rank (higher = more capacity)
```

## LoRA Target Modules

### Standard Training
```bash
--lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,txt_mlp.net.0.proj,txt_mlp.net.2"
```

### Merged QKV Training
```bash
--lora_target_modules "to_qkv,add_qkv_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,txt_mlp.net.0.proj,txt_mlp.net.2"
```

Notice:
- `to_q,to_k,to_v` → `to_qkv`
- `add_q_proj,add_k_proj,add_v_proj` → `add_qkv_proj`

## Converting Checkpoints

If you trained with merged QKV and need standard format:

```bash
python ../utils/convert_merged_qkv_checkpoint.py \
  --input merged_checkpoint.safetensors \
  --output standard_checkpoint.safetensors \
  --direction to_standard
```

## Performance Tips

1. **Batch Size**: Larger batches = better GPU utilization
2. **Gradient Checkpointing**: Saves memory, minimal speed impact
3. **Mixed Precision**: Use bfloat16 (already enabled in scripts)
4. **Data Workers**: Increase `--dataset_num_workers` if CPU is fast

## Output Structure

After training, you'll find:

```
models/train/Qwen-Image-Edit-Ghost_lora_merged_qkv_1012/
├── checkpoint-100.safetensors
├── checkpoint-200.safetensors
├── ...
├── checkpoint-final.safetensors
└── training_log.txt
```

## Next Steps

1. **Start Training**: Run one of the scripts above
2. **Monitor Progress**: Check loss curves and sample outputs
3. **Use Your Model**: See inference examples in `../model_inference/`
4. **Share Results**: Convert if needed and share on HuggingFace

## Support

- **Merged QKV Guide**: `../../MERGED_QKV_COMPLETE_GUIDE.md`
- **Quick Start**: `../QUICKSTART_MERGED_QKV.md`
- **Technical Details**: `../README_MERGED_QKV.md`

## Summary

🚀 **Recommendation**: Use `Qwen-Image-Edit-Ghost-qkv-merged-efficient.sh` for **10-15% faster training** with identical quality!

