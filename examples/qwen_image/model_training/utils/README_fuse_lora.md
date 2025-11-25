# LoRA Weight Fusion Tool

This tool allows you to permanently fuse LoRA weights into a base model and save the result. This is useful for:
- Eliminating runtime LoRA overhead during inference
- Creating a new base model for further fine-tuning
- Distributing a single model file instead of base + LoRA

## Features

- Supports both standard LoRA format and Lightning LoRA format
- Automatically detects the LoRA format
- Saves fused weights in safetensors format
- Preserves model precision (fp32, fp16, bf16)
- Generates metadata about the fusion process

## Usage

### Basic Usage

```bash
python examples/qwen_image/model_training/utils/fuse_lora_weights.py \
  --lora_path "path/to/your/lora.safetensors" \
  --output_dir "./models/fused_model" \
  --target_model "dit" \
  --lora_alpha 1.0
```

### Parameters

- `--lora_path`: Path to the LoRA weights file (required)
- `--output_dir`: Directory to save the fused model (required)
- `--target_model`: Which model component to apply LoRA to (default: "dit")
- `--lora_alpha`: LoRA scaling factor (default: 1.0)
- `--save_precision`: Precision for saving weights - fp32, fp16, or bf16 (default: "bf16")
- `--model_paths`: Custom model paths in JSON format (optional)
- `--model_id_with_origin_paths`: Model ID with origin paths (optional)

### Examples

#### 1. Fuse Lightning LoRA

```bash
python examples/qwen_image/model_training/utils/fuse_lora_weights.py \
  --lora_path "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors" \
  --output_dir "./models/Qwen-Image-Edit-Lightning-Fused" \
  --target_model "dit" \
  --lora_alpha 1.0
```

#### 2. Fuse Custom LoRA with Reduced Alpha

```bash
python examples/qwen_image/model_training/utils/fuse_lora_weights.py \
  --lora_path "./models/my_style_lora.safetensors" \
  --output_dir "./models/Qwen-Image-MyStyle-Fused" \
  --target_model "dit" \
  --lora_alpha 0.7 \
  --save_precision "fp16"
```

## Output

The tool creates the following files in the output directory:
- `transformer.safetensors`: Fused transformer/DIT weights
- `text_encoder.safetensors`: Text encoder weights (unchanged)
- `vae.safetensors`: VAE weights (unchanged)
- `fusion_metadata.json`: Metadata about the fusion process

## Using the Fused Model

### For Training

```bash
accelerate launch examples/qwen_image/model_training/train.py \
  --model_paths '[
    "./models/fused_model/transformer.safetensors",
    "./models/fused_model/text_encoder.safetensors",
    "./models/fused_model/vae.safetensors"
  ]' \
  # ... other training arguments
```

### For Inference

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

model_configs = [
    ModelConfig("./models/fused_model/transformer.safetensors"),
    ModelConfig("./models/fused_model/text_encoder.safetensors"),
    ModelConfig("./models/fused_model/vae.safetensors"),
]

pipe = QwenImagePipeline.from_pretrained(
    model_configs=model_configs,
    torch_dtype=torch.bfloat16,
    device='cuda'
)
```

## Notes

- The fusion is permanent - the LoRA weights are merged into the base weights
- Only the target model component (usually "dit") is modified
- Text encoder and VAE are copied unchanged
- The original model files are not modified
