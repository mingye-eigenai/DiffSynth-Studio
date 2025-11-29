# SkyPilot Training Quick Guide

Train Qwen-Image-Edit LoRA models with your own dataset using SkyPilot.

## Directory Structure

### Base Models (Shared)

Base models are stored in `/shared/base_models/${MODEL_FAMILY}/${MODEL_NAME}/` with a two-level hierarchy:

```
/shared/base_models/
└── Qwen/                          # Model family
    ├── Qwen-Image-Edit/           # Default edit model
    │   ├── transformer/
    │   └── processor/
    ├── Qwen-Image-Edit-2509/      # Alternative edit model (if available)
    │   ├── transformer/
    │   └── processor/
    └── Qwen-Image/                # Shared components (used by all edit models)
        ├── text_encoder/
        ├── vae/
        └── tokenizer/
```

The script will automatically download models if they don't exist.

**Note**: If you already have models at the old location, use the helper script to migrate them:

```bash
bash setup_base_models.sh
```

Or manually copy:
```bash
mkdir -p /shared/base_models/Qwen
cp -r /shared/mingye/DiffSynth-Studio/examples/qwen_image/model_training/lora/models/Qwen/* /shared/base_models/Qwen/
```

### Dataset Setup

Each user has their own dataset folder under `/shared/dataset/`:

```
/shared/dataset/
├── alice_ghost/          # Alice's ghost style dataset
│   ├── images/           # Original images
│   ├── edit_images/      # Edited images (same filenames as images/)
│   └── prompts.txt       # Optional: one prompt per line
├── bob_pico/             # Bob's pico style dataset
│   ├── images/
│   ├── edit_images/
│   └── prompts.txt
└── charlie_ukiyoe/       # Charlie's ukiyoe style dataset
    ├── images/
    ├── edit_images/
    └── prompts.txt
```

**Important**:
- Image filenames must match between `images/` and `edit_images/`
- Supported formats: `.png`, `.jpg`, `.jpeg`
- `prompts.txt` is optional (one prompt per line)

## Launch Training

### Basic Training

```bash
sky launch -c my-training --env DATASET_NAME=alice_ghost skypilot.yaml
```

### Custom Parameters

```bash
sky launch -c my-training \
  --env DATASET_NAME=bob_pico \
  --env NUM_EPOCHS=10 \
  --env LEARNING_RATE=5e-5 \
  --env LORA_RANK=64 \
  skypilot.yaml
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_NAME` | `default` | **Required**: Your unique dataset folder name |
| `MODEL_FAMILY` | `Qwen` | Model family (top-level folder in `/shared/base_models/`) |
| `MODEL_NAME` | `Qwen-Image-Edit` | Specific model to train with: `Qwen-Image-Edit`, `Qwen-Image-Edit-2509`, etc. |
| `NUM_EPOCHS` | `5` | Number of training epochs |
| `LEARNING_RATE` | `1e-4` | Learning rate |
| `LORA_RANK` | `32` | LoRA rank (8, 16, 32, 64, 128) |
| `SAVE_STEPS` | `3000` | Save checkpoint every N steps |

## Examples

### Quick test (3 epochs, rank 16)
```bash
sky launch -c test \
  --env DATASET_NAME=alice_ghost \
  --env NUM_EPOCHS=3 \
  --env LORA_RANK=16 \
  skypilot.yaml
```

### Use newer model (Qwen-Image-Edit-2509)
```bash
sky launch -c newer-model \
  --env DATASET_NAME=bob_pico \
  --env MODEL_NAME=Qwen-Image-Edit-2509 \
  --env NUM_EPOCHS=5 \
  skypilot.yaml
```

### Production training (10 epochs, rank 64)
```bash
sky launch -c prod \
  --env DATASET_NAME=bob_pico \
  --env NUM_EPOCHS=10 \
  --env LEARNING_RATE=5e-5 \
  --env LORA_RANK=64 \
  skypilot.yaml
```

### High-rank training with specific model
```bash
sky launch -c high-rank \
  --env DATASET_NAME=charlie_ukiyoe \
  --env MODEL_NAME=Qwen-Image-Edit-2509 \
  --env LORA_RANK=128 \
  skypilot.yaml
```

## Monitor Training

```bash
# Check status
sky status my-training

# View logs
sky logs my-training --follow

# SSH into cluster
sky ssh my-training
```

## Outputs

Results are saved to `/shared/output/{DATASET_NAME}_{MODEL_NAME}_lora_rank{RANK}_{EPOCHS}epochs/`

Examples:
- `/shared/output/alice_ghost_Qwen-Image-Edit_lora_rank32_5epochs/`
- `/shared/output/bob_pico_Qwen-Image-Edit-2509_lora_rank64_10epochs/`

Download results:
```bash
sky down my-training:/shared/output ./local_output
```

## Stop/Delete

```bash
# Stop cluster
sky stop my-training

# Delete cluster
sky down my-training
```

## Multi-Node Training

Edit `num_nodes` in `skypilot.yaml`:

```yaml
num_nodes: 2  # or 4, 8
```

Then launch:
```bash
sky launch -c multi-node --env DATASET_NAME=alice_ghost skypilot.yaml
```

## Tips

- **Out of memory?** Lower `LORA_RANK` to 16 or 8
- **Training too slow?** Increase learning rate slightly
- **Need faster convergence?** Try learning rate 2e-4 with more epochs
- **Multiple datasets?** Each user can have multiple dataset folders

## Complete Workflow Example

```bash
# 1. Prepare your dataset
mkdir -p /shared/dataset/myname_ghost/images
mkdir -p /shared/dataset/myname_ghost/edit_images
# Copy your images...
echo "Convert to ghost illustration style" > /shared/dataset/myname_ghost/prompts.txt

# 2. Launch training
sky launch -c ghost-training \
  --env DATASET_NAME=myname_ghost \
  --env NUM_EPOCHS=5 \
  --env LORA_RANK=32 \
  skypilot.yaml

# 3. Monitor
sky logs ghost-training --follow

# 4. Download results
sky down ghost-training:/shared/output/myname_ghost_lora_rank32_5epochs ./my_lora

# 5. Clean up
sky stop ghost-training
```
