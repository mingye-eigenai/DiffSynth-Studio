from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth import load_state_dict
from diffsynth.models import ModelManager
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
import torch
from PIL import Image
from pathlib import Path

# Initialize pipeline with merged QKV DiT
print("Initializing pipeline with merged QKV architecture...")

# Step 1: Load base components (text encoder, VAE, tokenizer, processor)
pipe = QwenImagePipeline(device="cuda", torch_dtype=torch.bfloat16)

# Load text encoder and VAE
text_encoder_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")

# Download and load
text_encoder_config.download_if_necessary()
vae_config.download_if_necessary()
processor_config.download_if_necessary()
tokenizer_config.download_if_necessary()

model_manager = ModelManager()
model_manager.load_model(text_encoder_config.path, device="cuda", torch_dtype=torch.bfloat16)
model_manager.load_model(vae_config.path, device="cuda", torch_dtype=torch.bfloat16)

pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
pipe.vae = model_manager.fetch_model("qwen_image_vae")

# Load tokenizer and processor
from transformers import Qwen2Tokenizer, Qwen2VLProcessor
pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)

# Step 2: Load DiT with merged QKV architecture
print("Loading DiT with merged QKV architecture...")
dit_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
dit_config.download_if_necessary()

# Load and convert pretrained weights FIRST
dit_path = dit_config.path
if isinstance(dit_path, list):
    # Sharded model
    print(f"Loading sharded DiT from {len(dit_path)} files...")
    state_dict = {}
    for shard_path in dit_path:
        shard_dict = load_state_dict(shard_path)
        state_dict.update(shard_dict)
else:
    state_dict = load_state_dict(dit_path)

print(f"Loaded {len(state_dict)} parameters from base DiT")

# Create merged QKV DiT
dit_merged = QwenImageDiTMergedQKV(num_layers=60)

# Convert to merged QKV format
converter = dit_merged.state_dict_converter()
merged_state_dict = converter.from_diffusers(state_dict)
print(f"Converted to merged QKV format: {len(merged_state_dict)} parameters")

# CRITICAL: Convert model to bfloat16 BEFORE loading bfloat16 weights
dit_merged = dit_merged.to(dtype=torch.bfloat16, device="cuda")

# NOW load the bfloat16 weights into bfloat16 model
dit_merged.load_state_dict(merged_state_dict, strict=False)

pipe.dit = dit_merged
print("✓ Base DiT with merged QKV loaded successfully")

# Step 3: (Optional) Fuse Lightning LoRA into base model if needed
# Uncomment if your LoRA was trained on top of Lightning LoRA
# lightning_lora_path = "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
# print(f"Fusing Lightning LoRA from: {lightning_lora_path}")
# lightning_lora_state_dict = load_state_dict(lightning_lora_path)
# lightning_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
# 
# # Convert Lightning LoRA to merged QKV format if needed
# lightning_converter = dit_merged.state_dict_converter()
# lightning_merged = lightning_converter.from_diffusers(lightning_lora_state_dict)
# 
# num_updated = lightning_loader.load(pipe.dit, lightning_merged, alpha=1.0)
# print(f"Lightning LoRA fused: {num_updated} tensors updated")

print(pipe.dit)

# Step 4: Load your trained merged QKV LoRA
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora_merged_qkv_1013/epoch-4.safetensors"
# lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"Loading merged QKV LoRA from: {lora_path}")

# Load LoRA - it's already in merged QKV format
# Use alpha=1.0 because the LoRA was trained with PEFT which already applies (alpha/rank) scaling
# During training: lora_alpha=32, rank=32, so scale = 32/32 = 1.0
# The saved LoRA weights are already properly scaled for alpha=1.0 inference
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
print("✓ Merged QKV LoRA loaded successfully (alpha=1.0)")

"""Batch edit a folder of images with the same prompt using merged QKV LoRA."""

# Settings
input_dir = Path("/data/kuan/datasets/avatar_images/fg_im_input")
# output_dir = Path("./outputs_lora_ghost_edit/fg_im_ghost_lora_merged_qkv_1013_epoch4_alpha32_20steps_seed3")
output_dir = Path("./outputs_lora_ghost_edit/fg_im_ghost_lora_merged_qkv_1013_epoch4_20steps_seed4")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."
height = 1152
width = 768
num_inference_steps = 20
seed = 4
edit_image_auto_resize = True  # keep True to auto-fit different aspect ratios

# Supported image extensions
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\n{'='*60}")
print(f"Starting batch inference")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Prompt: {edit_prompt}")
print(f"Settings: {width}x{height}, {num_inference_steps} steps, seed={seed}")
print(f"{'='*60}\n")

for img_path in sorted(input_dir.rglob("*")):
    if img_path.suffix.lower() not in extensions:
        continue
    try:
        input_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue

    print(f"Processing: {img_path.name}...")
    image = pipe(
        edit_prompt,
        edit_image=input_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=edit_image_auto_resize,
    )
    out_path = output_dir / f"{img_path.stem}_lora.jpg"
    image.save(str(out_path))
    print(f"✓ Saved: {out_path}")

print(f"\n{'='*60}")
print(f"Batch inference complete!")
print(f"Processed images saved to: {output_dir}")
print(f"{'='*60}\n")

