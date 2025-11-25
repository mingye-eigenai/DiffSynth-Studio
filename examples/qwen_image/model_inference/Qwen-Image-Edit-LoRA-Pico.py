from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
import torch
from PIL import Image
from pathlib import Path

# Initialize pipeline (base Qwen-Image weights)
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# First, fuse Lightning LoRA into base model (since our LoRA was trained on top of Lightning)
# lightning_lora_path = "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
# print(f"Fusing Lightning LoRA from: {lightning_lora_path}")
# lightning_lora_state_dict = load_state_dict(lightning_lora_path)
# lightning_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
# num_updated = lightning_loader.load(pipe.dit, lightning_lora_state_dict, alpha=1.0)
# print(f"Lightning LoRA fused: {num_updated} tensors updated")

print(pipe.dit)
# Then load our trained LoRA on top of the Lightning-fused model
# Replace this with your LoRA checkpoint path
# lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
lora_path = "./models/train/Qwen-Image-Edit-Pico_lora_merged_qkv_1026/step-60000.safetensors"

# lora_path = "./models/train/Qwen-Image-Edit-Lightning-Ghost_lora/epoch-4.safetensors"
# lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora_1010/epoch-4.safetensors"
print(f"Loading custom LoRA from: {lora_path}")
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

"""Batch edit a folder of images with the same prompt using LoRA."""

# Settings
# input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
input_dir = Path("/data/kuan/datasets/avatar_images/fg_im_input")
# output_dir = Path("./outputs_lora_ghost_edit/asian_lightning_fusion")
output_dir = Path("./outputs_lora_pico_edit/fg_im_pico_lora_merged_qkv_1026_20steps_seed1")
output_dir.mkdir(parents=True, exist_ok=True)

# edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."
edit_prompt = "Please remove the hair from the person's head."

height = 1152
width = 768
num_inference_steps = 20
seed = 1
edit_image_auto_resize = True  # keep True to auto-fit different aspect ratios

# Supported image extensions
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

for img_path in sorted(input_dir.rglob("*")):
    if img_path.suffix.lower() not in extensions:
        continue
    try:
        input_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue

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
    print(f"Saved: {out_path}")

