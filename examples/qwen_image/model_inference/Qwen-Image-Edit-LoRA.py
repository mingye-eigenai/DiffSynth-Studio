from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
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

# Load trained LoRA into the DiT backbone
# Replace this with your LoRA checkpoint path
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
# lora_path = "./models/train/Qwen-Image-Edit-Lightning-Ghost_lora/epoch-4.safetensors"

pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

"""Batch edit a folder of images with the same prompt using LoRA."""

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_ghost_lora_1006")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."
height = 1152
width = 768
num_inference_steps = 40
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

