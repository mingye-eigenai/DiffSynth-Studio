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
# lora_path = "./models/train/Qwen-Image-Edit-Ukiyoe_lora/epoch-4.safetensors"
lora_path = "./models/train/Qwen-Image-Edit-Ukiyoe_lora-1009/epoch-4.safetensors"
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

"""Batch edit a folder of images with the same prompt using LoRA."""

# Settings
# input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
# output_dir = Path("./outputs_lora_ukiyoe_edit/asian")
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/phoebe")
output_dir = Path("./outputs_lora_ukiyoe_edit/phoebe4_1009_40steps")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Traditional Edo-period Japanese Ukiyo-e woodblock print. Bold, even sumi-ink outlines; simplified anatomy and facial features. Flat color blocks (little to no soft shading), subtle bokashi only for sky/water mist. Limited palette: Prussian blue/indigo, vermilion, ochre, muted greens, warm washi paper beige. Visible washi/woodgrain texture; slight registration misalignment (hand-printed look). Keep the background, but simplify it. Optionaly addd kento registration marks, a small red artist seal. Optional vertical title cartouche with 2–3 kanji. No modern photographic glare, no glossy reflections, no lens effects. Negative / Never include: photographic realism, smooth gradients/HDR, lens flare, glossy highlights, 3D render/plastic shine, neon palette, bokeh, detailed skin pores"
num_inference_steps = 40
seed = 3
edit_image_auto_resize = False  # keep False to preserve input resolution

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

    # Auto-adjust output size to match the input image (rounded to valid multiples)
    input_width, input_height = input_image.size
    target_height, target_width = pipe.check_resize_height_width(input_height, input_width)
    if (target_width, target_height) != (input_width, input_height):
        input_image = input_image.resize((target_width, target_height), Image.BICUBIC)

    image = pipe(
        edit_prompt,
        edit_image=input_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=target_height,
        width=target_width,
        edit_image_auto_resize=edit_image_auto_resize,
    )
    out_path = output_dir / f"{img_path.stem}_lora.jpg"
    image.save(str(out_path))
    print(f"Saved: {out_path}")

