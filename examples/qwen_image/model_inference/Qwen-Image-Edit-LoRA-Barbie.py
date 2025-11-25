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
# lora_path = "./models/train/Qwen-Image-Edit-Barbie_lora-1003-5/epoch-4.safetensors"
# lora_path = "./models/train/Qwen-Image-Edit-Barbie_lora-1006/epoch-4.safetensors"
# lora_path = "./models/train/Qwen-Image-Edit-Barbie_lora-1006/epoch-4.safetensors"
lora_path = "./models/train/Qwen-Image-Edit-Barbie_lora-1009/epoch-4.safetensors"

# Option 1 (CURRENT): Fused LoRA - merges weights into base model (faster inference)
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

# Option 2: Hotload LoRA - keeps weights separate (allows dynamic LoRA swapping)
# Uncomment these two lines to use hotload instead:
# pipe.enable_lora_magic()  # Wraps Linear layers with AutoWrappedLinear
# pipe.load_lora(pipe.dit, lora_path, alpha=1.0, hotload=True)

# """Batch edit a folder of images with the same prompt using LoRA."""

# Settings
# input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
# output_dir = Path("./outputs_lora_ukiyoe_edit/asian")
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
# output_dir = Path("./outputs_lora_barbie_edit/asian_1003-5_20steps_epoch4")
output_dir = Path("./outputs_lora_barbie_edit/asian_1009_12steps_epoch4")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Barbie style transfer. Primary goal: convert the entire image into a realistic photograph of Barbie dolls inside a Barbie playset, rendering the whole scene as fully 3D with correct depth, perspective, parallax, occlusion and contact shadows—nothing flat or painted-on—and a non-plain background made of real 3D objects that correspond exactly to the original background elements but faithfully transformed into toy versions. Everything should read delightful, beautiful and delicate: convert every person into a realistic plastic Barbie doll with glossy/vinyl skin, smooth highlights, subtle mold seams and toy-like articulation while preserving the original facial expression (default to a gentle, cheerful, delighted smile only if the expression is unclear or occluded), and keep original hair, pose, clothes and accessories precisely. Joints must be visible: always show shoulder joints; for knees, show visible knee joints when legs are exposed but do not show knee joints when pants or other coverings obscure the legs. Transform every foreground and background object into Barbie playset versions with smooth plastic surfaces, clean edges, subtle mold lines and a consistent toy scale, using a pastel Barbie palette (pink, lavender, aqua blue, mint green) with bright, high-key photographic lighting and soft, clean shadows. Maintain the original framing, layout and scene composition, strictly keep only items present in the source image (no added elements, characters, props, text, logos, stickers or decorations), and deliver an overall polished product-photo realism of a Barbie diorama—bright, playful, clean and highly toy-like—while faithfully retaining and rendering every original scene element as convincingly 3D."
num_inference_steps = 12
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

