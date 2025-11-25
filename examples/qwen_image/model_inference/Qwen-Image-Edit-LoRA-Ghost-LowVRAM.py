from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
import torch
from PIL import Image
from pathlib import Path

# Initialize pipeline WITHOUT quantization first for faster LoRA loading
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

# # First, fuse Lightning LoRA into base model
# lightning_lora_path = "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
# print(f"Fusing Lightning LoRA from: {lightning_lora_path}")
# lightning_lora_state_dict = load_state_dict(lightning_lora_path)
# lightning_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
# num_updated = lightning_loader.load(pipe.dit, lightning_lora_state_dict, alpha=1.0)
# print(f"Lightning LoRA fused: {num_updated} tensors updated")

# Load custom LoRA
lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"
print(f"Loading custom LoRA from: {lora_path}")
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

# NOW enable aggressive VRAM management AFTER LoRA is loaded
print("Enabling low VRAM mode with FP8 quantization...")
pipe.enable_vram_management(
    vram_limit=8.0,  # Limit to 8GB VRAM (adjust based on your GPU)
    vram_buffer=1.0,  # Keep 1GB buffer for safety
    auto_offload=True,  # Enable automatic offloading
    enable_dit_fp8_computation=True  # Enable FP8 computation
)

"""Low VRAM batch inference with aggressive quantization and offloading."""

# Settings
input_dir = Path("/home/kuan/workspace/repos/avatar/fg_special/asian")
output_dir = Path("./outputs_lora_ghost_edit/asian_lightning_lowvram")
output_dir.mkdir(parents=True, exist_ok=True)

edit_prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."

# Reduced resolution for lower VRAM usage
height = 768  # Reduced from 1152
width = 512   # Reduced from 768
num_inference_steps = 40
seed = 1
edit_image_auto_resize = True
enable_fp8_attention = True  # Enable FP8 attention

# Process images with reduced batch size
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

print(f"\nRunning LOW VRAM inference with FP8 quantization and offloading")
print(f"VRAM limit: 8GB")
print(f"Resolution: {width}x{height}")
print(f"Output directory: {output_dir}")

for img_path in sorted(input_dir.rglob("*")):
    if img_path.suffix.lower() not in extensions:
        continue
    
    try:
        input_image = Image.open(img_path).convert("RGB")
        
        # Optional: Resize input image to reduce memory usage
        if input_image.width > 1024 or input_image.height > 1024:
            input_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
    except Exception as e:
        print(f"Skip {img_path}: {e}")
        continue

    # Generate image
    image = pipe(
        edit_prompt,
        edit_image=input_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=edit_image_auto_resize,
        enable_fp8_attention=enable_fp8_attention,
    )
    
    out_path = output_dir / f"{img_path.stem}_lora_lowvram.jpg"
    image.save(str(out_path))
    print(f"Saved: {out_path}")
    
    # Optional: Clear CUDA cache after each image to prevent OOM
    torch.cuda.empty_cache()

print("\nDone! Low VRAM mode should work on GPUs with 8GB or less VRAM.")
