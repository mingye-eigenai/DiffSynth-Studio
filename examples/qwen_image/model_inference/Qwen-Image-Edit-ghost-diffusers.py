import os 
from PIL import Image 
import torch 
from diffusers import QwenImageEditPipeline 

lora_path = "./models/train/Qwen-Image-Edit-Ghost_lora2/epoch-4.safetensors"

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit") 
print("pipeline loaded") 
# pipeline.to(torch.bfloat16) 




image = Image.open("/home/kuan/workspace/repos/DiffSynth-Studio/outputs_lora_ukiyoe_edit/asian/284802453_5165986493496345_4245860059802904025_n_lora.jpg").convert("RGB")
prompt = "Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(1),
    "true_cfg_scale": 0.0,
    "negative_prompt": " ",
    "num_inference_steps": 30,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit1.png")
    print("image saved at", os.path.abspath("output_image_edit1.png"))

