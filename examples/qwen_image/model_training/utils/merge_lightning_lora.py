import torch
import os
from safetensors.torch import save_file
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.lora import GeneralLoRALoader

def merge_lightning_lora():
    # Load base Qwen-Image-Edit pipeline
    print("Loading base Qwen-Image-Edit model...")
    model_configs = [
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ]
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, 
        device="cpu",
        model_configs=model_configs,
    )
    
    # Load Lightning LoRA weights
    print("Loading Lightning LoRA weights...")
    lora_path = "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
    lora_state_dict = load_state_dict(lora_path)
    
    # Merge LoRA into base model
    print("Merging LoRA weights into base model...")
    lora_loader = GeneralLoRALoader(torch_dtype=pipe.torch_dtype, device="cpu")
    lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)
    
    # Save merged transformer weights
    output_dir = "./models/Qwen-Image-Edit-Lightning-merged"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving merged model to {output_dir}...")
    # Save transformer (dit) weights
    transformer_path = os.path.join(output_dir, "transformer.safetensors")
    save_file(pipe.dit.state_dict(), transformer_path)
    
    # Copy text encoder and VAE (they don't change)
    print("Copying text encoder and VAE...")
    text_encoder_path = os.path.join(output_dir, "text_encoder.safetensors")
    save_file(pipe.text_encoder.state_dict(), text_encoder_path)
    
    vae_path = os.path.join(output_dir, "vae.safetensors")
    save_file(pipe.vae.state_dict(), vae_path)
    
    print(f"Done! Merged model saved to {output_dir}")
    print("\nTo use the merged model in training, update your script with:")
    print(f'  --model_paths \'["{transformer_path}", "{text_encoder_path}", "{vae_path}"]\' \\')
    print("  # Remove --lora_checkpoint line")

if __name__ == "__main__":
    merge_lightning_lora()
