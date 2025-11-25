"""
Single image inference script for merged QKV LoRA.
Simpler version for quick testing.
"""

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth import load_state_dict
from diffsynth.models import ModelManager
import torch
from PIL import Image
import argparse

def load_pipeline_with_merged_qkv_lora(lora_path, device="cuda", torch_dtype=torch.bfloat16):
    """Load pipeline with merged QKV architecture and LoRA."""
    print("="*60)
    print("Loading Qwen-Image-Edit with Merged QKV LoRA")
    print("="*60)
    
    # Initialize pipeline
    pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
    
    # Load base components
    print("\n1. Loading base components (text encoder, VAE)...")
    text_encoder_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
    vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
    processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
    tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
    
    for config in [text_encoder_config, vae_config, processor_config, tokenizer_config]:
        config.download_if_necessary()
    
    model_manager = ModelManager()
    model_manager.load_model(text_encoder_config.path, device=device, torch_dtype=torch_dtype)
    model_manager.load_model(vae_config.path, device=device, torch_dtype=torch_dtype)
    
    pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
    pipe.vae = model_manager.fetch_model("qwen_image_vae")
    
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor
    pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
    pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
    print("✓ Base components loaded")
    
    # Load DiT with merged QKV
    print("\n2. Loading DiT with merged QKV architecture...")
    dit_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
    dit_config.download_if_necessary()
    
    # Load base weights FIRST
    dit_path = dit_config.path
    if isinstance(dit_path, list):
        print(f"   Loading {len(dit_path)} shards...")
        state_dict = {}
        for shard_path in dit_path:
            state_dict.update(load_state_dict(shard_path))
    else:
        state_dict = load_state_dict(dit_path)
    
    # Create model
    dit_merged = QwenImageDiTMergedQKV(num_layers=60)
    
    # Convert to merged QKV format
    converter = dit_merged.state_dict_converter()
    merged_state_dict = converter.from_diffusers(state_dict)
    
    # CRITICAL: Convert model to target dtype BEFORE loading weights
    dit_merged = dit_merged.to(dtype=torch_dtype, device=device)
    
    # NOW load weights (dtypes must match)
    dit_merged.load_state_dict(merged_state_dict, strict=False)
    
    pipe.dit = dit_merged
    print("✓ DiT loaded")
    
    # Load LoRA
    print(f"\n3. Loading merged QKV LoRA from: {lora_path}")
    # Use alpha=1.0 because PEFT training already applies (lora_alpha/rank) scaling
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    print("✓ LoRA loaded (alpha=1.0)")
    
    print("\n" + "="*60)
    print("Pipeline ready for inference!")
    print("="*60 + "\n")
    
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Single image inference with merged QKV LoRA")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_image", default="output.jpg", help="Path to save output image")
    parser.add_argument("--lora_path", default="./models/train/Qwen-Image-Edit-Ghost_lora_merged_qkv_1012/epoch-4.safetensors", 
                       help="Path to merged QKV LoRA checkpoint")
    parser.add_argument("--prompt", default="Chibi-style 3D cartoon ghost, Pixar/Disney style, smooth and soft, 1.5:1 head-to-body ratio, short rounded ghost arms, floating taper body. Use selfie as reference, preserve unique face structure, hairstyle, and expression, avoid round generic face. Playful, cute, and approachable.", 
                       help="Editing prompt")
    parser.add_argument("--height", type=int, default=1152, help="Output height")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load pipeline
    pipe = load_pipeline_with_merged_qkv_lora(
        lora_path=args.lora_path,
        device=args.device,
        torch_dtype=torch.bfloat16
    )
    
    # Load input image
    print(f"Loading input image: {args.input_image}")
    input_image = Image.open(args.input_image).convert("RGB")
    print(f"✓ Input image loaded: {input_image.size}")
    
    # Run inference
    print(f"\nRunning inference...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")
    print(f"  CFG Scale: {args.cfg_scale}")
    
    output_image = pipe(
        args.prompt,
        edit_image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        edit_image_auto_resize=True,
    )
    
    # Save output
    output_image.save(args.output_image)
    print(f"\n✓ Output saved to: {args.output_image}")
    print("\nDone!")


if __name__ == "__main__":
    main()

