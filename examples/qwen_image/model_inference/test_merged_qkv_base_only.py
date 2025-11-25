"""
Simple test script for merged QKV base model (no LoRA).
Just generates an image to verify the base conversion works.
"""

import torch
from PIL import Image
import argparse
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth import load_state_dict
from diffsynth.models import ModelManager


def main():
    parser = argparse.ArgumentParser(description="Test merged QKV base model (no LoRA)")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_image", default="output_merged_qkv_base.jpg", help="Output image path")
    parser.add_argument("--prompt", default="Chibi-style 3D cartoon ghost, Pixar/Disney style", help="Editing prompt")
    parser.add_argument("--height", type=int, default=1152, help="Output height")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Testing Merged QKV Base Model (NO LoRA)")
    print("="*60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipe = QwenImagePipeline(device=args.device, torch_dtype=torch.bfloat16)
    
    # Load base components
    print("2. Loading base components (text encoder, VAE)...")
    text_encoder_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
    vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
    processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
    tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
    
    for config in [text_encoder_config, vae_config, processor_config, tokenizer_config]:
        config.download_if_necessary()
    
    model_manager = ModelManager()
    model_manager.load_model(text_encoder_config.path, device=args.device, torch_dtype=torch.bfloat16)
    model_manager.load_model(vae_config.path, device=args.device, torch_dtype=torch.bfloat16)
    
    pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
    pipe.vae = model_manager.fetch_model("qwen_image_vae")
    
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor
    pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
    pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
    print("   ✓ Base components loaded")
    
    # Load DiT with merged QKV
    print("\n3. Loading DiT with merged QKV architecture...")
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
    
    print(f"   Loaded {len(state_dict)} parameters")
    
    # Create model
    print("   Creating merged QKV model...")
    dit_merged = QwenImageDiTMergedQKV(num_layers=60)
    
    # Convert to merged QKV format
    print("   Converting weights to merged QKV format...")
    converter = dit_merged.state_dict_converter()
    merged_state_dict = converter.from_diffusers(state_dict)
    print(f"   Converted: {len(merged_state_dict)} parameters")
    
    # CRITICAL: Convert model to bfloat16 BEFORE loading bfloat16 weights
    # This preserves the weights exactly without dtype conversion
    print(f"   Converting model to bfloat16...")
    dit_merged = dit_merged.to(dtype=torch.bfloat16, device="cpu")
    
    # NOW load the bfloat16 weights into bfloat16 model
    print("   Loading weights...")
    missing, unexpected = dit_merged.load_state_dict(merged_state_dict, strict=False)
    
    # Finally move to target device
    if args.device != "cpu":
        print(f"   Moving to {args.device}...")
        dit_merged = dit_merged.to(device=args.device)
    
    if missing:
        print(f"\n   ⚠ Missing keys: {len(missing)}")
        if len(missing) <= 5:
            for key in missing:
                print(f"     - {key}")
    if unexpected:
        print(f"   ⚠ Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 5:
            for key in unexpected:
                print(f"     - {key}")
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("   ✓ All weights loaded successfully!")
    
    pipe.dit = dit_merged
    print("   ✓ Merged QKV DiT ready")
    
    # Load input image
    print(f"\n4. Loading input image: {args.input_image}")
    input_image = Image.open(args.input_image).convert("RGB")
    print(f"   ✓ Input size: {input_image.size}")
    
    # Run inference
    print(f"\n5. Running inference...")
    print(f"   Prompt: {args.prompt[:50]}...")
    print(f"   Size: {args.width}x{args.height}")
    print(f"   Steps: {args.steps}")
    print(f"   Seed: {args.seed}")
    
    output = pipe(
        args.prompt,
        edit_image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        edit_image_auto_resize=True,
    )
    
    # Save output
    output.save(args.output_image)
    print(f"\n✓ Output saved to: {args.output_image}")
    
    # Check if output looks valid (not all black/white/noise)
    import numpy as np
    output_np = np.array(output)
    mean_value = output_np.mean()
    std_value = output_np.std()
    
    print("\n" + "="*60)
    print("OUTPUT VALIDATION")
    print("="*60)
    print(f"Mean pixel value: {mean_value:.2f} (should be 50-200)")
    print(f"Std deviation: {std_value:.2f} (should be > 10)")
    
    if mean_value < 10 or mean_value > 245:
        print("⚠️  WARNING: Image might be all black or all white!")
    elif std_value < 5:
        print("⚠️  WARNING: Image has very low variation (might be blank)!")
    else:
        print("✓ Output looks valid")
    
    print("="*60)
    print("\nDone! Check the output image to verify quality.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

