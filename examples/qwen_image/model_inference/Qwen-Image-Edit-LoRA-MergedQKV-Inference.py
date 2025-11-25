"""
Inference script for Qwen-Image-Edit with merged QKV LoRA.

This script demonstrates how to use a model trained with merged QKV architecture.
The merged QKV model is more efficient (fewer kernel calls) while being mathematically
equivalent to the standard architecture.

Usage:
    python Qwen-Image-Edit-LoRA-MergedQKV-Inference.py \
        --input_image path/to/input.png \
        --edit_image path/to/edit.png \
        --prompt "your prompt here" \
        --output path/to/output.png \
        --lora_path path/to/merged_qkv_lora.safetensors
"""

import argparse
from PIL import Image
import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth.models import ModelManager, load_state_dict


def load_pipeline_with_merged_qkv(device="cuda", torch_dtype=torch.bfloat16):
    """
    Load QwenImagePipeline with merged QKV DiT architecture.
    """
    # Load standard components
    pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
    
    # Load models from HuggingFace
    text_encoder_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
    vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
    dit_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
    
    # Download and load
    text_encoder_config.download_if_necessary()
    vae_config.download_if_necessary()
    dit_config.download_if_necessary()
    
    model_manager = ModelManager()
    model_manager.load_model(text_encoder_config.path, device=device, torch_dtype=torch_dtype)
    model_manager.load_model(vae_config.path, device=device, torch_dtype=torch_dtype)
    
    pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
    pipe.vae = model_manager.fetch_model("qwen_image_vae")
    
    # Load tokenizer and processor
    tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
    processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
    tokenizer_config.download_if_necessary()
    processor_config.download_if_necessary()
    
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor
    pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
    pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
    
    # Load DiT with merged QKV architecture
    print("Loading DiT with merged QKV architecture...")
    dit_merged = QwenImageDiTMergedQKV(num_layers=60)
    dit_merged = dit_merged.to(dtype=torch_dtype, device=device)
    
    # Load and convert pretrained weights
    state_dict = load_state_dict(dit_config.path)
    converter = dit_merged.state_dict_converter()
    merged_state_dict = converter.from_diffusers(state_dict)
    
    missing, unexpected = dit_merged.load_state_dict(merged_state_dict, strict=False)
    print(f"Loaded DiT with {len(merged_state_dict)} parameters")
    
    pipe.dit = dit_merged
    
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Inference with merged QKV Qwen-Image-Edit")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--edit_image", required=True, help="Path to edit/reference image")
    parser.add_argument("--prompt", required=True, help="Text prompt for editing")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--lora_path", default=None, help="Path to merged QKV LoRA checkpoint (optional)")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha/scale")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    
    args = parser.parse_args()
    
    # Setup
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]
    
    # Load pipeline
    print("Loading pipeline with merged QKV architecture...")
    pipe = load_pipeline_with_merged_qkv(device=args.device, torch_dtype=torch_dtype)
    
    # Load LoRA if provided
    if args.lora_path:
        print(f"Loading merged QKV LoRA from: {args.lora_path}")
        lora_state_dict = load_state_dict(args.lora_path)
        
        # Check if LoRA contains merged QKV modules
        has_merged_qkv = any("to_qkv" in k or "add_qkv_proj" in k for k in lora_state_dict.keys())
        if has_merged_qkv:
            print("✓ Detected merged QKV LoRA format")
        else:
            print("⚠ Warning: LoRA does not appear to be in merged QKV format")
            print("  If training was done with standard format, consider converting it first.")
        
        from diffsynth.lora import GeneralLoRALoader
        lora_loader = GeneralLoRALoader(torch_dtype=torch_dtype, device=args.device)
        lora_loader.load(pipe.dit, lora_state_dict, alpha=args.lora_alpha)
        print(f"LoRA loaded with alpha={args.lora_alpha}")
    
    # Load images
    print("Loading images...")
    input_image = Image.open(args.input_image).convert("RGB")
    edit_image = Image.open(args.edit_image).convert("RGB")
    
    # Run inference
    print("Running inference...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.num_steps}")
    print(f"  CFG Scale: {args.cfg_scale}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Seed: {args.seed}")
    
    output_image = pipe(
        prompt=args.prompt,
        input_image=input_image,
        edit_image=edit_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    
    # Save output
    output_image.save(args.output)
    print(f"✓ Output saved to: {args.output}")


if __name__ == "__main__":
    main()

