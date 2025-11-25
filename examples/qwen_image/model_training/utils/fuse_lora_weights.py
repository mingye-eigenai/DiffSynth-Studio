#!/usr/bin/env python3
"""
General-purpose tool to fuse LoRA weights into base model and save the result.
Supports both standard LoRA format and Lightning LoRA format.
"""

import torch
import os
import argparse
import json
from safetensors.torch import save_file
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.lora import GeneralLoRALoader
from diffsynth.lora.lightning_lora_loader import LightningLoRALoader


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse LoRA weights into base model")
    
    # Model configuration
    parser.add_argument("--model_paths", type=str, default=None, 
                        help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g., Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors")
    
    # LoRA configuration
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to the LoRA weights to fuse")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA alpha scaling factor (default: 1.0)")
    parser.add_argument("--target_model", type=str, default="dit",
                        help="Which model component to apply LoRA to (default: dit)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fused model")
    parser.add_argument("--save_precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision for saving weights (default: bf16)")
    
    return parser.parse_args()


def get_torch_dtype(precision):
    """Convert precision string to torch dtype"""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map[precision]


def fuse_lora_weights():
    args = parse_args()
    
    # Parse model configs
    if args.model_paths:
        model_configs = json.loads(args.model_paths)
        model_configs = [ModelConfig(path) for path in model_configs]
    elif args.model_id_with_origin_paths:
        # Default to Qwen-Image-Edit if not specified
        parts = args.model_id_with_origin_paths.split(":")
        model_id = parts[0]
        origin_pattern = parts[1] if len(parts) > 1 else "transformer/diffusion_pytorch_model*.safetensors"
        model_configs = [ModelConfig(model_id=model_id, origin_file_pattern=origin_pattern)]
    else:
        # Default configuration for Qwen-Image-Edit
        print("Using default Qwen-Image-Edit configuration")
        model_configs = [
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
    
    # Load base model
    print("Loading base model...")
    torch_dtype = get_torch_dtype(args.save_precision)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device="cpu",
        model_configs=model_configs,
    )
    
    # Load LoRA weights
    print(f"\nLoading LoRA from: {args.lora_path}")
    lora_state_dict = load_state_dict(args.lora_path)
    print(f"LoRA contains {len(lora_state_dict)} keys")
    
    # Detect LoRA format
    is_lightning_lora = any("lora_down" in k or "lora_up" in k for k in lora_state_dict.keys())
    
    if is_lightning_lora:
        print("Detected Lightning LoRA format")
        lora_loader = LightningLoRALoader(torch_dtype=torch_dtype, device="cpu")
    else:
        print("Detected standard LoRA format")
        lora_loader = GeneralLoRALoader(torch_dtype=torch_dtype, device="cpu")
    
    # Apply LoRA to target model
    print(f"\nFusing LoRA into {args.target_model} with alpha={args.lora_alpha}...")
    target_module = getattr(pipe, args.target_model)
    num_updated = lora_loader.load(target_module, lora_state_dict, alpha=args.lora_alpha)
    
    if num_updated == 0:
        print("WARNING: No tensors were updated! Check if:")
        print("  - The LoRA was trained for this model architecture")
        print("  - The --target_model parameter is correct")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the fused models
    print(f"\nSaving fused model to {args.output_dir}...")
    
    # Save each component
    saved_files = []
    
    # Save transformer/DIT
    if hasattr(pipe, 'dit'):
        dit_path = os.path.join(args.output_dir, "transformer.safetensors")
        save_file(pipe.dit.state_dict(), dit_path)
        saved_files.append(("transformer", dit_path))
        print(f"  Saved transformer to: {dit_path}")
    
    # Save text encoder
    if hasattr(pipe, 'text_encoder'):
        text_encoder_path = os.path.join(args.output_dir, "text_encoder.safetensors")
        save_file(pipe.text_encoder.state_dict(), text_encoder_path)
        saved_files.append(("text_encoder", text_encoder_path))
        print(f"  Saved text encoder to: {text_encoder_path}")
    
    # Save VAE
    if hasattr(pipe, 'vae'):
        vae_path = os.path.join(args.output_dir, "vae.safetensors")
        save_file(pipe.vae.state_dict(), vae_path)
        saved_files.append(("vae", vae_path))
        print(f"  Saved VAE to: {vae_path}")
    
    # Save metadata
    metadata = {
        "base_model": args.model_id_with_origin_paths or "Qwen/Qwen-Image-Edit",
        "lora_path": args.lora_path,
        "lora_alpha": args.lora_alpha,
        "target_model": args.target_model,
        "num_updated_tensors": num_updated,
        "save_precision": args.save_precision,
        "is_lightning_lora": is_lightning_lora,
        "saved_files": saved_files
    }
    
    metadata_path = os.path.join(args.output_dir, "fusion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to: {metadata_path}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("Fusion complete! To use the fused model:")
    print("\n1. For training with the fused weights:")
    print("```bash")
    print("accelerate launch examples/qwen_image/model_training/train.py \\")
    print(f'  --model_paths \'[')
    for component, path in saved_files:
        print(f'    "{path}",')
    print("  ]' \\")
    print("  # ... other training arguments")
    print("```")
    
    print("\n2. For inference:")
    print("```python")
    print("from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig")
    print("model_configs = [")
    for component, path in saved_files:
        print(f'    ModelConfig("{path}"),')
    print("]")
    print("pipe = QwenImagePipeline.from_pretrained(")
    print("    model_configs=model_configs,")
    print("    torch_dtype=torch.bfloat16,")
    print("    device='cuda'")
    print(")")
    print("```")
    print("="*60)


if __name__ == "__main__":
    fuse_lora_weights()
