#!/usr/bin/env python3
"""
Compare multiple checkpoints on the same test cases
Usage: python compare_checkpoints.py --checkpoint_dir ./models/train/Qwen-Image-Edit-Pico_8nodes/
"""

import argparse
import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import os
import glob
from safetensors.torch import load_file

def load_checkpoint(pipe, checkpoint_path):
    """Load trained checkpoint into the pipeline"""
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("pipe.dit.", "")
        cleaned_state_dict[new_key] = value
    
    pipe.dit.load_state_dict(cleaned_state_dict, strict=False)
    return pipe


def compare_checkpoints(checkpoint_dir, output_dir="./checkpoint_comparison", device="cuda"):
    """Compare all checkpoints in a directory"""
    
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "step-*.safetensors")))
    
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    print("="*60)
    print("Checkpoint Comparison")
    print("="*60)
    print(f"Found {len(checkpoint_files)} checkpoints:")
    for ckpt in checkpoint_files:
        print(f"  - {os.path.basename(ckpt)}")
    print("")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test prompt
    test_prompt_gen = "一个女孩站在花园里，阳光明媚，写实风格"
    test_prompt_edit = "将背景改为夜晚星空，保持人物不变"
    
    print("Test case:")
    print(f"  Generation: {test_prompt_gen}")
    print(f"  Edit: {test_prompt_edit}")
    print("")
    
    # Load base pipeline once
    print("Loading base pipeline...")
    base_pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    # Generate input image (same for all checkpoints)
    print("Generating input image...")
    input_image = base_pipe(prompt=test_prompt_gen, seed=42, num_inference_steps=40, height=1024, width=1024)
    input_image.save(os.path.join(output_dir, "input.jpg"))
    print("✓ Input saved\n")
    
    # Test each checkpoint
    for i, ckpt_path in enumerate(checkpoint_files):
        ckpt_name = os.path.basename(ckpt_path).replace(".safetensors", "")
        print(f"[{i+1}/{len(checkpoint_files)}] Testing {ckpt_name}...")
        
        # Reload pipeline for each checkpoint
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        )
        
        pipe = load_checkpoint(pipe, ckpt_path)
        
        # Run edit
        edited_image = pipe(
            test_prompt_edit, 
            edit_image=input_image, 
            seed=42, 
            num_inference_steps=40,
            height=1024, 
            width=1024, 
            edit_image_auto_resize=True
        )
        
        output_path = os.path.join(output_dir, f"{ckpt_name}_output.jpg")
        edited_image.save(output_path)
        print(f"  ✓ Saved to {output_path}\n")
    
    print("="*60)
    print("✅ Comparison complete!")
    print(f"Check outputs in: {output_dir}")
    print("\nTo view results, open the directory and compare:")
    print("  - input.jpg (original)")
    for ckpt in checkpoint_files:
        ckpt_name = os.path.basename(ckpt).replace(".safetensors", "")
        print(f"  - {ckpt_name}_output.jpg")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple Qwen-Image-Edit checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoint_comparison",
        help="Directory to save comparison outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ Error: Directory not found: {args.checkpoint_dir}")
        exit(1)
    
    compare_checkpoints(args.checkpoint_dir, args.output_dir, args.device)

