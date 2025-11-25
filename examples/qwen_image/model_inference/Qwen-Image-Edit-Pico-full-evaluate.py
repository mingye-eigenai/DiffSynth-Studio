#!/usr/bin/env python3
"""
Evaluate trained Qwen-Image-Edit checkpoint
Usage: python evaluate_checkpoint.py --checkpoint step-5000.safetensors
"""

import argparse
import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import os
from safetensors.torch import load_file

def load_checkpoint(pipe, checkpoint_path):
    """Load trained checkpoint into the pipeline"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint state dict (using safetensors for .safetensors files)
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Remove the prefix if it exists (from training config: remove_prefix_in_ckpt "pipe.dit.")
    # The checkpoint might have keys like "pipe.dit.xxx" or just "xxx"
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove "pipe.dit." prefix if present
        new_key = key.replace("pipe.dit.", "")
        cleaned_state_dict[new_key] = value
    
    # Load into the DIT model
    missing, unexpected = pipe.dit.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing:
        print(f"⚠️  Missing keys: {len(missing)}")
        if len(missing) < 10:
            for key in missing:
                print(f"  - {key}")
    
    if unexpected:
        print(f"⚠️  Unexpected keys: {len(unexpected)}")
        if len(unexpected) < 10:
            for key in unexpected:
                print(f"  - {key}")
    
    print("✅ Checkpoint loaded successfully!")
    return pipe


def run_evaluation(checkpoint_path, output_dir="./eval_outputs", device="cuda"):
    """Run image editing evaluation with the checkpoint"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Qwen-Image-Edit Checkpoint Evaluation")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print("")
    
    # Load base pipeline
    print("Loading base pipeline...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            # Use base model first, then load checkpoint
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    # Load trained checkpoint
    pipe = load_checkpoint(pipe, checkpoint_path)
    
    print("\n" + "="*60)
    print("Running evaluation test cases...")
    print("="*60 + "\n")
    
    # Test Case 1: Original example from inference script
    print("Test 1: Underwater girl with dress color change")
    prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
    input_image = pipe(prompt=prompt, seed=0, num_inference_steps=40, height=1328, width=1024)
    input_image.save(os.path.join(output_dir, "test1_input.jpg"))
    print(f"  ✓ Input image saved")
    
    prompt = "将裙子改为粉色"  # Change dress to pink
    edited_image = pipe(prompt, edit_image=input_image, seed=1, num_inference_steps=40, 
                       height=1328, width=1024, edit_image_auto_resize=True)
    edited_image.save(os.path.join(output_dir, "test1_output.jpg"))
    print(f"  ✓ Edited image saved")
    
    # Test Case 2: Simple color change
    print("\nTest 2: Portrait with background change")
    prompt = "一个女孩站在花园里，阳光明媚"  # A girl standing in a garden, sunny
    input_image = pipe(prompt=prompt, seed=42, num_inference_steps=40, height=1024, width=1024)
    input_image.save(os.path.join(output_dir, "test2_input.jpg"))
    print(f"  ✓ Input image saved")
    
    prompt = "将背景改为夜晚星空"  # Change background to night starry sky
    edited_image = pipe(prompt, edit_image=input_image, seed=42, num_inference_steps=40,
                       height=1024, width=1024, edit_image_auto_resize=True)
    edited_image.save(os.path.join(output_dir, "test2_output.jpg"))
    print(f"  ✓ Edited image saved")
    
    # Test Case 3: Style transfer
    print("\nTest 3: Cat with style change")
    prompt = "一只可爱的小猫坐在窗台上，写实风格"  # A cute cat sitting on windowsill, realistic style
    input_image = pipe(prompt=prompt, seed=123, num_inference_steps=40, height=1024, width=1024)
    input_image.save(os.path.join(output_dir, "test3_input.jpg"))
    print(f"  ✓ Input image saved")
    
    prompt = "改为水彩画风格"  # Change to watercolor painting style
    edited_image = pipe(prompt, edit_image=input_image, seed=123, num_inference_steps=40,
                       height=1024, width=1024, edit_image_auto_resize=True)
    edited_image.save(os.path.join(output_dir, "test3_output.jpg"))
    print(f"  ✓ Edited image saved")
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print(f"Check outputs in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen-Image-Edit checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., ./models/train/Qwen-Image-Edit-Pico_8nodes/step-10000.safetensors)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    run_evaluation(args.checkpoint, args.output_dir, args.device)

