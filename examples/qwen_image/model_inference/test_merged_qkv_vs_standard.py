"""
Debug script to compare merged QKV model vs standard model (both without LoRA).
This verifies the merged QKV conversion is working correctly.
"""

import torch
from PIL import Image
import argparse
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth import load_state_dict
from diffsynth.models import ModelManager


def load_standard_pipeline(device="cuda", torch_dtype=torch.bfloat16):
    """Load standard Qwen-Image-Edit pipeline (no LoRA)."""
    print("\n" + "="*60)
    print("Loading STANDARD pipeline (separate Q, K, V)")
    print("="*60)
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    print("✓ Standard pipeline loaded")
    return pipe


def load_merged_qkv_pipeline(device="cuda", torch_dtype=torch.bfloat16):
    """Load merged QKV pipeline (no LoRA)."""
    print("\n" + "="*60)
    print("Loading MERGED QKV pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
    
    # Load base components
    print("1. Loading base components...")
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
    
    dit_merged = QwenImageDiTMergedQKV(num_layers=60)
    dit_merged = dit_merged.to(dtype=torch_dtype, device=device)
    
    # Load base weights
    dit_path = dit_config.path
    if isinstance(dit_path, list):
        print(f"   Loading {len(dit_path)} shards...")
        state_dict = {}
        for shard_path in dit_path:
            state_dict.update(load_state_dict(shard_path))
    else:
        state_dict = load_state_dict(dit_path)
    
    print(f"   Loaded {len(state_dict)} parameters from base DiT")
    
    # Convert to merged QKV format
    print("   Converting to merged QKV format...")
    converter = dit_merged.state_dict_converter()
    merged_state_dict = converter.from_diffusers(state_dict)
    print(f"   Converted: {len(merged_state_dict)} parameters")
    
    # Load converted weights
    missing, unexpected = dit_merged.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"   Missing keys: {len(missing)} (this can be normal)")
    if unexpected:
        print(f"   Unexpected keys: {len(unexpected)} (this can be normal)")
    
    pipe.dit = dit_merged
    print("✓ Merged QKV DiT loaded")
    
    return pipe


def compare_outputs(standard_output, merged_output, threshold=0.01):
    """Compare two images and return similarity metrics."""
    import numpy as np
    
    # Convert to numpy arrays
    standard_np = np.array(standard_output)
    merged_np = np.array(merged_output)
    
    # Calculate differences
    diff = np.abs(standard_np.astype(float) - merged_np.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # Calculate similarity percentage
    total_pixels = standard_np.size
    similar_pixels = np.sum(diff < threshold * 255)
    similarity = (similar_pixels / total_pixels) * 100
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'similarity_percent': similarity,
        'are_similar': mean_diff < 5.0  # Threshold: mean difference < 5 pixel values
    }


def main():
    parser = argparse.ArgumentParser(description="Compare merged QKV vs standard model")
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_standard", default="output_standard.jpg", help="Output from standard model")
    parser.add_argument("--output_merged", default="output_merged.jpg", help="Output from merged QKV model")
    parser.add_argument("--output_diff", default="output_diff.jpg", help="Difference visualization")
    parser.add_argument("--prompt", default="Chibi-style 3D cartoon ghost, Pixar/Disney style", help="Editing prompt")
    parser.add_argument("--height", type=int, default=1152, help="Output height")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load input image
    print("\n" + "="*60)
    print("TEST: Merged QKV vs Standard Model (NO LoRA)")
    print("="*60)
    print(f"\nInput image: {args.input_image}")
    input_image = Image.open(args.input_image).convert("RGB")
    print(f"Input size: {input_image.size}")
    
    # Test parameters
    print(f"\nTest parameters:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")
    
    # Load standard pipeline
    standard_pipe = load_standard_pipeline(device=args.device, torch_dtype=torch.bfloat16)
    
    # Generate with standard model
    print("\n" + "="*60)
    print("Generating with STANDARD model...")
    print("="*60)
    standard_output = standard_pipe(
        args.prompt,
        edit_image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        edit_image_auto_resize=True,
    )
    standard_output.save(args.output_standard)
    print(f"✓ Standard output saved: {args.output_standard}")
    
    # Clear GPU memory
    del standard_pipe
    torch.cuda.empty_cache()
    
    # Load merged QKV pipeline
    merged_pipe = load_merged_qkv_pipeline(device=args.device, torch_dtype=torch.bfloat16)
    
    # Generate with merged QKV model
    print("\n" + "="*60)
    print("Generating with MERGED QKV model...")
    print("="*60)
    merged_output = merged_pipe(
        args.prompt,
        edit_image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        edit_image_auto_resize=True,
    )
    merged_output.save(args.output_merged)
    print(f"✓ Merged QKV output saved: {args.output_merged}")
    
    # Compare outputs
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    comparison = compare_outputs(standard_output, merged_output)
    
    print(f"\nPixel-level comparison:")
    print(f"  Max difference: {comparison['max_diff']:.2f} (0-255 scale)")
    print(f"  Mean difference: {comparison['mean_diff']:.2f} (0-255 scale)")
    print(f"  Similarity: {comparison['similarity_percent']:.2f}%")
    
    if comparison['are_similar']:
        print("\n✅ SUCCESS: Outputs are very similar!")
        print("   The merged QKV model is working correctly.")
        print("   Mean difference < 5.0 indicates proper conversion.")
    else:
        print("\n⚠️  WARNING: Outputs differ significantly!")
        print("   Mean difference >= 5.0 indicates possible conversion issue.")
        print("   Please review the conversion logic.")
    
    # Create difference visualization
    import numpy as np
    standard_np = np.array(standard_output)
    merged_np = np.array(merged_output)
    diff_np = np.abs(standard_np.astype(float) - merged_np.astype(float))
    diff_np = (diff_np * 10).clip(0, 255).astype(np.uint8)  # Amplify differences
    diff_img = Image.fromarray(diff_np)
    diff_img.save(args.output_diff)
    print(f"\n✓ Difference map saved: {args.output_diff}")
    print("  (Differences amplified 10x for visibility)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard output:  {args.output_standard}")
    print(f"Merged output:    {args.output_merged}")
    print(f"Difference map:   {args.output_diff}")
    print(f"Similarity:       {comparison['similarity_percent']:.2f}%")
    print(f"Status:           {'✅ PASS' if comparison['are_similar'] else '⚠️  FAIL'}")
    print("="*60 + "\n")
    
    return 0 if comparison['are_similar'] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


