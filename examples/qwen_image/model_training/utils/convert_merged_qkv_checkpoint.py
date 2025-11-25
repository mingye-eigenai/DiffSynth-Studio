"""
Utility script to convert checkpoints between standard and merged QKV formats.

Usage:
    # Convert standard checkpoint to merged QKV format
    python convert_merged_qkv_checkpoint.py --input model.safetensors --output model_merged_qkv.safetensors --direction to_merged
    
    # Convert merged QKV checkpoint back to standard format
    python convert_merged_qkv_checkpoint.py --input model_merged_qkv.safetensors --output model.safetensors --direction to_standard
"""

import argparse
import torch
import sys
import os
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

# Add DiffSynth-Studio to path to import local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
diffsynth_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
if diffsynth_root not in sys.path:
    sys.path.insert(0, diffsynth_root)

from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKVStateDictConverter


def main():
    parser = argparse.ArgumentParser(description="Convert between standard and merged QKV checkpoint formats")
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--direction", required=True, choices=["to_merged", "to_standard"],
                       help="Conversion direction: 'to_merged' merges Q,K,V; 'to_standard' splits them")
    parser.add_argument("--device", default="cpu", help="Device to use for loading (default: cpu)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                       help="Data type for conversion (default: bfloat16)")
    
    args = parser.parse_args()
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Loading checkpoint from: {args.input}")
    state_dict = load_safetensors(args.input)
    
    # Convert to target device and dtype
    state_dict = {k: v.to(device=args.device, dtype=dtype) for k, v in state_dict.items()}
    
    print(f"Loaded {len(state_dict)} tensors")
    print(f"Converting {args.direction}...")
    
    converter = QwenImageDiTMergedQKVStateDictConverter()
    
    if args.direction == "to_merged":
        converted_state_dict = converter.from_diffusers(state_dict)
        print(f"Converted to merged QKV format")
        print("Merged layers:")
        print("  - to_q, to_k, to_v → to_qkv")
        print("  - add_q_proj, add_k_proj, add_v_proj → add_qkv_proj")
    else:  # to_standard
        converted_state_dict = converter.to_diffusers(state_dict)
        print(f"Converted to standard format")
        print("Split layers:")
        print("  - to_qkv → to_q, to_k, to_v")
        print("  - add_qkv_proj → add_q_proj, add_k_proj, add_v_proj")
    
    print(f"Saving checkpoint to: {args.output}")
    save_safetensors(converted_state_dict, args.output)
    print(f"Saved {len(converted_state_dict)} tensors")
    
    # Show sample keys
    print("\nSample keys in output (first 10):")
    for i, key in enumerate(sorted(converted_state_dict.keys())[:10]):
        print(f"  {key}: {tuple(converted_state_dict[key].shape)}")
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()

