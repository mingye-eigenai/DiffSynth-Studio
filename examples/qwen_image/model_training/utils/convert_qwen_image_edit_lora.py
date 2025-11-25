import argparse
import os
import json
import re
from typing import Dict, Tuple, List

import torch
from safetensors.torch import save_file as save_safetensors


def load_safetensors_auto(path: str, device: str = "cpu", dtype: torch.dtype | None = None) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open
    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if dtype is not None and isinstance(t, torch.Tensor):
                t = t.to(dtype)
            state_dict[k] = t
    return state_dict


def iter_index_json_shards(index_json_path: str) -> List[Tuple[str, List[str]]]:
    with open(index_json_path, "r") as f:
        idx = json.load(f)
    # weight_map: {param_name: shard_filename}
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    # group params by shard file
    shard_to_keys: Dict[str, List[str]] = {}
    for k, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(k)
    # Return sorted for determinism
    return sorted([(shard, sorted(keys)) for shard, keys in shard_to_keys.items()])


def load_state_from_index_json(index_json_path: str, root_dir: str | None = None, device: str = "cpu", dtype: torch.dtype | None = None) -> Dict[str, torch.Tensor]:
    if root_dir is None:
        root_dir = os.path.dirname(index_json_path)
    state: Dict[str, torch.Tensor] = {}
    for shard, _ in iter_index_json_shards(index_json_path):
        shard_path = os.path.join(root_dir, shard)
        if not os.path.isfile(shard_path):
            continue
        shard_sd = load_safetensors_auto(shard_path, device=device, dtype=dtype)
        state.update(shard_sd)
    return state


def strip_prefixes(name: str) -> str:
    # Remove common top-level prefixes
    prefixes = [
        "model.",
        "base_model.model.",
        "diffusion_model.",
        "transformer.",  # sometimes used
    ]
    for p in prefixes:
        if name.startswith(p):
            return name[len(p):]
    return name


def map_block_prefix(name: str) -> str:
    # Normalize block list names to transformer_blocks.*
    name = name.replace("blocks.", "transformer_blocks.")
    name = name.replace("layers.", "transformer_blocks.")
    return name


def map_out_proj(name: str) -> str:
    # Map output projection naming variants to attn.to_out.0.*
    name = name.replace("attn.o_proj.", "attn.to_out.0.")
    name = name.replace("attn.proj.", "attn.to_out.0.")
    name = name.replace("attn.to_out.", "attn.to_out.0.") if not ".to_out.0." in name else name
    return name


def map_qkv_names_for_base(name: str) -> Tuple[bool, str]:
    # Return (is_qkv_fused, mapped_base_prefix)
    # Recognize fused qkv param name and return base prefix before the suffix
    # Examples matched: attn.qkv.weight, attn.to_qkv.bias, to_qkv.weight
    m = re.search(r"(.*?)(?:attn\.)?(?:to_)?qkv\.(weight|bias)$", name)
    if m:
        return True, m.group(1) + ("attn." if not m.group(1).endswith("attn.") else "")
    return False, name


def map_single_proj_name(name: str) -> str:
    # Map single q/k/v projection aliases to to_q/to_k/to_v
    name = re.sub(r"attn\.(q_proj)\.", "attn.to_q.", name)
    name = re.sub(r"attn\.(k_proj)\.", "attn.to_k.", name)
    name = re.sub(r"attn\.(v_proj)\.", "attn.to_v.", name)
    name = re.sub(r"attn\.(q_proj|k_proj|v_proj)$", lambda m: f"attn.to_{m.group(1)[0]}.", name)
    return name


def map_added_text_proj(name: str) -> str:
    # Keep add_q_proj/add_k_proj/add_v_proj and to_add_out consistent
    name = name.replace("attn.add_out_proj.", "attn.to_add_out.")
    return name


def convert_base_key(name: str) -> Tuple[bool, List[Tuple[str, str]]]:
    # Returns (is_split, [(target_name_suffix, part)])
    # part indicates which of [q,k,v] when split, or "single" for non-split
    original = name
    name = strip_prefixes(name)
    name = map_block_prefix(name)
    name = map_out_proj(name)
    name = map_single_proj_name(name)
    name = map_added_text_proj(name)

    is_qkv, base_prefix = map_qkv_names_for_base(name)
    if is_qkv:
        # expect suffix to be added next
        if name.endswith("weight"):
            return True, [
                (base_prefix + "to_q.weight", "q"),
                (base_prefix + "to_k.weight", "k"),
                (base_prefix + "to_v.weight", "v"),
            ]
        elif name.endswith("bias"):
            return True, [
                (base_prefix + "to_q.bias", "q"),
                (base_prefix + "to_k.bias", "k"),
                (base_prefix + "to_v.bias", "v"),
            ]
    # Non-split case
    return False, [(name, "single")]


def convert_lora_key(name: str) -> Tuple[bool, List[Tuple[str, str]]]:
    # Returns (is_split, [(target_lora_key, part)])
    # Normalize and then produce .lora_A.default.weight / .lora_B.default.weight
    name = strip_prefixes(name)
    name = map_block_prefix(name)
    name = map_out_proj(name)
    name = map_single_proj_name(name)
    name = map_added_text_proj(name)

    # Detect which of A/B
    is_A = ".lora_A" in name
    is_B = ".lora_B" in name
    if not (is_A or is_B):
        # Other LoRA naming (e.g., lora_A.weight without dot path)
        if name.endswith("lora_A.weight"):
            is_A = True
            name = name[:-len("lora_A.weight")] + ".lora_A.weight"
        elif name.endswith("lora_B.weight"):
            is_B = True
            name = name[:-len("lora_B.weight")] + ".lora_B.weight"

    # Remove optional trailing ".weight"
    suffix = ".lora_A" if is_A else ".lora_B" if is_B else None
    if suffix is None:
        return False, []
    base = name.split(suffix)[0]

    # Fused qkv?
    if re.search(r"(.*?)(?:attn\.)?(?:to_)?qkv$", base):
        targets = ["to_q", "to_k", "to_v"]
        out: List[Tuple[str, str]] = []
        for part in targets:
            out_name = f"{re.sub(r'(?:attn\.)?(?:to_)?qkv$', 'attn.' + part, base)}{suffix}.default.weight"
            out.append((out_name, part))
        return True, out

    # Non-fused
    out_name = f"{base}{suffix}.default.weight"
    return False, [(out_name, "single")]


def split_tensor_along_first_dim(t: torch.Tensor, parts: int) -> List[torch.Tensor]:
    assert t.shape[0] % parts == 0, f"Cannot split tensor of shape {tuple(t.shape)} into {parts} parts along dim 0"
    chunk = t.shape[0] // parts
    return [t[i*chunk:(i+1)*chunk].clone() for i in range(parts)]


def convert_state_dict(
    src: Dict[str, torch.Tensor],
    expected_base_keys: List[str] | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    base_out: Dict[str, torch.Tensor] = {}
    lora_out: Dict[str, torch.Tensor] = {}

    # Pass 1: Extract LoRA from SVDQ-style checkpoints (proj_up/proj_down)
    # Rules:
    #  - lora_B = proj_up (out_dim, r)
    #  - lora_A = proj_down.T (r, in_dim)
    #  - For fused qkv (to_qkv/add_qkv_proj), split lora_B along dim 0 into [q,k,v] and share lora_A
    #  - Map names to DiffSynth expected module names ending with .lora_{A,B}.default.weight
    proj_up_cache: Dict[str, torch.Tensor] = {}
    proj_down_cache: Dict[str, torch.Tensor] = {}

    for k, v in src.items():
        if k.endswith('.proj_up'):
            proj_up_cache[k[:-len('.proj_up')]] = v
        elif k.endswith('.proj_down'):
            proj_down_cache[k[:-len('.proj_down')]] = v

    def save_lora_pair(base_name: str, target_base: str, fused_qkv: bool):
        if base_name not in proj_up_cache or base_name not in proj_down_cache:
            return
        up = proj_up_cache[base_name]
        down = proj_down_cache[base_name]
        # Orient as B and A
        lora_B_tensor = up  # (out_dim, r)
        lora_A_tensor = down.transpose(0, 1).contiguous()  # (r, in_dim)

        if fused_qkv:
            # Split B into q,k,v along dim 0
            parts = split_tensor_along_first_dim(lora_B_tensor, 3)
            mapping = {
                'to_q': parts[0],
                'to_k': parts[1],
                'to_v': parts[2],
            }
            for suffix, tensor_B in mapping.items():
                name_base = f"{target_base}.{suffix}"
                lora_out[f"{name_base}.lora_B.default.weight"] = tensor_B
                lora_out[f"{name_base}.lora_A.default.weight"] = lora_A_tensor
        else:
            name_base = target_base
            lora_out[f"{name_base}.lora_B.default.weight"] = lora_B_tensor
            lora_out[f"{name_base}.lora_A.default.weight"] = lora_A_tensor

    # Attention projections (image stream): to_qkv -> to_q/to_k/to_v
    for i in range(0, 200):  # generous upper bound on blocks
        base_name = f"transformer_blocks.{i}.attn.to_qkv"
        target_base = f"transformer_blocks.{i}.attn"
        save_lora_pair(base_name, target_base, fused_qkv=True)

    # Attention projections (text stream): add_qkv_proj -> add_q_proj/add_k_proj/add_v_proj
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.attn.add_qkv_proj"
        # Our module names are add_q_proj/add_k_proj/add_v_proj under attn
        # save_lora_pair handles fused split and uses to_q/to_k/to_v names, so remap target base to attn.add_* names by temporarily saving and then renaming keys
        # We'll call with target_base pointing to attn (then rename below)
        target_base = f"transformer_blocks.{i}.attn"
        if base_name in proj_up_cache and base_name in proj_down_cache:
            up = proj_up_cache[base_name]
            down = proj_down_cache[base_name]
            A = down.transpose(0, 1).contiguous()
            Bs = split_tensor_along_first_dim(up, 3)
            name_map = {
                'add_q_proj': Bs[0],
                'add_k_proj': Bs[1],
                'add_v_proj': Bs[2],
            }
            for suf, B in name_map.items():
                base = f"transformer_blocks.{i}.attn.{suf}"
                lora_out[f"{base}.lora_B.default.weight"] = B
                lora_out[f"{base}.lora_A.default.weight"] = A

    # Attention outputs
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.attn.to_out.0"
        target_base = f"transformer_blocks.{i}.attn.to_out.0"
        save_lora_pair(base_name, target_base, fused_qkv=False)

    # Added text outputs
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.attn.to_add_out"
        target_base = f"transformer_blocks.{i}.attn.to_add_out"
        save_lora_pair(base_name, target_base, fused_qkv=False)

    # MLPs (image/text) first linear (net.0.proj)
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.img_mlp.net.0.proj"
        target_base = f"transformer_blocks.{i}.img_mlp.net.0.proj"
        save_lora_pair(base_name, target_base, fused_qkv=False)
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.txt_mlp.net.0.proj"
        target_base = f"transformer_blocks.{i}.txt_mlp.net.0.proj"
        save_lora_pair(base_name, target_base, fused_qkv=False)

    # MLPs (image/text) second linear (net.2)
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.img_mlp.net.2"
        target_base = f"transformer_blocks.{i}.img_mlp.net.2"
        save_lora_pair(base_name, target_base, fused_qkv=False)
    for i in range(0, 200):
        base_name = f"transformer_blocks.{i}.txt_mlp.net.2"
        target_base = f"transformer_blocks.{i}.txt_mlp.net.2"
        save_lora_pair(base_name, target_base, fused_qkv=False)

    # Pass 2: Non-LoRA base tensors (float weights/biases already present)
    for k, v in src.items():
        k_lower = k.lower()
        if "lora" in k_lower or k.endswith(('.proj_up', '.proj_down', '.qweight', '.wscales', '.wcscales', '.wzeros', '.wtscale', '.smooth_factor', '.smooth_factor_orig')):
            # Skip SVDQ/quantization metadata and synthetic lora keys
            continue

        # Direct carry-over of float weights/biases (e.g., img_in, txt_in, time_text_embed, proj_out, norms)
        is_split, targets = convert_base_key(k)
        if is_split:
            slices = split_tensor_along_first_dim(v, 3)
            mapping = {"q": slices[0], "k": slices[1], "v": slices[2]}
            for target_name, part in targets:
                if part in mapping:
                    base_out[target_name] = mapping[part]
        else:
            base_out[targets[0][0]] = v

    # Optionally filter base_out to expected_base_keys if provided
    if expected_base_keys is not None:
        base_out = {k: v for k, v in base_out.items() if k in expected_base_keys}

    return base_out, lora_out


def collect_expected_base_keys_from_index(index_json_path: str) -> List[str]:
    keys: List[str] = []
    with open(index_json_path, "r") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    for k in weight_map.keys():
        # We only care about the DiT backbone keys (typical name contains transformer_blocks or attn/mlp/img_in/txt_in/proj_out)
        keys.append(k)
    return keys


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen Image Edit LoRA checkpoint into base+LoRA with DiffSynth naming")
    parser.add_argument("--src", required=True, help="Path to source .safetensors file (merged LoRA or LoRA)")
    parser.add_argument("--out_main", required=True, help="Output path for converted base model .safetensors")
    parser.add_argument("--out_lora", required=True, help="Output path for converted LoRA .safetensors")
    parser.add_argument("--index_json", required=False, help="Path to original Qwen-Image-Edit diffusion_pytorch_model.safetensors.index.json for expected keys")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading source: {args.src}")
    src_sd = load_safetensors_auto(args.src, device=args.device, dtype=dtype)
    print(f"  Loaded {len(src_sd)} tensors from source")

    expected_keys = None
    if args.index_json is not None:
        print(f"Reading expected base keys from index: {args.index_json}")
        expected_keys = collect_expected_base_keys_from_index(args.index_json)
        print(f"  Found {len(expected_keys)} expected base keys in index")

    base_sd, lora_sd = convert_state_dict(src_sd, expected_base_keys=expected_keys)

    os.makedirs(os.path.dirname(args.out_main) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_lora) or ".", exist_ok=True)

    print(f"Writing base model to: {args.out_main} ({len(base_sd)} tensors)")
    save_safetensors(base_sd, args.out_main)

    print(f"Writing LoRA model to: {args.out_lora} ({len(lora_sd)} tensors)")
    save_safetensors(lora_sd, args.out_lora)

    # Quick sanity: show a couple of keys
    show_n = 10
    print("\nSample base keys:")
    for i, k in enumerate(sorted(base_sd.keys())):
        if i >= show_n:
            break
        print(" ", k, tuple(base_sd[k].shape))
    print("\nSample LoRA keys:")
    for i, k in enumerate(sorted(lora_sd.keys())):
        if i >= show_n:
            break
        print(" ", k, tuple(lora_sd[k].shape))


if __name__ == "__main__":
    main()


