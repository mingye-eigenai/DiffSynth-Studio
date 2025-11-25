import torch, math
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from einops import rearrange
from .sd3_dit import TimestepEmbeddings, RMSNorm
from .flux_dit import AdaLayerNorm
from .qwen_image_dit import (
    QwenEmbedRope, QwenFeedForward, QwenImageTransformerBlock,
    qwen_image_flash_attention, apply_rotary_emb_qwen
)


class QwenDoubleStreamAttentionMergedQKV(nn.Module):
    """
    Efficient version of QwenDoubleStreamAttention with merged Q, K, V projections.
    Reduces kernel call overhead by merging to_q, to_k, to_v into to_qkv
    and add_q_proj, add_k_proj, add_v_proj into add_qkv_proj.
    """
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Merged Q, K, V projections for image stream
        self.to_qkv = nn.Linear(dim_a, dim_a * 3)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        # Merged Q, K, V projections for text stream
        self.add_qkv_proj = nn.Linear(dim_b, dim_b * 3)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Compute merged QKV for image stream and split
        img_qkv = self.to_qkv(image)
        img_q, img_k, img_v = img_qkv.chunk(3, dim=-1)
        
        # Compute merged QKV for text stream and split
        txt_qkv = self.add_qkv_proj(text)
        txt_q, txt_k, txt_v = txt_qkv.chunk(3, dim=-1)
        
        seq_txt = txt_q.shape[1]

        # Reshape to multi-head format
        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        # Apply RMS normalization
        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        # Apply rotary embeddings
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        # Concatenate text and image streams
        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        # Apply attention
        joint_attn_out = qwen_image_flash_attention(
            joint_q, joint_k, joint_v, 
            num_heads=joint_q.shape[1], 
            attention_mask=attention_mask, 
            enable_fp8_attention=enable_fp8_attention
        ).to(joint_q.dtype)

        # Split attention outputs
        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        # Apply output projections
        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlockMergedQKV(nn.Module):
    """Transformer block with merged QKV attention."""
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int, 
        attention_head_dim: int, 
        eps: float = 1e-6,
    ):    
        super().__init__()
        
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim), 
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttentionMergedQKV(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)


    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
    
    def forward(
        self,
        image: torch.Tensor,  
        text: torch.Tensor,
        temb: torch.Tensor, 
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
        )
        
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image


class QwenImageDiTMergedQKV(torch.nn.Module):
    """
    QwenImageDiT with merged QKV projections for efficiency.
    This reduces kernel call overhead during training and inference.
    """
    def __init__(
        self,
        num_layers: int = 60,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True) 

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlockMergedQKV(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
        enable_fp8_attention=False,
    ):
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        
        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        conditioning = self.time_text_embed(timestep, image.dtype)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
                enable_fp8_attention=enable_fp8_attention,
            )
        
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        
        latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
        return image  # Return image (sequence format), not latents, to match standard implementation
    
    @staticmethod
    def state_dict_converter():
        return QwenImageDiTMergedQKVStateDictConverter()


class QwenImageDiTMergedQKVStateDictConverter():
    """
    Converts between standard QKV format and merged QKV format.
    
    When loading pretrained weights:
    - Merges to_q, to_k, to_v weights -> to_qkv
    - Merges add_q_proj, add_k_proj, add_v_proj weights -> add_qkv_proj
    
    When saving trained weights:
    - Splits to_qkv -> to_q, to_k, to_v
    - Splits add_qkv_proj -> add_q_proj, add_k_proj, add_v_proj
    """
    def __init__(self):
        pass
    
    def from_diffusers(self, state_dict):
        """Load from standard Qwen-Image format and merge QKV projections."""
        # Merge QKV projections
        merged_state_dict = {}
        skip_keys = set()
        
        # Iterate through all blocks
        for i in range(200):  # Support up to 200 blocks
            # Try to merge image stream QKV (to_q, to_k, to_v -> to_qkv)
            q_key = f"transformer_blocks.{i}.attn.to_q.weight"
            k_key = f"transformer_blocks.{i}.attn.to_k.weight"
            v_key = f"transformer_blocks.{i}.attn.to_v.weight"
            
            if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                # Merge QKV weights: concatenate along output dimension
                merged_qkv = torch.cat([
                    state_dict[q_key],
                    state_dict[k_key],
                    state_dict[v_key]
                ], dim=0)
                merged_state_dict[f"transformer_blocks.{i}.attn.to_qkv.weight"] = merged_qkv
                skip_keys.update([q_key, k_key, v_key])
                
                # Handle biases if present
                q_bias_key = f"transformer_blocks.{i}.attn.to_q.bias"
                k_bias_key = f"transformer_blocks.{i}.attn.to_k.bias"
                v_bias_key = f"transformer_blocks.{i}.attn.to_v.bias"
                if q_bias_key in state_dict:
                    merged_qkv_bias = torch.cat([
                        state_dict[q_bias_key],
                        state_dict[k_bias_key],
                        state_dict[v_bias_key]
                    ], dim=0)
                    merged_state_dict[f"transformer_blocks.{i}.attn.to_qkv.bias"] = merged_qkv_bias
                    skip_keys.update([q_bias_key, k_bias_key, v_bias_key])
            
            # Try to merge text stream QKV (add_q_proj, add_k_proj, add_v_proj -> add_qkv_proj)
            add_q_key = f"transformer_blocks.{i}.attn.add_q_proj.weight"
            add_k_key = f"transformer_blocks.{i}.attn.add_k_proj.weight"
            add_v_key = f"transformer_blocks.{i}.attn.add_v_proj.weight"
            
            if add_q_key in state_dict and add_k_key in state_dict and add_v_key in state_dict:
                # Merge add_QKV weights
                merged_add_qkv = torch.cat([
                    state_dict[add_q_key],
                    state_dict[add_k_key],
                    state_dict[add_v_key]
                ], dim=0)
                merged_state_dict[f"transformer_blocks.{i}.attn.add_qkv_proj.weight"] = merged_add_qkv
                skip_keys.update([add_q_key, add_k_key, add_v_key])
                
                # Handle biases if present
                add_q_bias_key = f"transformer_blocks.{i}.attn.add_q_proj.bias"
                add_k_bias_key = f"transformer_blocks.{i}.attn.add_k_proj.bias"
                add_v_bias_key = f"transformer_blocks.{i}.attn.add_v_proj.bias"
                if add_q_bias_key in state_dict:
                    merged_add_qkv_bias = torch.cat([
                        state_dict[add_q_bias_key],
                        state_dict[add_k_bias_key],
                        state_dict[add_v_bias_key]
                    ], dim=0)
                    merged_state_dict[f"transformer_blocks.{i}.attn.add_qkv_proj.bias"] = merged_add_qkv_bias
                    skip_keys.update([add_q_bias_key, add_k_bias_key, add_v_bias_key])
        
        # Copy all other weights that weren't merged
        for key, value in state_dict.items():
            if key not in skip_keys:
                merged_state_dict[key] = value
        
        return merged_state_dict
    
    def from_civitai(self, state_dict):
        """Alias for from_diffusers."""
        return self.from_diffusers(state_dict)
    
    def to_diffusers(self, state_dict):
        """Convert merged QKV format back to standard format."""
        split_state_dict = {}
        skip_keys = set()
        
        # Iterate through all blocks
        for i in range(200):
            # Try to split image stream QKV (to_qkv -> to_q, to_k, to_v)
            qkv_key = f"transformer_blocks.{i}.attn.to_qkv.weight"
            if qkv_key in state_dict:
                qkv_weight = state_dict[qkv_key]
                dim = qkv_weight.shape[0] // 3
                q, k, v = qkv_weight.split(dim, dim=0)
                
                split_state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = q
                split_state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = k
                split_state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = v
                skip_keys.add(qkv_key)
                
                # Handle biases
                qkv_bias_key = f"transformer_blocks.{i}.attn.to_qkv.bias"
                if qkv_bias_key in state_dict:
                    qkv_bias = state_dict[qkv_bias_key]
                    q_b, k_b, v_b = qkv_bias.split(dim, dim=0)
                    split_state_dict[f"transformer_blocks.{i}.attn.to_q.bias"] = q_b
                    split_state_dict[f"transformer_blocks.{i}.attn.to_k.bias"] = k_b
                    split_state_dict[f"transformer_blocks.{i}.attn.to_v.bias"] = v_b
                    skip_keys.add(qkv_bias_key)
            
            # Try to split text stream QKV (add_qkv_proj -> add_q_proj, add_k_proj, add_v_proj)
            add_qkv_key = f"transformer_blocks.{i}.attn.add_qkv_proj.weight"
            if add_qkv_key in state_dict:
                add_qkv_weight = state_dict[add_qkv_key]
                dim = add_qkv_weight.shape[0] // 3
                q, k, v = add_qkv_weight.split(dim, dim=0)

                split_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.weight"] = q
                split_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.weight"] = k
                split_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.weight"] = v
                skip_keys.add(add_qkv_key)

                # Handle biases
                add_qkv_bias_key = f"transformer_blocks.{i}.attn.add_qkv_proj.bias"
                if add_qkv_bias_key in state_dict:
                    add_qkv_bias = state_dict[add_qkv_bias_key]
                    q_b, k_b, v_b = add_qkv_bias.split(dim, dim=0)
                    split_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.bias"] = q_b
                    split_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.bias"] = k_b
                    split_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.bias"] = v_b
                    skip_keys.add(add_qkv_bias_key)

        # Handle LoRA keys - split merged QKV LoRA weights
        for key in list(state_dict.keys()):
            # Match pattern: transformer_blocks.{i}.attn.to_qkv.lora_{A|B}.{adapter}.weight
            if ".attn.to_qkv.lora_" in key:
                parts = key.split(".to_qkv.")
                prefix = parts[0]  # e.g., "transformer_blocks.0.attn"
                lora_suffix = parts[1]  # e.g., "lora_A.default.weight"

                qkv_weight = state_dict[key]
                # For LoRA: lora_A splits on dim=1 (input), lora_B splits on dim=0 (output)
                if ".lora_A." in key:
                    dim = qkv_weight.shape[1] // 3
                    q, k, v = qkv_weight.split(dim, dim=1)
                else:  # lora_B
                    dim = qkv_weight.shape[0] // 3
                    q, k, v = qkv_weight.split(dim, dim=0)

                split_state_dict[f"{prefix}.to_q.{lora_suffix}"] = q.contiguous()
                split_state_dict[f"{prefix}.to_k.{lora_suffix}"] = k.contiguous()
                split_state_dict[f"{prefix}.to_v.{lora_suffix}"] = v.contiguous()
                skip_keys.add(key)

            # Match pattern: transformer_blocks.{i}.attn.add_qkv_proj.lora_{A|B}.{adapter}.weight
            elif ".attn.add_qkv_proj.lora_" in key:
                parts = key.split(".add_qkv_proj.")
                prefix = parts[0]  # e.g., "transformer_blocks.0.attn"
                lora_suffix = parts[1]  # e.g., "lora_A.default.weight"

                qkv_weight = state_dict[key]
                # For LoRA: lora_A splits on dim=1 (input), lora_B splits on dim=0 (output)
                if ".lora_A." in key:
                    dim = qkv_weight.shape[1] // 3
                    q, k, v = qkv_weight.split(dim, dim=1)
                else:  # lora_B
                    dim = qkv_weight.shape[0] // 3
                    q, k, v = qkv_weight.split(dim, dim=0)

                split_state_dict[f"{prefix}.add_q_proj.{lora_suffix}"] = q.contiguous()
                split_state_dict[f"{prefix}.add_k_proj.{lora_suffix}"] = k.contiguous()
                split_state_dict[f"{prefix}.add_v_proj.{lora_suffix}"] = v.contiguous()
                skip_keys.add(key)

        # Copy all other weights
        for key, value in state_dict.items():
            if key not in skip_keys:
                split_state_dict[key] = value

        return split_state_dict

