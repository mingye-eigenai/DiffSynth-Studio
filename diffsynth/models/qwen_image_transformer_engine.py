"""
Transformer Engine FP8 wrapper for QwenImage model
Provides true FP8 acceleration on H100/H200 GPUs
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("⚠️ Transformer Engine not available. Install with: bash install_transformer_engine.sh")

class TransformerEngineFP8Linear(nn.Module):
    """FP8 Linear layer using Transformer Engine"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if not TRANSFORMER_ENGINE_AVAILABLE:
            # Fallback to regular Linear
            self.te_module = nn.Linear(in_features, out_features, bias=bias)
        else:
            # Use Transformer Engine Linear with FP8
            self.te_module = te.Linear(
                in_features, 
                out_features, 
                bias=bias,
                params_dtype=torch.bfloat16,  # Weights in BF16
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.te_module(x)

class TransformerEngineFP8Attention(nn.Module):
    """FP8 Attention using Transformer Engine"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        if not TRANSFORMER_ENGINE_AVAILABLE:
            # Fallback to regular attention
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
            self.proj = nn.Linear(hidden_size, hidden_size)
            self.use_te = False
        else:
            # Use Transformer Engine modules
            self.attention = te.DotProductAttention(
                num_attention_heads=num_attention_heads,
                kv_channels=self.head_dim,
                attention_dropout=attention_dropout,
                attn_mask_type="no_mask",
            )
            self.qkv = te.Linear(hidden_size, 3 * hidden_size, bias=True)
            self.proj = te.Linear(hidden_size, hidden_size, bias=True)
            self.use_te = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute QKV
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_te:
            # Use Transformer Engine attention (with FP8)
            context = self.attention(q, k, v, attention_mask=attention_mask)
        else:
            # Fallback to PyTorch SDPA
            context = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        output = self.proj(context)
        
        return output

def get_fp8_recipe(margin: int = 0, fp8_format: str = "HYBRID"):
    """Get FP8 recipe for Transformer Engine
    
    Args:
        margin: Margin for FP8 scaling factor
        fp8_format: FP8 format - "E4M3" or "HYBRID" (E4M3 for forward, E5M2 for backward)
    """
    if not TRANSFORMER_ENGINE_AVAILABLE:
        return None
        
    return recipe.DelayedScaling(
        margin=margin,
        fp8_format=recipe.Format[fp8_format],
        amax_history_len=16,
        amax_compute_algo="max",
    )

def replace_linear_with_te_fp8(model: nn.Module, skip_patterns: Optional[list] = None):
    """Replace nn.Linear layers with Transformer Engine FP8 Linear layers
    
    Args:
        model: The model to modify
        skip_patterns: List of module name patterns to skip
    """
    if not TRANSFORMER_ENGINE_AVAILABLE:
        print("⚠️ Transformer Engine not available, skipping FP8 replacement")
        return model
    
    skip_patterns = skip_patterns or []
    
    def should_skip(name: str) -> bool:
        for pattern in skip_patterns:
            if pattern in name:
                return True
        return False
    
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # Replace with TE Linear
            te_linear = TransformerEngineFP8Linear(
                module.in_features,
                module.out_features,
                module.bias is not None
            )
            
            # Copy weights
            with torch.no_grad():
                te_linear.te_module.weight.copy_(module.weight)
                if module.bias is not None:
                    te_linear.te_module.bias.copy_(module.bias)
            
            # Replace module
            setattr(parent, module_name, te_linear)
            replaced_count += 1
    
    print(f"✅ Replaced {replaced_count} Linear layers with Transformer Engine FP8 Linear")
    return model

def enable_te_fp8_autocast(enabled: bool = True, fp8_recipe=None):
    """Context manager for Transformer Engine FP8 autocast
    
    Usage:
        with enable_te_fp8_autocast():
            output = model(input)
    """
    if not TRANSFORMER_ENGINE_AVAILABLE:
        # Return a no-op context manager
        import contextlib
        return contextlib.nullcontext()
    
    fp8_recipe = fp8_recipe or get_fp8_recipe()
    return te.fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe)

# Export info
__all__ = [
    'TRANSFORMER_ENGINE_AVAILABLE',
    'TransformerEngineFP8Linear',
    'TransformerEngineFP8Attention',
    'get_fp8_recipe',
    'replace_linear_with_te_fp8',
    'enable_te_fp8_autocast',
]
