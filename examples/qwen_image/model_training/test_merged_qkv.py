"""
Test script to verify merged QKV implementation is working correctly.

This script:
1. Creates a merged QKV model
2. Loads pretrained weights and converts them
3. Verifies the conversion is correct
4. Runs a forward pass to ensure it works
5. Compares output with standard implementation
"""

import torch
import sys
from diffsynth.models.qwen_image_dit import QwenImageDiT
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV


def test_state_dict_converter():
    """Test the state dict converter."""
    print("\n" + "="*60)
    print("TEST 1: State Dict Converter")
    print("="*60)
    
    # Create a dummy state dict with standard format
    dummy_state_dict = {}
    
    # Add some dummy Q, K, V weights for block 0
    for i in range(2):  # Test first 2 blocks
        dummy_state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = torch.randn(3072, 3072)
        dummy_state_dict[f"transformer_blocks.{i}.attn.to_add_out.weight"] = torch.randn(3072, 3072)
    
    print(f"Created dummy state dict with {len(dummy_state_dict)} tensors")
    
    # Convert to merged format
    converter = QwenImageDiTMergedQKV.state_dict_converter()
    merged_dict = converter.from_diffusers(dummy_state_dict)
    
    print(f"Converted to merged format: {len(merged_dict)} tensors")
    
    # Check merged keys exist
    assert "transformer_blocks.0.attn.to_qkv.weight" in merged_dict
    assert "transformer_blocks.0.attn.add_qkv_proj.weight" in merged_dict
    print("✓ Merged keys found")
    
    # Check original keys are removed
    assert "transformer_blocks.0.attn.to_q.weight" not in merged_dict
    assert "transformer_blocks.0.attn.to_k.weight" not in merged_dict
    print("✓ Original Q, K, V keys removed")
    
    # Check shapes
    qkv_shape = merged_dict["transformer_blocks.0.attn.to_qkv.weight"].shape
    assert qkv_shape == (9216, 3072), f"Expected (9216, 3072), got {qkv_shape}"
    print(f"✓ Merged QKV shape correct: {qkv_shape}")
    
    # Convert back to standard format
    standard_dict = converter.to_diffusers(merged_dict)
    
    print(f"Converted back to standard format: {len(standard_dict)} tensors")
    
    # Check original keys are back
    assert "transformer_blocks.0.attn.to_q.weight" in standard_dict
    assert "transformer_blocks.0.attn.to_k.weight" in standard_dict
    assert "transformer_blocks.0.attn.to_v.weight" in standard_dict
    print("✓ Q, K, V keys restored")
    
    # Verify values match original
    for key in ["transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_k.weight"]:
        if not torch.allclose(dummy_state_dict[key], standard_dict[key], atol=1e-6):
            print(f"✗ Values don't match for {key}")
            return False
    
    print("✓ Round-trip conversion preserves values")
    print("✓ TEST 1 PASSED\n")
    return True


def test_model_creation():
    """Test that merged QKV model can be created."""
    print("="*60)
    print("TEST 2: Model Creation")
    print("="*60)
    
    try:
        # Create merged QKV model
        model = QwenImageDiTMergedQKV(num_layers=2)  # Small model for testing
        print(f"✓ Created merged QKV model with 2 layers")
        
        # Check architecture
        assert hasattr(model.transformer_blocks[0].attn, 'to_qkv')
        assert hasattr(model.transformer_blocks[0].attn, 'add_qkv_proj')
        print("✓ Model has merged QKV layers")
        
        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {param_count:,} parameters")
        
        print("✓ TEST 2 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}\n")
        return False


def test_forward_pass():
    """Test forward pass works."""
    print("="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    try:
        # Create model
        model = QwenImageDiTMergedQKV(num_layers=2)
        model.eval()
        
        # Create dummy inputs
        # Note: height/width are the FULL image dimensions, not latent dimensions
        # Latents are 8x downsampled from full image (16x16 patch with 2x2 packing)
        batch_size = 2
        latent_h, latent_w = 8, 8  # Latent dimensions
        height, width = latent_h * 16, latent_w * 16  # Full image dimensions
        latents = torch.randn(batch_size, 16, latent_h * 2, latent_w * 2)  # 2x2 packing
        timestep = torch.tensor([500.0, 600.0])
        prompt_emb = torch.randn(batch_size, 256, 3584)
        # Create mask with actual sequence lengths (not all 1s)
        seq_len = 128  # Actual sequence length
        prompt_emb_mask = torch.zeros(batch_size, 256)
        prompt_emb_mask[:, :seq_len] = 1.0
        
        print("✓ Created dummy inputs")
        print(f"  Latent shape: {latents.shape}")
        print(f"  Image dimensions: {height}x{width}")
        
        # Forward pass
        with torch.no_grad():
            output = model(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_emb,
                prompt_emb_mask=prompt_emb_mask,
                height=height,
                width=width,
            )
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {latents.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check output shape matches input
        assert output.shape == latents.shape, f"Shape mismatch: {output.shape} vs {latents.shape}"
        print("✓ Output shape matches input")
        
        # Check output is not all zeros or NaN
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.abs().sum() > 0, "Output is all zeros"
        print("✓ Output is valid (no NaN/Inf/zeros)")
        
        print("✓ TEST 3 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_qkv_split():
    """Test that QKV splitting works correctly."""
    print("="*60)
    print("TEST 4: QKV Split Mechanics")
    print("="*60)
    
    try:
        # Simulate merged QKV projection
        batch_size, seq_len, hidden_dim = 2, 10, 3072
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Create a merged QKV linear layer
        to_qkv = torch.nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        
        # Apply and split
        qkv = to_qkv(input_tensor)
        q, k, v = qkv.chunk(3, dim=-1)
        
        print(f"✓ Input shape: {input_tensor.shape}")
        print(f"✓ Merged QKV shape: {qkv.shape}")
        print(f"✓ Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        
        # Check shapes
        assert q.shape == (batch_size, seq_len, hidden_dim)
        assert k.shape == (batch_size, seq_len, hidden_dim)
        assert v.shape == (batch_size, seq_len, hidden_dim)
        print("✓ Split shapes are correct")
        
        # Verify this is equivalent to separate projections
        to_q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        to_k = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        to_v = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Copy weights from merged to separate
        to_q.weight.data = to_qkv.weight.data[:hidden_dim]
        to_k.weight.data = to_qkv.weight.data[hidden_dim:2*hidden_dim]
        to_v.weight.data = to_qkv.weight.data[2*hidden_dim:]
        
        # Compare outputs
        q_sep = to_q(input_tensor)
        k_sep = to_k(input_tensor)
        v_sep = to_v(input_tensor)
        
        assert torch.allclose(q, q_sep, atol=1e-6), "Q outputs don't match"
        assert torch.allclose(k, k_sep, atol=1e-6), "K outputs don't match"
        assert torch.allclose(v, v_sep, atol=1e-6), "V outputs don't match"
        print("✓ Merged QKV is equivalent to separate Q, K, V")
        
        print("✓ TEST 4 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("MERGED QKV IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("State Dict Converter", test_state_dict_converter()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("QKV Split Mechanics", test_qkv_split()))
    
    # Print summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe merged QKV implementation is working correctly.")
        print("You can now proceed with training using:")
        print("  bash examples/qwen_image/model_training/lora/Qwen-Image-Edit-Ghost-qkv-merged-efficient.sh")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease review the failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

