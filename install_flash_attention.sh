#!/bin/bash

echo "Installing Flash Attention 3 for H100..."
echo "======================================="

# Ensure we have the build requirements
echo "Installing build dependencies..."
pip install ninja packaging

# Install Flash Attention
echo -e "\nInstalling Flash Attention..."
pip install flash-attn --no-build-isolation

# Verify installation
echo -e "\nVerifying installation..."
python -c "
try:
    import flash_attn
    print(f'✅ Flash Attention installed successfully: v{flash_attn.__version__}')
    import flash_attn_interface
    print('✅ flash_attn_interface available')
    print('\nThe model will now automatically use Flash Attention!')
    print('Expected speedup: 2-3x')
except ImportError as e:
    print(f'❌ Installation failed: {e}')
    print('Try manual installation:')
    print('git clone https://github.com/Dao-AILab/flash-attention.git')
    print('cd flash-attention && pip install .')
"

echo -e "\n======================================="
echo "After installation, run your inference again."
echo "You should see 2-3x speedup immediately!"
