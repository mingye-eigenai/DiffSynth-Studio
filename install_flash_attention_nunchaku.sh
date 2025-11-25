#!/bin/bash

echo "Installing Flash Attention in nunchaku environment"
echo "=================================================="

# Make sure we're in the right conda environment
if [[ "$CONDA_DEFAULT_ENV" != "nunchaku" ]]; then
    echo "⚠️  Not in nunchaku environment. Activating it..."
    conda activate nunchaku
fi

echo "✅ Using conda environment: $CONDA_DEFAULT_ENV"

# Set CUDA paths (CUDA 12.8 is installed)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "✅ CUDA_HOME set to: $CUDA_HOME"

# Check nvcc
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "✅ nvcc found: $CUDA_HOME/bin/nvcc"
    $CUDA_HOME/bin/nvcc --version | head -n1
else
    echo "❌ nvcc not found!"
    exit 1
fi

# Install build dependencies
echo ""
echo "Installing build dependencies..."
pip install ninja packaging wheel

# Try different installation methods
echo ""
echo "Method 1: Trying with explicit CUDA_HOME..."
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn --no-build-isolation

# Check if it worked
python -c "import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed!')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "Method 1 failed. Trying Method 2: Pre-built wheel..."
    
    # Try pre-built wheel for H100 (sm_90)
    pip install flash-attn --extra-index-url https://download.pytorch.org/whl/cu121
    
    python -c "import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed!')" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "Method 2 failed. Trying Method 3: From GitHub directly..."
        
        # Clone and install from source
        cd /tmp
        rm -rf flash-attention
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
        
        # For H100, we need to specify the architecture
        export TORCH_CUDA_ARCH_LIST="9.0"
        export MAX_JOBS=8  # Limit parallel compilation jobs
        
        CUDA_HOME=/usr/local/cuda-12.8 python setup.py install
        
        cd /home/kuan/workspace/repos/DiffSynth-Studio
    fi
fi

# Final verification
echo ""
echo "Verifying installation..."
python -c "
import sys
try:
    import flash_attn
    import flash_attn_interface
    print(f'✅ SUCCESS! Flash Attention {flash_attn.__version__} is installed!')
    
    # Check if it's detected by the model
    try:
        from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
        if FLASH_ATTN_3_AVAILABLE:
            print('✅ Model detects Flash Attention 3!')
        else:
            print('⚠️  Flash Attention installed but not detected by model')
    except:
        print('⚠️  Could not verify model detection')
    
    print('')
    print('🚀 You can now run the H100-optimized script!')
    print('   Expected speedup: 2-3x (from ~42s to ~15s per image)')
    
except ImportError as e:
    print(f'❌ Flash Attention not installed: {e}')
    print('')
    print('Try manual installation:')
    print('1. cd /tmp')
    print('2. git clone https://github.com/Dao-AILab/flash-attention.git')
    print('3. cd flash-attention')
    print('4. export CUDA_HOME=/usr/local/cuda-12.8')
    print('5. export TORCH_CUDA_ARCH_LIST=\"9.0\"')
    print('6. python setup.py install')
    sys.exit(1)
"

# Add environment variables reminder
echo ""
echo "To make CUDA settings permanent, add to ~/.bashrc:"
echo "export CUDA_HOME=/usr/local/cuda-12.8"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
