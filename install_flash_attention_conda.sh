#!/bin/bash

echo "Installing Flash Attention using Conda (easier method)..."
echo "========================================================"

# 1. Install CUDA toolkit via conda
echo "1. Installing CUDA toolkit through conda..."
conda install -y -c nvidia cuda-toolkit

if [ $? -ne 0 ]; then
    echo "❌ Failed to install CUDA toolkit"
    echo "Try: conda install -c conda-forge cudatoolkit-dev"
    exit 1
fi

# 2. Set CUDA_HOME to conda environment
echo ""
echo "2. Setting CUDA environment..."
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "✅ CUDA_HOME set to: $CUDA_HOME"

# 3. Verify nvcc is available
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc found: $(which nvcc)"
    echo "   Version: $(nvcc --version | head -n1)"
else
    echo "❌ nvcc still not found"
    exit 1
fi

# 4. Install dependencies
echo ""
echo "3. Installing build dependencies..."
pip install ninja packaging

# 5. Try to install pre-built wheel first
echo ""
echo "4. Trying pre-built wheel installation..."
pip install flash-attn --index-url https://flash-attention.github.io/whl/cu128

# Check if it worked
python -c "import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed!')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "Pre-built wheel not available, building from source..."
    echo "This will take 5-15 minutes..."
    
    # Build from source
    CUDA_HOME=$CONDA_PREFIX pip install flash-attn --no-build-isolation
    
    # Final check
    python -c "import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed!')" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed"
        echo ""
        echo "Manual steps to try:"
        echo "1. cd ~/workspace/repos/flash-attention"
        echo "2. export CUDA_HOME=$CONDA_PREFIX"
        echo "3. python setup.py install"
        exit 1
    fi
fi

echo ""
echo "🎉 Success! Flash Attention is installed!"
echo ""
echo "To make CUDA settings permanent, add to ~/.bashrc:"
echo "export CUDA_HOME=\$CONDA_PREFIX"
echo ""
echo "Now run your optimized inference script!"
