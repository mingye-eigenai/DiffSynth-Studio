#!/bin/bash

echo "Fixing Flash Attention installation..."
echo "====================================="

# 1. Find CUDA installation
echo "1. Locating CUDA installation..."
if [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.8" ]; then
    CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12" ]; then
    CUDA_HOME="/usr/local/cuda-12"
else
    echo "Searching for CUDA..."
    CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo "not_found")))
fi

if [ "$CUDA_HOME" == "not_found" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "❌ CUDA not found in standard locations"
    echo ""
    echo "Please install CUDA toolkit or find your CUDA installation:"
    echo "  - Check: ls /usr/local/ | grep cuda"
    echo "  - Or: which nvcc"
    echo "  - Or: conda install -c nvidia cuda-toolkit"
    exit 1
fi

echo "✅ Found CUDA at: $CUDA_HOME"

# 2. Check for nvcc
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "✅ nvcc found at: $CUDA_HOME/bin/nvcc"
else
    echo "❌ nvcc not found at $CUDA_HOME/bin/nvcc"
    echo "Installing CUDA toolkit..."
    echo "Run: conda install -c nvidia cuda-toolkit"
    exit 1
fi

# 3. Set environment variables
echo ""
echo "2. Setting environment variables..."
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "✅ Environment variables set:"
echo "   CUDA_HOME=$CUDA_HOME"
echo "   PATH includes $CUDA_HOME/bin"

# 4. Install Flash Attention with proper flags
echo ""
echo "3. Installing Flash Attention..."
echo "This will take a few minutes to compile..."

# Option 1: Try pip install with env vars
CUDA_HOME=$CUDA_HOME pip install flash-attn --no-build-isolation

# Check if installation succeeded
python -c "import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed successfully!')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation failed. Trying alternative method..."
    echo ""
    echo "Alternative installation commands to try:"
    echo ""
    echo "1. With conda CUDA toolkit:"
    echo "   conda install -c nvidia cuda-toolkit"
    echo "   export CUDA_HOME=\$CONDA_PREFIX"
    echo "   pip install flash-attn --no-build-isolation"
    echo ""
    echo "2. Pre-built wheel (if available):"
    echo "   pip install flash-attn --no-build-isolation --index-url https://flash-attention.github.io/whl/cu128"
    echo ""
    echo "3. From source with specific CUDA:"
    echo "   cd /path/to/flash-attention"
    echo "   CUDA_HOME=$CUDA_HOME python setup.py install"
else
    echo ""
    echo "🎉 Flash Attention installed successfully!"
    echo ""
    echo "Add these to your ~/.bashrc to make permanent:"
    echo "export CUDA_HOME=$CUDA_HOME"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
fi
