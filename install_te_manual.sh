#!/bin/bash
# Manual installation of Transformer Engine with proper paths

echo "Installing Transformer Engine manually..."
echo "============================================================"

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export CUDNN_INCLUDE_DIR=/usr/include
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu
export CPATH=/usr/include:$CPATH

# Show environment
echo "Environment setup:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_INCLUDE_DIR: $CUDNN_INCLUDE_DIR"
echo "CUDNN_LIB_DIR: $CUDNN_LIB_DIR"

# Install with proper flags
echo -e "\nInstalling Transformer Engine..."
CUDNN_INCLUDE_DIR=/usr/include CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu pip install transformer-engine[pytorch] --no-build-isolation --verbose

# If that fails, try git install
if [ $? -ne 0 ]; then
    echo -e "\nTrying git install..."
    pip install git+https://github.com/NVIDIA/TransformerEngine.git --no-build-isolation
fi

# Verify
python -c "
try:
    import transformer_engine.pytorch as te
    print(f'✅ Transformer Engine {te.__version__} installed!')
except ImportError as e:
    print('❌ Installation failed:', e)
"
