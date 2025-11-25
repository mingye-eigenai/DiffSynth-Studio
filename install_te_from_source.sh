#!/bin/bash
# Install Transformer Engine from source with proper cuDNN setup

echo "Installing Transformer Engine from source..."
echo "============================================================"

# Step 1: Install cuDNN via conda
echo "Step 1: Installing cuDNN..."
conda install -c conda-forge cudnn -y

# Step 2: Set environment variables
echo -e "\nStep 2: Setting environment variables..."
export CUDA_HOME=/usr/local/cuda
export CUDNN_PATH=$CONDA_PREFIX
export CUDNN_INCLUDE_DIR=$CUDNN_PATH/include
export CUDNN_LIBRARY=$CUDNN_PATH/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_PATH/lib:$LD_LIBRARY_PATH

# Show environment
echo "Environment setup:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDNN_PATH: $CUDNN_PATH"
echo "  CUDNN_INCLUDE_DIR: $CUDNN_INCLUDE_DIR"
echo "  CUDNN_LIBRARY: $CUDNN_LIBRARY"

# Step 3: Install from GitHub
echo -e "\nStep 3: Installing Transformer Engine from source..."
echo "This may take 5-10 minutes as it compiles CUDA kernels..."

# Use pip with verbose output to see progress
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable --verbose

# Step 4: Verify installation
echo -e "\nStep 4: Verifying installation..."
python -c "
try:
    import transformer_engine
    print(f'✅ Transformer Engine {transformer_engine.__version__} successfully installed!')
    
    import transformer_engine.pytorch as te
    print('✅ PyTorch backend available')
    
    # Check FP8 support
    import torch
    if torch.cuda.get_device_capability()[0] >= 9:
        print('✅ Your GPU supports FP8 (Hopper architecture)')
    else:
        print('⚠️  Your GPU does not support native FP8')
        
except ImportError as e:
    print(f'❌ Installation failed: {e}')
    print('Try the fallback method below.')
"

echo -e "\n============================================================"
echo "If installation failed, try this fallback method:"
echo "============================================================"
echo "# 1. Clone and build manually"
echo "git clone https://github.com/NVIDIA/TransformerEngine.git"
echo "cd TransformerEngine"
echo "git checkout stable"
echo "export CUDNN_PATH=\$CONDA_PREFIX"
echo "pip install ."
