#!/bin/bash
# Install NVIDIA Transformer Engine for H100 FP8 support

echo "Installing NVIDIA Transformer Engine for H100 FP8 acceleration..."
echo "============================================================"

# Check CUDA version
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "Detected CUDA version: $cuda_version"

# Check if we're on H100
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $gpu_name"

if [[ ! "$gpu_name" =~ "H100" ]]; then
    echo "⚠️ WARNING: Not running on H100. Transformer Engine FP8 requires Hopper architecture."
fi

# Install Transformer Engine
echo -e "\nInstalling Transformer Engine..."

# First try to install cuDNN if not present
if ! python -c "import os; exit(0 if any(os.path.exists(p) for p in ['/usr/include/cudnn.h', '/usr/local/cuda/include/cudnn.h', os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include/cudnn.h')]) else 1)" 2>/dev/null; then
    echo "cuDNN headers not found. Installing via conda..."
    conda install -c conda-forge cudnn -y
fi

# Set environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [ -n "$CONDA_PREFIX" ]; then
    export CUDNN_HOME=$CONDA_PREFIX
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib:$LD_LIBRARY_PATH
fi

# Try pre-built wheel first (faster, no compilation)
echo "Attempting to install pre-built wheel..."
python_version=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
cuda_version=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")
te_version="1.7.0"
wheel_url="https://github.com/NVIDIA/TransformerEngine/releases/download/v${te_version}/transformer_engine-${te_version}+${cuda_version}-cp${python_version}-cp${python_version}-linux_x86_64.whl"

if pip install "${wheel_url}" 2>/dev/null; then
    echo "✅ Installed pre-built wheel successfully!"
else
    echo "Pre-built wheel not available, building from source..."
    # For PyTorch 2.4+ with CUDA 12.x
    pip install transformer-engine[pytorch] --no-build-isolation
fi

# Verify installation
echo -e "\nVerifying installation..."
python -c "
try:
    import transformer_engine.pytorch as te
    print(f'✅ Transformer Engine {te.__version__} successfully installed!')
    print('   FP8 acceleration is now available for H100')
except ImportError as e:
    print('❌ Installation failed:', e)
    print('   Try manual installation:')
    print('   pip install git+https://github.com/NVIDIA/TransformerEngine.git')
"

echo -e "\nInstallation complete!"
echo "Note: Transformer Engine requires:"
echo "  - Hopper architecture GPU (H100/H200)"
echo "  - PyTorch 2.1+"
echo "  - CUDA 11.8+"
