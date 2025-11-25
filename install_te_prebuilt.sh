#!/bin/bash
# Install Transformer Engine using pre-built wheels (no compilation needed)

echo "Installing Transformer Engine using pre-built wheels..."
echo "============================================================"

# Detect Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "Python version: ${python_version}"

# Detect CUDA version
cuda_version=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")
echo "CUDA version: ${cuda_version}"

# Detect system architecture
arch=$(uname -m)
echo "Architecture: ${arch}"

# Construct wheel URL
# Check latest releases at: https://github.com/NVIDIA/TransformerEngine/releases
te_version="1.7.0"
wheel_name="transformer_engine-${te_version}+${cuda_version}-cp${python_version}-cp${python_version}-linux_${arch}.whl"
wheel_url="https://github.com/NVIDIA/TransformerEngine/releases/download/v${te_version}/${wheel_name}"

echo -e "\nDownloading pre-built wheel..."
echo "URL: ${wheel_url}"

# Try to install the pre-built wheel
pip install "${wheel_url}"

if [ $? -eq 0 ]; then
    echo -e "\n✅ Installation successful!"
    
    # Verify installation
    python -c "
try:
    import transformer_engine.pytorch as te
    print(f'✅ Transformer Engine {te.__version__} successfully installed!')
    print('   You can now use FP8 acceleration on H100/H200')
except ImportError as e:
    print('❌ Import failed:', e)
"
else
    echo -e "\n❌ Installation failed!"
    echo "Possible issues:"
    echo "1. No pre-built wheel for your Python/CUDA combination"
    echo "2. Network issues downloading the wheel"
    echo ""
    echo "Alternative: Install cuDNN and build from source:"
    echo "  conda install -c conda-forge cudnn"
    echo "  pip install transformer-engine[pytorch]"
    echo ""
    echo "Or check available wheels at:"
    echo "https://github.com/NVIDIA/TransformerEngine/releases"
fi
