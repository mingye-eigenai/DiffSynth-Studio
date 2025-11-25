#!/bin/bash
# Fix Transformer Engine installation issues

echo "Diagnosing Transformer Engine installation issues..."
echo "============================================================"

# Check CUDA installation
echo -e "\n1. Checking CUDA installation..."
if [ -d "/usr/local/cuda" ]; then
    echo "✅ CUDA found at /usr/local/cuda"
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "   Version: $cuda_version"
else
    echo "❌ CUDA not found at /usr/local/cuda"
fi

# Check for cuDNN
echo -e "\n2. Checking cuDNN..."
cudnn_locations=(
    "/usr/include/cudnn.h"
    "/usr/local/cuda/include/cudnn.h"
    "/usr/local/cuda/include/cudnn_version.h"
    "$CONDA_PREFIX/include/cudnn.h"
    "/usr/include/x86_64-linux-gnu/cudnn_v*.h"
)

cudnn_found=false
for loc in "${cudnn_locations[@]}"; do
    if [ -f "$loc" ]; then
        echo "✅ cuDNN header found at: $loc"
        cudnn_found=true
        # Try to get version
        if grep -q "CUDNN_MAJOR" "$loc" 2>/dev/null; then
            major=$(grep "CUDNN_MAJOR" "$loc" | head -1 | awk '{print $3}')
            echo "   Version: $major.x"
        fi
    fi
done

if [ "$cudnn_found" = false ]; then
    echo "❌ cuDNN headers not found in standard locations"
fi

# Check environment variables
echo -e "\n3. Checking environment variables..."
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "CUDNN_HOME: ${CUDNN_HOME:-not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"

# Provide solutions
echo -e "\n============================================================"
echo "SOLUTIONS:"
echo "============================================================"

echo -e "\nOption 1: Install cuDNN development headers"
echo "# For Ubuntu/Debian:"
echo "sudo apt-get update"
echo "sudo apt-get install libcudnn8-dev"
echo ""
echo "# Or download from NVIDIA:"
echo "# https://developer.nvidia.com/cudnn"

echo -e "\nOption 2: Use conda to install cuDNN"
echo "conda install -c conda-forge cudnn"

echo -e "\nOption 3: Set environment variables (if cuDNN is installed elsewhere)"
echo "export CUDA_HOME=/usr/local/cuda"
echo "export CUDNN_HOME=/usr/local/cuda"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"

echo -e "\nOption 4: Try pip install with --no-build-isolation"
echo "pip install transformer-engine[pytorch] --no-build-isolation"

echo -e "\nOption 5: Use pre-built wheel (recommended)"
echo "# This avoids compilation entirely"
cat << 'EOF'
python -c "
import torch
cuda_version = torch.version.cuda.replace('.', '')
wheel_url = f'https://github.com/NVIDIA/TransformerEngine/releases/download/v1.7/transformer_engine-1.7.0+{cuda_version}-cp311-cp311-linux_x86_64.whl'
print(f'pip install {wheel_url}')
"
EOF

echo -e "\n============================================================"
echo "Quick Fix Script:"
echo "============================================================"
cat << 'EOF'
#!/bin/bash
# Save this as install_te_fixed.sh

# Try conda install first
echo "Installing cuDNN via conda..."
conda install -c conda-forge cudnn -y

# Set environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDNN_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib:$LD_LIBRARY_PATH

# Try installation again
echo "Installing Transformer Engine..."
pip install transformer-engine[pytorch] --no-build-isolation

# Verify
python -c "import transformer_engine.pytorch as te; print(f'✅ TE {te.__version__} installed!')"
EOF
