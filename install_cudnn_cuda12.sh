#!/bin/bash

# Simplified cuDNN Installation for CUDA 12.8 on Ubuntu
# Fixes cuDNN library issues for SecureTranscribe with RTX 4090

set -e

echo "🔧 cuDNN Installation for CUDA 12.8"
echo "===================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Verify CUDA 12.x is available
echo "🔍 Verifying CUDA 12.x installation..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found"
    echo "Install with: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "✅ CUDA version: $CUDA_VERSION"

if [[ $CUDA_VERSION != 12.* ]]; then
    echo "⚠️  This script is optimized for CUDA 12.x"
    echo "   Your version: $CUDA_VERSION"
fi

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA driver not found"
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "✅ NVIDIA Driver: $DRIVER_VERSION"

# Determine cuDNN version to download
CUDNN_VERSION="8.9.7"
echo "📋 Using cuDNN $CUDNN_VERSION (compatible with CUDA 12.x)"

# Download cuDNN
CUDNN_TAR="cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"

if [ ! -f "$CUDNN_TAR" ]; then
    echo ""
    echo "📥 cuDNN Download Required"
    echo "========================"
    echo "cuDNN $CUDNN_VERSION must be downloaded manually:"
    echo ""
    echo "1. Go to: https://developer.nvidia.com/cudnn"
    echo "2. Login or create free NVIDIA Developer account"
    echo "3. Download: cuDNN v8.9.7 for CUDA 12.x"
    echo "   File: cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"
    echo "4. Upload to: $(pwd)"
    echo ""
    echo "After uploading, run this script again."
    exit 1
fi

echo "✅ cuDNN archive found: $CUDNN_TAR"

# Extract and install
echo ""
echo "📦 Extracting cuDNN..."
tar -xf "$CUDNN_TAR"

# Find extracted directory
CUDNN_DIR=$(find . -maxdepth 1 -type d -name "*cudnn*" | head -1)
CUDNN_DIR=${CUDNN_DIR#./}

if [ ! -d "$CUDNN_DIR" ]; then
    echo "❌ Could not find cuDNN extraction directory"
    exit 1
fi

echo "✅ Extracted to: $CUDNN_DIR"

# Install to system
echo ""
echo "🔧 Installing cuDNN system-wide..."
cd "$CUDNN_DIR"

# Copy headers
echo "   Installing header files..."
sudo cp include/cudnn*.h /usr/local/cuda/include/

# Copy libraries
echo "   Installing library files..."
sudo cp lib/libcudnn* /usr/local/cuda/lib64/

# Set permissions
echo "   Setting permissions..."
sudo chmod 644 /usr/local/cuda/include/cudnn*.h
sudo chmod 755 /usr/local/cuda/lib64/libcudnn*

# Configure system
echo ""
echo "⚙️  Configuring system..."

# Add to dynamic linker
echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf

# Update library cache
sudo ldconfig

# Return to original directory
cd ..

# Clean up extraction
echo "🧹 Cleaning up..."
rm -rf "$CUDNN_DIR"

# Verify installation
echo ""
echo "🧪 Verifying cuDNN installation..."

# Check system library cache
if ldconfig -p | grep -q "libcudnn.so"; then
    echo "✅ cuDNN libraries registered in system:"
    ldconfig -p | grep "libcudnn.so" | head -3
else
    echo "❌ cuDNN libraries not found in system cache"
    echo "Manual fix: sudo ldconfig"
    exit 1
fi

# Test with Python
echo ""
echo "🧪 Testing cuDNN with PyTorch..."

# Activate virtual environment
cd /home/cmorgan/Devel/Personal/SecureTranscribe
source venv/bin/activate

# Run comprehensive test
python3 << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU devices: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

    if torch.backends.cudnn.version() > 0:
        print("✅ cuDNN is working!")

        # Test cuDNN operations
        try:
            import torch.nn.functional as F
            x = torch.randn(1, 3, 32, 32).cuda()
            conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
            y = F.conv2d(x, conv.weight.cuda())
            print("✅ cuDNN operations test passed!")
            print("✅ GPU acceleration ready for SecureTranscribe!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ cuDNN operations test failed: {e}")
            sys.exit(1)
    else:
        print("❌ cuDNN version is 0 - not working")
        sys.exit(1)
else:
    print("⚠️  CUDA not available - CPU mode only")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: cuDNN installation complete!"
    echo ""
    echo "📋 SecureTranscribe is ready with GPU acceleration!"
    echo ""
    echo "🚀 Start SecureTranscribe:"
    echo "   cd /home/cmorgan/Devel/Personal/SecureTranscribe"
    echo "   source venv/bin/activate"
    echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001"
    echo ""
    echo "🧪 Test with audio file to verify GPU processing"
else
    echo ""
    echo "❌ cuDNN test failed"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Reboot system: sudo reboot"
    echo "2. Check NVIDIA driver: nvidia-smi"
    echo "3. Verify CUDA: nvcc --version"
    echo "4. Check cuDNN: ldconfig -p | grep cudnn"
    echo "5. Test manually:"
    echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "   python3 -c \"import torch; print(torch.cuda.is_available())\""
    exit 1
fi

# Clean up downloaded archive
echo ""
echo "🧹 Removing downloaded archive..."
rm -f "$CUDNN_TAR"

echo ""
echo "✅ cuDNN installation process completed!"
