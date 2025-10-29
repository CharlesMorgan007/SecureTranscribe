#!/bin/bash

# SecureTranscribe cuDNN Library Path Fix
# Fixes "Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}"
# This script ensures cuDNN libraries are properly accessible to PyTorch

set -e

echo "🔧 SecureTranscribe cuDNN Library Path Fix"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "app" ]; then
    echo "❌ Error: Please run this script from the SecureTranscribe root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment 'venv' not found"
    exit 1
fi

echo "✅ Found virtual environment: $(pwd)/venv"

# Determine cuDNN library path
CUDNN_PATH="$(pwd)/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

if [ ! -d "$CUDNN_PATH" ]; then
    echo "❌ Error: cuDNN library directory not found at $CUDNN_PATH"
    echo "Searching for cuDNN libraries..."
    find "$(pwd)/venv" -name "libcudnn*" -type f | head -5
    exit 1
fi

echo "✅ Found cuDNN library directory: $CUDNN_PATH"

# Check if required libraries exist
REQUIRED_LIBS=("libcudnn_ops.so" "libcudnn_ops.so.9" "libcudnn_adv.so" "libcudnn.so")
MISSING_LIBS=()

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$CUDNN_PATH/$lib" ]; then
        echo "✅ Found: $lib"
    else
        echo "❌ Missing: $lib"
        MISSING_LIBS+=("$lib")
    fi
done

# List available libraries
echo ""
echo "📋 Available cuDNN libraries:"
ls -la "$CUDNN_PATH"/libcudnn* 2>/dev/null || echo "No cuDNN libraries found"

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: Some required libraries are missing"
    echo "This might indicate an incomplete PyTorch installation"
fi

# Fix 1: Update LD_LIBRARY_PATH
echo ""
echo "🔧 Fix 1: Setting LD_LIBRARY_PATH"

# Remove any existing cuDNN paths from LD_LIBRARY_PATH
CURRENT_LD_PATH="$LD_LIBRARY_PATH"
CLEAN_LD_PATH=$(echo "$CURRENT_LD_PATH" | sed 's|[^:]*nvidia/cudnn/lib[^:]*||g' | sed 's|::|:|g' | sed 's|^:||' | sed 's:$::')

# Add our cuDNN path
if [ -n "$CLEAN_LD_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH:$CLEAN_LD_PATH"
else
    export LD_LIBRARY_PATH="$CUDNN_PATH"
fi

echo "✅ LD_LIBRARY_PATH updated: $LD_LIBRARY_PATH"

# Fix 2: Create missing symlinks if needed
echo ""
echo "🔧 Fix 2: Creating library symlinks"

cd "$CUDNN_PATH"

# Check for libcudnn_ops.so variants
if [ -f "libcudnn_ops.so.9" ] && [ ! -f "libcudnn_ops.so.9.1" ]; then
    ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1
    echo "✅ Created symlink: libcudnn_ops.so.9.1 -> libcudnn_ops.so.9"
fi

if [ -f "libcudnn_ops.so.9.1" ] && [ ! -f "libcudnn_ops.so.9.1.0" ]; then
    ln -sf libcudnn_ops.so.9.1 libcudnn_ops.so.9.1.0
    echo "✅ Created symlink: libcudnn_ops.so.9.1.0 -> libcudnn_ops.so.9.1"
fi

# Ensure basic libcudnn_ops.so exists
if [ -f "libcudnn_ops.so.9" ] && [ ! -f "libcudnn_ops.so" ]; then
    ln -sf libcudnn_ops.so.9 libcudnn_ops.so
    echo "✅ Created symlink: libcudnn_ops.so -> libcudnn_ops.so.9"
fi

cd - > /dev/null

# Fix 3: Test cuDNN loading
echo ""
echo "🧪 Fix 3: Testing cuDNN loading"

# Test with Python in the virtual environment
python -c "
import sys
import os

# Ensure our path is set
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH'

print('Testing cuDNN library access...')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        print(f'✅ CUDA version: {torch.version.cuda}')
        print(f'✅ cuDNN version: {torch.backends.cudnn.version()}')
        print('✅ cuDNN libraries loaded successfully!')
    else:
        print('⚠️  CUDA not available (CPU mode only)')

except Exception as e:
    print(f'❌ Error testing cuDNN: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 cuDNN path fix successful!"
else
    echo ""
    echo "❌ cuDNN test failed. Trying alternative fix..."

    # Alternative: Force CPU mode as fallback
    echo ""
    echo "🔧 Fallback: Force CPU mode"

    export CUDA_VISIBLE_DEVICES=""
    export TORCH_CUDA_ARCH_LIST=""

    python -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA disabled - running in CPU mode')
print('✅ Ready for SecureTranscribe (CPU-only)')
"
fi

# Create startup script
echo ""
echo "🚀 Creating startup script..."

cat > start_with_cudnn_fix.sh << 'EOF'
#!/bin/bash

# SecureTranscribe startup with cuDNN path fix
# Run this script to start the application with proper cuDNN library paths

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set cuDNN library path
CUDNN_PATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

# Update LD_LIBRARY_PATH
CURRENT_LD_PATH="$LD_LIBRARY_PATH"
CLEAN_LD_PATH=$(echo "$CURRENT_LD_PATH" | sed 's|[^:]*nvidia/cudnn/lib[^:]*||g' | sed 's|::|:|g' | sed 's|^:||' | sed 's:$::')

if [ -n "$CLEAN_LD_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH:$CLEAN_LD_PATH"
else
    export LD_LIBRARY_PATH="$CUDNN_PATH"
fi

echo "🔒 cuDNN library path set: $CUDNN_PATH"
echo "🚀 Starting SecureTranscribe..."

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
EOF

chmod +x start_with_cudnn_fix.sh

echo ""
echo "✅ Created startup script: start_with_cudnn_fix.sh"

# Create CPU fallback script
cat > start_cpu_fallback.sh << 'EOF'
#!/bin/bash

# SecureTranscribe CPU-only fallback
# Use this if GPU/cuDNN issues persist

echo "🔒 Forcing CPU mode (disabling CUDA)"
echo "🚀 Starting SecureTranscribe in CPU mode..."

# Disable CUDA
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST=""
export PYTORCH_CUDA_ALLOC_CONF=""

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
EOF

chmod +x start_cpu_fallback.sh

echo "✅ Created CPU fallback script: start_cpu_fallback.sh"

echo ""
echo "🎯 Fix Complete!"
echo ""
echo "📋 How to start the application:"
echo "1. With cuDNN fix: ./start_with_cudnn_fix.sh"
echo "2. CPU fallback:   ./start_cpu_fallback.sh"
echo ""
echo "📋 What this fix does:"
echo "   • Adds cuDNN library directory to LD_LIBRARY_PATH"
echo "   • Creates missing library symlinks if needed"
echo "   • Tests cuDNN loading before starting"
echo "   • Provides CPU fallback option"
echo ""
echo "💡 If you still see cuDNN errors:"
echo "   - Use the CPU fallback script"
echo "   - Check if NVIDIA drivers are installed: nvidia-smi"
echo "   - Verify virtual environment activation"
echo ""
echo "✅ Ready to start SecureTranscribe!"
