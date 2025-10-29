#!/usr/bin/env python3
"""
Working cuDNN Test Script
Tests cuDNN library functionality without syntax errors
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def print_section(title):
    print("\n" + "=" * 60)
    print("🔍 " + title)
    print("=" * 60)


def print_subsection(title):
    print("\n📋 " + title)
    print("-" * 40)


def get_cudnn_path():
    """Get cuDNN library path."""
    venv_path = Path(sys.executable).parent.parent
    cudnn_path = venv_path / "lib/python3.12/site-packages/nvidia/cudnn/lib"
    return venv_path, cudnn_path


def check_cudnn_libraries(cudnn_path):
    """Check cuDNN libraries."""
    print_section("cuDNN Library Check")

    print("cuDNN path:", cudnn_path)
    print("Exists:", cudnn_path.exists())

    if not cudnn_path.exists():
        print("❌ cuDNN directory not found!")
        return False

    # List libraries
    print_subsection("Available Libraries")
    cudnn_files = list(cudnn_path.glob("libcudnn*"))
    for lib in sorted(cudnn_files):
        if lib.is_symlink():
            target = lib.resolve()
            print("📄 " + lib.name + " -> " + target.name)
        else:
            size_mb = lib.stat().st_size / (1024 * 1024)
            print(f"📄 {lib.name} ({size_mb:.1f} MB)")

    # Check required versions
    print_subsection("Required Versions")
    required = [
        "libcudnn_ops.so",
        "libcudnn_ops.so.9",
        "libcudnn_ops.so.9.1",
        "libcudnn_ops.so.9.1.0",
    ]
    all_present = True

    for lib_name in required:
        lib_path = cudnn_path / lib_name
        if lib_path.exists():
            print("✅ " + lib_name)
        else:
            print("❌ " + lib_name)
            all_present = False

    return all_present


def test_pytorch_cuda():
    """Test PyTorch CUDA functionality."""
    print_section("PyTorch CUDA Test")

    # Create a temporary test file
    test_file = "/tmp/test_cuda.py"
    test_code = """
import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

        # Test cuDNN operations
        try:
            import torch.nn.functional as F
            x = torch.randn(1, 1, 32, 32).cuda()
            conv = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
            y = F.conv2d(x, conv.weight.cuda())
            print("✅ cuDNN operations working")
            sys.exit(0)
        except Exception as e:
            print(f"❌ cuDNN operations failed: {e}")
            sys.exit(1)
    else:
        print("⚠️ CUDA not available")
        sys.exit(0)

except Exception as e:
    print(f"❌ PyTorch test failed: {e}")
    sys.exit(1)
"""

    # Write test file
    with open(test_file, "w") as f:
        f.write(test_code)

    # Run test
    success, stdout, stderr = run_command(f"python3 {test_file}")

    print("Test output:")
    print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)

    # Clean up
    try:
        os.remove(test_file)
    except:
        pass

    return success


def test_direct_library_loading(cudnn_path):
    """Test direct library loading."""
    print_section("Direct Library Loading Test")

    # Set LD_LIBRARY_PATH
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_path = (
        str(cudnn_path) + ":" + current_ld_path if current_ld_path else str(cudnn_path)
    )
    os.environ["LD_LIBRARY_PATH"] = new_ld_path

    print(f"LD_LIBRARY_PATH: {new_ld_path}")

    # Test with ldd
    lib_path = cudnn_path / "libcudnn_ops.so"
    if lib_path.exists():
        print_subsection("Library Dependencies")
        success, stdout, stderr = run_command(f"ldd {lib_path}")
        if success:
            print(stdout[:500])  # First 500 chars
        else:
            print(f"ldd failed: {stderr}")

    # Create ctypes test
    ctypes_file = "/tmp/test_ctypes.py"
    ctypes_code = f'''
import os
import sys
os.environ["LD_LIBRARY_PATH"] = "{new_ld_path}"

try:
    import ctypes
    lib_path = "{cudnn_path}/libcudnn_ops.so"
    lib = ctypes.CDLL(lib_path)
    print(f"✅ Successfully loaded: {lib_path}")

    # Test if we can access some functions
    try:
        # Try a common cuDNN function
        func = getattr(lib, 'cudnnCreateTensorDescriptor', None)
        if func:
            print("✅ cuDNN symbols available")
            sys.exit(0)
        else:
            print("⚠️ cuDNN CreateTensorDescriptor not found")
            sys.exit(1)
    except Exception as e:
        print(f"❌ cuDNN symbol access failed: {e}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Library loading failed: {e}")
    sys.exit(1)
'''

    # Write ctypes test
    with open(ctypes_file, "w") as f:
        f.write(ctypes_code)

    # Run ctypes test
    success, stdout, stderr = run_command(f"python3 {ctypes_file}")

    print("Direct loading test:")
    print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)

    # Clean up
    try:
        os.remove(ctypes_file)
    except:
        pass

    return success


def create_startup_scripts(cudnn_path):
    """Create startup scripts."""
    print_section("Creating Startup Scripts")

    # CPU-only script
    cpu_script = """#!/bin/bash
echo "🔒 Starting SecureTranscribe in CPU-only mode"
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST=""
export PYTORCH_CUDA_ALLOC_CONF=""
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
"""

    with open("start_cpu_only.sh", "w") as f:
        f.write(cpu_script)
    os.chmod("start_cpu_only.sh", 0o755)
    print("✅ Created: start_cpu_only.sh")

    # CUDA-enabled script
    cuda_script = f'''#!/bin/bash
echo "🚀 Starting SecureTranscribe with CUDA support"
export CUDA_VISIBLE_DEVICES="0"
export LD_LIBRARY_PATH="{cudnn_path}:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
'''

    with open("start_with_cuda.sh", "w") as f:
        f.write(cuda_script)
    os.chmod("start_with_cuda.sh", 0o755)
    print("✅ Created: start_with_cuda.sh")

    # Test script
    test_script = """#!/bin/bash
echo "🧪 Testing cuDNN setup..."
export LD_LIBRARY_PATH="/home/cmorgan/Devel/Personal/SecureTranscribe/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    # Test simple tensor
    x = torch.randn(10, 10).cuda()
    y = x + x
    print('✅ CUDA operations working')
else:
    print('⚠️ CPU mode only')
"
"""

    with open("test_cudnn.sh", "w") as f:
        f.write(test_script)
    os.chmod("test_cudnn.sh", 0o755)
    print("✅ Created: test_cudnn.sh")


def main():
    """Main function."""
    print("🚀 Working cuDNN Test and Fix")
    print("=" * 70)

    # Get paths
    venv_path, cudnn_path = get_cudnn_path()
    print(f"Virtual environment: {venv_path}")
    print(f"cuDNN path: {cudnn_path}")

    # Check libraries
    libs_ok = check_cudnn_libraries(cudnn_path)

    # Test PyTorch
    pytorch_ok = test_pytorch_cuda()

    # Test direct loading
    if not pytorch_ok:
        direct_ok = test_direct_library_loading(cudnn_path)
    else:
        direct_ok = True

    # Create startup scripts
    create_startup_scripts(cudnn_path)

    # Summary
    print_section("Summary")

    if libs_ok and pytorch_ok:
        print("🎉 SUCCESS: cuDNN is working!")
        print("\n📋 Next steps:")
        print("1. Run: ./start_with_cuda.sh")
        print("2. Test with audio file")
    elif libs_ok and direct_ok:
        print("⚠️ PARTIAL SUCCESS: Libraries load but PyTorch has issues")
        print("\n📋 Next steps:")
        print("1. Try: ./start_with_cuda.sh (may work)")
        print("2. Or use: ./start_cpu_only.sh")
    else:
        print("❌ ISSUES DETECTED: Using CPU fallback")
        print("\n📋 Next steps:")
        print("1. Use: ./start_cpu_only.sh (reliable)")
        print("2. Or try: ./start_with_cuda.sh (may fail)")

    print("\n💡 To test cuDNN setup:")
    print("   ./test_cudnn.sh")

    print("\n🔍 Additional diagnostics:")
    print("   nvidia-smi  # Check NVIDIA driver")
    print("   ldconfig -p | grep cudnn  # Check system cuDNN")


if __name__ == "__main__":
    main()
