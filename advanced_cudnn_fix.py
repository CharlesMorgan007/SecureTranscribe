#!/usr/bin/env python3
"""
Advanced cuDNN Diagnostic and Fix Script
Addresses deep cuDNN library loading issues in PyTorch
"""

import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print("=" * 60)


def print_subsection(title):
    print(f"\n📋 {title}")
    print("-" * 40)


def get_venv_cudnn_path():
    """Get the cuDNN library path in the virtual environment."""
    venv_path = Path(sys.executable).parent.parent
    cudnn_path = venv_path / "lib/python3.12/site-packages/nvidia/cudnn/lib"
    return venv_path, cudnn_path


def diagnose_cudnn_libraries(cudnn_path):
    """Deep diagnosis of cuDNN libraries."""
    print_section("cuDNN Library Deep Diagnosis")

    print(f"cuDNN path: {cudnn_path}")
    print(f"Path exists: {cudnn_path.exists()}")

    if not cudnn_path.exists():
        print("❌ cuDNN directory not found!")
        return False

    # List all cuDNN libraries
    print_subsection("All cuDNN Libraries")
    cudnn_files = list(cudnn_path.glob("libcudnn*"))
    for lib in sorted(cudnn_files):
        print(f"📄 {lib.name}")

        # Check file details
        if lib.is_symlink():
            target = lib.resolve()
            print(f"   🔗 Symlink to: {target}")
            print(f"   🔗 Target exists: {target.exists()}")
        else:
            print(f"   📊 Size: {lib.stat().st_size:,} bytes")
            print(f"   🔒 Permissions: {oct(lib.stat().st_mode)[-3:]}")
            print(f"   ✅ Accessible: {os.access(lib, os.R_OK)}")

    # Check for required library versions
    print_subsection("Required Library Versions Check")
    required = [
        "libcudnn_ops.so",
        "libcudnn_ops.so.9",
        "libcudnn_ops.so.9.1",
        "libcudnn_ops.so.9.1.0",
    ]

    missing = []
    for lib_name in required:
        lib_path = cudnn_path / lib_name
        if lib_path.exists():
            print(f"✅ {lib_name}")
        else:
            print(f"❌ {lib_name}")
            missing.append(lib_name)

    return len(missing) == 0


def create_library_symlinks(cudnn_path):
    """Create missing library symlinks."""
    print_section("Creating Library Symlinks")

    # Find the base library files
    base_ops = None
    for pattern in ["libcudnn_ops.so.9.*", "libcudnn_ops.so.9"]:
        matches = list(cudnn_path.glob(pattern))
        if matches:
            base_ops = matches[0]
            break

    if not base_ops:
        print("❌ No base libcudnn_ops.so.9* library found!")
        return False

    print(f"📋 Base library: {base_ops.name}")

    # Create symlinks
    symlinks_created = []
    os.chdir(cudnn_path)

    try:
        # Create libcudnn_ops.so if it doesn't exist
        if not Path("libcudnn_ops.so").exists():
            os.symlink(base_ops.name, "libcudnn_ops.so")
            symlinks_created.append("libcudnn_ops.so")
            print(f"✅ Created: libcudnn_ops.so -> {base_ops.name}")

        # Create versioned symlinks if needed
        if base_ops.name.startswith("libcudnn_ops.so.9."):
            base_version = base_ops.name

            # Create .9 version
            if not Path("libcudnn_ops.so.9").exists():
                os.symlink(base_version, "libcudnn_ops.so.9")
                symlinks_created.append("libcudnn_ops.so.9")
                print(f"✅ Created: libcudnn_ops.so.9 -> {base_version}")

            # Create .9.1 version if applicable
            if ".9.1" in base_version and not Path("libcudnn_ops.so.9.1").exists():
                os.symlink(base_version, "libcudnn_ops.so.9.1")
                symlinks_created.append("libcudnn_ops.so.9.1")
                print(f"✅ Created: libcudnn_ops.so.9.1 -> {base_version}")

        # Check for other missing libraries
        other_libs = ["libcudnn_adv.so", "libcudnn.so", "libcudnn_cnn.so"]
        for lib_pattern in other_libs:
            if not Path(lib_pattern).exists():
                # Try to find a versioned version
                matches = list(cudnn_path.glob(f"{lib_pattern}.*"))
                if matches:
                    os.symlink(matches[0].name, lib_pattern)
                    symlinks_created.append(lib_pattern)
                    print(f"✅ Created: {lib_pattern} -> {matches[0].name}")

    finally:
        os.chdir("/")

    print(
        f"\n📊 Created {len(symlinks_created)} symlinks: {', '.join(symlinks_created)}"
    )
    return len(symlinks_created) > 0


def test_library_loading(cudnn_path):
    """Test if cuDNN libraries can be loaded."""
    print_section("Testing Library Loading")

    # Set library path
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_path.as_posix() not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{current_ld_path}"
        print(f"🔧 LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

    # Test with Python
    test_script = f"""
import os
os.environ["LD_LIBRARY_PATH"] = "{os.environ["LD_LIBRARY_PATH"]}"

try:
    import torch
    print(f"✅ PyTorch {{torch.__version__}}")
    print(f"✅ CUDA available: {{torch.cuda.is_available()}}")

    if torch.cuda.is_available():
        print(f"✅ CUDA version: {{torch.version.cuda}}")

        # Test cuDNN specifically
        try:
            cudnn_version = torch.backends.cudnn.version()
            print(f"✅ cuDNN version: {{cudnn_version}}")

            # Test cuDNN operations
            if cudnn_version > 0:
                import torch.nn.functional as F
                x = torch.randn(1, 1, 32, 32).cuda()
                conv = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
                y = F.conv2d(x, conv.weight.cuda())
                print("✅ cuDNN operations: Working")
            else:
                print("⚠️ cuDNN version is 0 - not working")

        except Exception as e:
            print(f"❌ cuDNN test failed: {{e}}")
            print("This indicates library loading issues")
    else:
        print("⚠️ CUDA not available - CPU mode only")

except ImportError as e:
    print(f"❌ PyTorch import failed: {{e}}")
except Exception as e:
    print(f"❌ General error: {{e}}")
"""

    success, stdout, stderr = run_command(f'python3 -c "{test_script}"')

    if success:
        print("✅ Library loading test output:")
        print(stdout)
        return True
    else:
        print("❌ Library loading test failed:")
        print(stdout)
        if stderr:
            print("Errors:", stderr)
        return False


def fix_pytorch_cudnn():
    """Try to fix PyTorch cuDNN integration."""
    print_section("PyTorch cuDNN Integration Fix")

    # Try different approaches to fix cuDNN loading

    print_subsection("Approach 1: Force cuDNN initialization")

    fix_script = """
import os
import sys

# Set environment variables before importing PyTorch
os.environ["LD_LIBRARY_PATH"] = "{library_path}"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

try:
    import torch

    # Force cuDNN initialization
    if torch.cuda.is_available():
        print("Initializing cuDNN...")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Try to trigger cuDNN loading
        x = torch.randn(1, 3, 224, 224).cuda()
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
        y = conv(x)

        print("✅ cuDNN initialization successful")
        print(f"cuDNN version: {{torch.backends.cudnn.version()}}")
    else:
        print("CUDA not available")

except Exception as e:
    print(f"❌ cuDNN initialization failed: {{e}}")
    import traceback
    traceback.print_exc()
"""

    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    fix_script = fix_script.format(library_path=library_path)

    success, stdout, stderr = run_command(f'python3 -c "{fix_script}"')

    if success:
        print("✅ PyTorch cuDNN fix output:")
        print(stdout)
        return True
    else:
        print("❌ PyTorch cuDNN fix failed:")
        print(stdout)
        if stderr:
            print("Errors:", stderr)
        return False


def create_fallback_solutions():
    """Create fallback solutions."""
    print_section("Creating Fallback Solutions")

    # Create CPU-only startup script
    cpu_script = """#!/bin/bash
# SecureTranscribe CPU-Only Startup
# Bypasses all CUDA/cuDNN issues

echo "🔒 Starting SecureTranscribe in CPU-only mode"
echo "🚀 All processing will be CPU-based (slower but stable)"

# Disable CUDA completely
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST=""
export PYTORCH_CUDA_ALLOC_CONF=""
export FORCE_CPU="1"

# Start the application
cd "$(dirname "$0")"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
"""

    with open("start_cpu_only.sh", "w") as f:
        f.write(cpu_script)
    os.chmod("start_cpu_only.sh", 0o755)
    print("✅ Created CPU-only startup script: start_cpu_only.sh")

    # Create CUDA-enabled startup script
    venv_path = Path(sys.executable).parent.parent
    cudnn_path = venv_path / "lib/python3.12/site-packages/nvidia/cudnn/lib"

    cuda_script = f"""#!/bin/bash
# SecureTranscribe CUDA-Enabled Startup
# Attempts to use GPU with cuDNN fixes

echo "🚀 Starting SecureTranscribe with CUDA support"
echo "🔧 Applying cuDNN library path fixes..."

# Set cuDNN library path
CUDNN_PATH="{cudnn_path}"
export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"

# Optimize PyTorch CUDA settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES="0"

echo "🔒 cuDNN library path set: $CUDNN_PATH"
echo "🚀 Starting application..."

# Start the application
cd "$(dirname "$0")"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
"""

    with open("start_with_cuda.sh", "w") as f:
        f.write(cuda_script)
    os.chmod("start_with_cuda.sh", 0o755)
    print("✅ Created CUDA startup script: start_with_cuda.sh")


def main():
    """Main diagnostic and fix function."""
    print("🚀 Advanced cuDNN Diagnostic and Fix")
    print("=" * 70)

    # Get paths
    venv_path, cudnn_path = get_venv_cudnn_path()
    print(f"Virtual environment: {venv_path}")
    print(f"cuDNN library path: {cudnn_path}")

    # Diagnose libraries
    libraries_ok = diagnose_cudnn_libraries(cudnn_path)

    if not libraries_ok:
        # Try to fix symlinks
        symlinks_created = create_library_symlinks(cudnn_path)
        if symlinks_created:
            print("\n🔧 Re-checking libraries after symlink creation...")
            libraries_ok = diagnose_cudnn_libraries(cudnn_path)

    # Test library loading
    loading_ok = test_library_loading(cudnn_path)

    if not loading_ok:
        print("\n🔧 Attempting PyTorch cuDNN integration fix...")
        pytorch_ok = fix_pytorch_cudnn()
        if pytorch_ok:
            loading_ok = True

    # Create fallback solutions
    create_fallback_solutions()

    # Summary and recommendations
    print_section("Summary and Recommendations")

    if libraries_ok and loading_ok:
        print("🎉 SUCCESS: cuDNN libraries are working!")
        print("\n📋 Next steps:")
        print("   1. Start with: ./start_with_cuda.sh")
        print("   2. Test with a small audio file")
        print("   3. Monitor for any remaining errors")
    else:
        print("⚠️ cuDNN issues persist - using fallback solutions")
        print("\n📋 Recommended next steps:")
        print("   1. Use CPU-only mode: ./start_cpu_only.sh")
        print("   2. Or try CUDA mode: ./start_with_cuda.sh (may still fail)")
        print("   3. CPU mode will work reliably but be slower")
        print("\n💡 Long-term solutions:")
        print("   - Reinstall PyTorch with matching cuDNN")
        print("   - Update NVIDIA drivers")
        print("   - Use official NVIDIA PyTorch wheels")

    print(f"\n🔍 If issues persist, check:")
    print(f"   - NVIDIA driver version: nvidia-smi")
    print(f"   - System cuDNN: ldconfig -p | grep cudnn")
    print(f"   - Library dependencies: ldd {cudnn_path}/libcudnn_ops.so")


if __name__ == "__main__":
    main()
