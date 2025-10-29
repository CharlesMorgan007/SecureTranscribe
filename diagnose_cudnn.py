#!/usr/bin/env python3
"""
SecureTranscribe cuDNN Library Diagnosis Script
Diagnoses cuDNN library availability and path issues
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print("=" * 60)


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n📋 {title}")
    print("-" * 40)


def check_virtual_environment():
    """Check if we're in a virtual environment."""
    print_section("Virtual Environment Check")

    print(f"Python executable: {sys.executable}")
    print(f"Virtual environment: {os.getenv('VIRTUAL_ENV', 'Not detected')}")
    print(f"Python version: {sys.version}")

    venv_path = Path(sys.executable).parent.parent
    print(f"Detected venv path: {venv_path}")
    print(f"Venv exists: {venv_path.exists()}")

    return venv_path


def find_cudnn_libraries(venv_path):
    """Find cuDNN libraries in the virtual environment."""
    print_section("cuDNN Library Search")

    # Search patterns from the error message
    search_patterns = [
        "libcudnn_ops.so.9.1.0",
        "libcudnn_ops.so.9.1",
        "libcudnn_ops.so.9",
        "libcudnn_ops.so",
    ]

    found_libraries = {}

    for pattern in search_patterns:
        print_subsection(f"Searching for: {pattern}")

        # Find all matching files
        cmd = f"find {venv_path} -name '{pattern}' 2>/dev/null"
        success, stdout, stderr = run_command(cmd)

        if success and stdout.strip():
            found_files = stdout.strip().split("\n")
            found_libraries[pattern] = found_files
            print(f"✅ Found {len(found_files)} file(s):")
            for file_path in found_files:
                print(f"   {file_path}")
                # Check if file exists and is readable
                file_obj = Path(file_path)
                if file_obj.exists():
                    print(f"   ✅ Exists, Size: {file_obj.stat().st_size} bytes")
                    # Check if it's a symlink
                    if file_obj.is_symlink():
                        target = file_obj.resolve()
                        print(f"   🔗 Symlink to: {target}")
                        print(f"   🔗 Target exists: {target.exists()}")
                else:
                    print(f"   ❌ File not accessible")
        else:
            found_libraries[pattern] = []
            print(f"❌ Not found")
            if stderr:
                print(f"   Error: {stderr}")

    return found_libraries


def check_library_paths(venv_path):
    """Check system library paths."""
    print_section("Library Path Environment")

    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"LD_LIBRARY_PATH: {ld_path or 'Not set'}")

    if ld_path:
        paths = ld_path.split(":")
        for i, path in enumerate(paths):
            path_obj = Path(path)
            print(f"  [{i + 1}] {path} - {'✅' if path_obj.exists() else '❌'}")

    # Check Python's library search paths
    print_subsection("Python Library Paths")
    for path in sys.path:
        path_obj = Path(path)
        print(f"  {path} - {'✅' if path_obj.exists() else '❌'}")

    # Check for cuDNN in common locations
    common_paths = [
        venv_path / "lib",
        venv_path / "lib64",
        venv_path / "lib/python3.12/site-packages/nvidia/cudnn/lib",
    ]

    print_subsection("cuDNN-Specific Paths")
    for path in common_paths:
        print(f"  {path} - {'✅' if path.exists() else '❌'}")
        if path.exists():
            cudnn_files = list(path.glob("libcudnn*"))
            if cudnn_files:
                print(f"    Found {len(cudnn_files)} cuDNN files")
                for lib in cudnn_files[:5]:  # Show first 5
                    print(f"      {lib.name}")


def check_torch_cuda_status():
    """Check PyTorch CUDA status and cuDNN version."""
    print_section("PyTorch CUDA Status")

    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✅ CUDA devices: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

            # Test basic CUDA operations
            try:
                x = torch.randn(3, 3).cuda()
                y = torch.randn(3, 3).cuda()
                z = x + y
                print("✅ Basic CUDA operations: Working")
            except Exception as e:
                print(f"❌ Basic CUDA operations: Failed - {e}")

        else:
            print("❌ CUDA not available")

    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    except Exception as e:
        print(f"❌ PyTorch CUDA check failed: {e}")


def test_cudnn_imports():
    """Test cuDNN-related imports."""
    print_section("cuDNN Import Tests")

    # Test nvidia.cudnn
    try:
        import nvidia.cudnn

        print("✅ nvidia.cudnn import successful")
        print(f"   Path: {nvidia.cudnn.__file__}")
    except Exception as e:
        print(f"❌ nvidia.cudnn import failed: {e}")

    # Test torch.backends.cudnn
    try:
        import torch

        cudnn = torch.backends.cudnn
        print(f"✅ torch.backends.cudnn available")
        print(f"   Version: {cudnn.version()}")
        print(f"   Enabled: {cudnn.enabled}")
    except Exception as e:
        print(f"❌ torch.backends.cudnn check failed: {e}")

    # Test faster-whisper CUDA
    try:
        from faster_whisper import WhisperModel

        print("✅ faster-whisper import successful")

        # Try to create a tiny model to test CUDA
        try:
            print("📋 Testing Whisper model creation with CUDA...")
            if torch.cuda.is_available():
                model = WhisperModel("tiny", device="cuda", compute_type="float16")
                print("✅ Whisper CUDA model creation: Successful")
            else:
                print("⚠️  Skipping Whisper CUDA test (CUDA not available)")
        except Exception as e:
            print(f"❌ Whisper CUDA model creation failed: {e}")

    except Exception as e:
        print(f"❌ faster-whisper import failed: {e}")


def diagnose_path_issues():
    """Diagnose potential path resolution issues."""
    print_section("Path Resolution Diagnosis")

    venv_path = Path(sys.executable).parent.parent

    # Check if cuDNN libraries are findable by the dynamic linker
    print_subsection("Dynamic Library Resolution")

    cudnn_libs = list(venv_path.rglob("libcudnn_ops.so*"))
    for lib_path in cudnn_libs[:3]:  # Check first 3
        print(f"\n🔍 Testing library: {lib_path}")
        print(f"   Exists: {lib_path.exists()}")
        print(f"   Readable: {os.access(lib_path, os.R_OK)}")

        if lib_path.exists():
            # Try ldd to see dependencies
            cmd = f"ldd {lib_path} 2>/dev/null | head -5"
            success, stdout, stderr = run_command(cmd)
            if success and stdout:
                print("   Dependencies:")
                for line in stdout.split("\n")[:5]:
                    print(f"     {line}")

        # Check if it's in LD_LIBRARY_PATH
        lib_dir = lib_path.parent
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir in current_ld_path:
            print(f"   ✅ Directory in LD_LIBRARY_PATH")
        else:
            print(f"   ⚠️  Directory NOT in LD_LIBRARY_PATH")


def generate_recommendations(found_libraries, venv_path):
    """Generate recommendations based on findings."""
    print_section("Recommendations")

    # Check what's missing
    missing_patterns = []
    for pattern, files in found_libraries.items():
        if not files:
            missing_patterns.append(pattern)

    if missing_patterns:
        print("❌ Missing cuDNN libraries:")
        for pattern in missing_patterns:
            print(f"   - {pattern}")

        print("\n🛠️  Recommended fixes:")
        print("1. Update LD_LIBRARY_PATH to include cuDNN library directory:")
        cuddn_path = venv_path / "lib/python3.12/site-packages/nvidia/cudnn/lib"
        if cuddn_path.exists():
            print(f"   export LD_LIBRARY_PATH={cuddn_path}:$LD_LIBRARY_PATH")

        print("\n2. Create missing symlinks if needed:")
        print(
            "   cd $(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cudnn/lib"
        )
        print("   ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1")
        print("   ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1.0")

        print("\n3. Quick fix - Disable CUDA to bypass the issue:")
        print("   export CUDA_VISIBLE_DEVICES=''")
        print("   export TORCH_CUDA_ARCH_LIST=''")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")

    else:
        print("✅ All cuDNN libraries found!")
        print("\nIf CUDA still fails, the issue might be:")
        print("1. LD_LIBRARY_PATH not including the cuDNN directory")
        print("2. Permission issues with library files")
        print("3. Version mismatch between PyTorch and cuDNN")
        print("4. Broken symlinks in the cuDNN directory")


def main():
    """Main diagnosis function."""
    print("🚀 SecureTranscribe cuDNN Library Diagnosis")
    print("=" * 70)

    # Check virtual environment
    venv_path = check_virtual_environment()

    # Find cuDNN libraries
    found_libraries = find_cudnn_libraries(venv_path)

    # Check library paths
    check_library_paths(venv_path)

    # Check PyTorch CUDA status
    check_torch_cuda_status()

    # Test imports
    test_cudnn_imports()

    # Diagnose path issues
    diagnose_path_issues()

    # Generate recommendations
    generate_recommendations(found_libraries, venv_path)

    print("\n" + "=" * 70)
    print("🎯 Diagnosis Complete")
    print("\n💡 If you need immediate help, run:")
    print("   export CUDA_VISIBLE_DEVICES=''")
    print("   export TORCH_CUDA_ARCH_LIST=''")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")


if __name__ == "__main__":
    main()
