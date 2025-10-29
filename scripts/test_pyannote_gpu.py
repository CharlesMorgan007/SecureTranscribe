#!/usr/bin/env python3
"""
Standalone diagnostic script to test loading the PyAnnote diarization pipeline
and moving it to GPU, with detailed CUDA/cuDNN environment checks.

Usage:
  python SecureTranscribe/scripts/test_pyannote_gpu.py \
      --model pyannote/speaker-diarization-3.1 \
      --device cuda \
      --hf-token YOUR_HF_TOKEN \
      --verbose

Notes:
- If the model is gated on Hugging Face, you must provide a valid token via
  --hf-token or the HUGGINGFACE_TOKEN environment variable.
- This script will:
  1) Print detailed PyTorch/CUDA/cuDNN and NVIDIA environment info.
  2) Validate basic tensor ops and a small Conv2D on the target device.
  3) Load the PyAnnote pipeline.
  4) Attempt to move the pipeline to the target device using torch.device.
  5) Optionally run a tiny inference on a generated 1-second sine wave.
"""

import argparse
import os
import sys
import time
import math
import subprocess
import logging
import traceback

# Optional: reduce noisy warnings (especially from librosa)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*dtype of input is float64.*")

try:
    import torch
except Exception as e:
    print("ERROR: Failed to import torch:", e)
    sys.exit(1)

try:
    from pyannote.audio import Pipeline
except Exception as e:
    print("ERROR: Failed to import pyannote.audio Pipeline:", e)
    sys.exit(1)

try:
    import numpy as np
except Exception as e:
    print("ERROR: Failed to import numpy:", e)
    sys.exit(1)


def run(cmd):
    """Run a shell command and return (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate(timeout=15)
        return proc.returncode, out.strip(), err.strip()
    except Exception as e:
        return 1, "", str(e)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_env_info(verbose: bool = False):
    print_header("PyTorch/CUDA/cuDNN Environment")
    print(f"torch.version: {torch.__version__}")
    print(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
    print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(
        f"torch.cuda.current_device(): {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}"
    )
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")

    print("\nKey environment variables:")
    for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF"]:
        print(f"  {k} = {os.environ.get(k)}")

    # nvidia-smi details if available
    code, out, err = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if code == 0:
        print("\nNVIDIA GPUs (nvidia-smi):")
        for line in out.splitlines():
            print(f"  {line}")
    else:
        code2, out2, err2 = run(["nvidia-smi", "--version"])
        if code2 == 0:
            print("\nNVIDIA SMI version:")
            print("  " + out2.replace("\n", "\n  "))
        else:
            print("\nNVIDIA SMI not available or error retrieving info.")

    if torch.cuda.is_available():
        print("\nCUDA devices (PyTorch):")
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                print(
                    f"  GPU {i}: {props.name}, "
                    f"total_memory={props.total_memory / 1024**3:.1f}GB, "
                    f"compute_capability={props.major}.{props.minor}, "
                    f"multiprocessors={props.multi_processor_count}"
                )
            except Exception as e:
                print(f"  Failed to get properties for device {i}: {e}")

    if verbose:
        print("\nDetailed sys.path (first 10 entries):")
        for p in sys.path[:10]:
            print("  ", p)


def test_torch_device_ops(device_str: str) -> bool:
    """
    Basic device sanity check:
    - create tensors on device
    - run matmul
    - run a small Conv2D op
    Returns True if all pass, else False.
    """
    print_header(f"Testing basic torch ops on device '{device_str}'")
    try:
        dev = torch.device(device_str)
    except Exception as e:
        print(f"ERROR: Invalid device string '{device_str}': {e}")
        return False

    try:
        # Tensor allocation & matmul
        a = torch.randn(1024, 1024, device=dev)
        b = torch.randn(1024, 1024, device=dev)
        t0 = time.time()
        c = a @ b
        _ = c.mean().item()
        t1 = time.time()
        print(f"Matmul (1024x1024) ok in {(t1 - t0) * 1000:.1f} ms")
        del a, b, c
        torch.cuda.synchronize() if dev.type == "cuda" else None
    except Exception as e:
        print("ERROR: Matmul failed:", e)
        print(traceback.format_exc())
        return False

    try:
        # Small Conv2D
        import torch.nn as nn

        x = torch.randn(1, 1, 64, 64, device=dev)
        conv = nn.Conv2d(1, 4, kernel_size=3).to(dev)
        t0 = time.time()
        y = conv(x)
        _ = y.mean().item()
        t1 = time.time()
        print(f"Conv2D (1x1x64x64 -> 4 channels) ok in {(t1 - t0) * 1000:.1f} ms")
        del x, conv, y
        torch.cuda.synchronize() if dev.type == "cuda" else None
    except Exception as e:
        print("ERROR: Conv2D failed:", e)
        print(traceback.format_exc())
        return False

    print("Basic torch ops succeeded on device:", device_str)
    return True


def load_pyannote_pipeline(model_name: str, hf_token: str = None):
    print_header(f"Loading PyAnnote pipeline: '{model_name}'")
    try:
        kwargs = {}
        if hf_token:
            kwargs["use_auth_token"] = hf_token
        else:
            env_token = os.environ.get("HUGGINGFACE_TOKEN")
            if env_token:
                kwargs["use_auth_token"] = env_token

        t0 = time.time()
        pipeline = Pipeline.from_pretrained(model_name, **kwargs)
        t1 = time.time()
        print(f"PyAnnote pipeline loaded in {(t1 - t0):.2f}s")
        return pipeline
    except Exception as e:
        print("ERROR: Failed to load PyAnnote pipeline:", e)
        print(traceback.format_exc())
        return None


def move_pipeline_to_device(pipeline, device_str: str) -> bool:
    print_header(f"Moving pipeline to device '{device_str}'")
    try:
        dev = torch.device(device_str)
    except Exception as e:
        print(f"ERROR: Invalid device for pipeline.to(): '{device_str}':", e)
        return False
    try:
        t0 = time.time()
        pipeline.to(dev)
        t1 = time.time()
        print(f"pipeline.to({dev}) succeeded in {(t1 - t0):.2f}s")
        return True
    except Exception as e:
        print("ERROR: pipeline.to(device) failed:", e)
        print(traceback.format_exc())
        if "cudnn" in str(e).lower():
            print("\nDetected a cuDNN-related error while moving the pipeline.")
            print(
                "This typically indicates a mismatch between Torch/CUDA/cuDNN/driver."
            )
        return False


def generate_sine_wave(
    seconds: float = 1.0, sr: int = 16000, freq: float = 440.0, channels: int = 1
) -> np.ndarray:
    """Generate a simple sine wave [channels, samples] in float32."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    wave = 0.1 * np.sin(2.0 * math.pi * freq * t).astype(np.float32)
    if channels == 1:
        return wave.reshape(1, -1)
    else:
        return np.vstack([wave for _ in range(channels)])


def test_pyannote_inference(pipeline, sr: int = 16000, seconds: float = 1.0) -> bool:
    """
    Run a minimal inference call on generated audio.
    Note: For pyannote/speaker-diarization-3.x, providing a dict with
    {"waveform": np.ndarray [channels, samples], "sample_rate": sr}
    is acceptable.
    """
    print_header("Running minimal PyAnnote inference (generated sine wave)")
    try:
        waveform = generate_sine_wave(seconds=seconds, sr=sr, freq=440.0, channels=1)
        waveform = torch.from_numpy(waveform).float()
        t0 = time.time()
        result = pipeline({"waveform": waveform, "sample_rate": sr})
        t1 = time.time()
        print(f"Inference ok in {(t1 - t0):.2f}s")
        # Print basic result summary if available
        try:
            # result could be an Annotation; print number of segments/labels if possible
            labels = getattr(result, "labels", lambda: [])()
            print(f"Result type: {type(result).__name__}, labels: {labels}")
        except Exception:
            pass
        return True
    except Exception as e:
        print("ERROR: Minimal inference failed:", e)
        print(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose PyAnnote GPU pipeline loading & device move."
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="PyAnnote model name or path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device: cpu | mps | cuda | cuda:0 | cuda:N",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (uses HUGGINGFACE_TOKEN env if not provided)",
    )
    parser.add_argument(
        "--sr", type=int, default=16000, help="Sample rate for test audio"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration (seconds) for test audio inference",
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip the minimal inference test"
    )
    parser.add_argument(
        "--list-devices-only",
        action="store_true",
        help="Only print environment and device info, then exit",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    print_env_info(verbose=args.verbose)

    if args.list_devices_only:
        print("\nExiting because --list-devices-only was set.")
        sys.exit(0)

    # Test basic torch ops on the target device first
    if args.device != "cpu":
        ok = test_torch_device_ops(args.device)
        if not ok:
            print("\nDevice basic ops failed. You can:")
            print("- Re-run with --device cpu to bypass GPU.")
            print("- Verify PyTorch/CUDA/cuDNN versions align.")
            print("- Check nvidia-smi output and LD_LIBRARY_PATH.")
            sys.exit(2)

    # Load PyAnnote
    pipeline = load_pyannote_pipeline(args.model, args.hf_token)
    if pipeline is None:
        print("\nFailed to load PyAnnote pipeline.")
        sys.exit(3)

    # Move to target device using torch.device
    moved = move_pipeline_to_device(pipeline, args.device)
    if not moved:
        print("\nPipeline move to device failed.")
        # Try a CPU fallback to isolate: does it work on CPU?
        if args.device != "cpu":
            print(
                "\nAttempting CPU fallback to check whether failure is device-specific..."
            )
            moved_cpu = move_pipeline_to_device(pipeline, "cpu")
            if not moved_cpu:
                print(
                    "ERROR: Moving pipeline to CPU failed as well. The issue is likely unrelated to CUDA."
                )
                sys.exit(4)
            else:
                print(
                    "CPU fallback succeeded. The issue is likely CUDA/cuDNN specific."
                )
                sys.exit(5)
        else:
            sys.exit(4)

    # Optional inference
    if not args.skip_inference:
        ok_inf = test_pyannote_inference(pipeline, sr=args.sr, seconds=args.duration)
        if not ok_inf:
            print(
                "\nInference failed. This confirms the problem occurs during or after pipeline move."
            )
            sys.exit(6)
        else:
            print(
                "\nSuccess: Pipeline loaded, moved to device, and inference ran correctly."
            )

    print("\nAll tests completed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
