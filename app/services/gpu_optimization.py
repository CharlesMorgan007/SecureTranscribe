"""
GPU optimization utilities for SecureTranscribe.
Manages GPU memory allocation, device selection, and performance optimization.
"""

import logging
import gc
import os
from typing import Optional, Tuple, Dict, Any
import torch

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    GPU optimization utilities to maximize performance and prevent memory issues.
    Handles device selection, memory management, and performance monitoring.
    """

    def __init__(self):
        self.device_info = self._get_device_info()
        self.memory_allocated = 0
        self.memory_reserved = 0

    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU device information."""
        device_info = {
            "cuda_available": False,
            "mps_available": False,
            "device_count": 0,
            "devices": [],
            "recommended_device": "cpu",
        }

        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                device_info["cuda_available"] = True
                device_info["device_count"] = torch.cuda.device_count()

                for i in range(device_info["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    device_info["devices"].append(
                        {
                            "id": i,
                            "name": props.name,
                            "total_memory": props.total_memory,
                            "major": props.major,
                            "minor": props.minor,
                            "multi_processor_count": props.multi_processor_count,
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )

                # Select best device
                device_info["recommended_device"] = self._select_best_cuda_device(
                    device_info["devices"]
                )

            # Check Apple MPS availability
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_info["mps_available"] = True
                device_info["recommended_device"] = "mps"

        except Exception as e:
            logger.warning(f"Error getting GPU device info: {e}")

        return device_info

    def _select_best_cuda_device(self, devices: list) -> str:
        """Select the best CUDA device based on memory and compute capability."""
        if not devices:
            return "cpu"

        # Sort by compute capability, then by memory
        def device_score(device):
            cc = float(device["compute_capability"])
            memory_gb = device["total_memory"] / (1024**3)
            # Prioritize newer compute capability, then more memory
            return (cc, memory_gb)

        best_device = max(devices, key=device_score)
        # For Whisper/faster-whisper compatibility, return just "cuda" for device 0
        # otherwise include device number for multi-GPU systems
        if best_device["id"] == 0:
            return "cuda"
        else:
            return f"cuda:{best_device['id']}"

    def get_optimal_device(self) -> str:
        """Get the optimal device for processing."""
        return self.device_info["recommended_device"]

    def optimize_model_loading(self, device: str, model_name: str) -> Dict[str, Any]:
        """
        Get optimal model loading parameters for a specific device.

        Args:
            device: Target device string
            model_name: Name of the model being loaded

        Returns:
            Dictionary of optimal loading parameters
        """
        params = {
            "device": device,
            "compute_type": "float32",  # Default
            "torch_dtype": torch.float32,
        }

        # Special handling for Whisper/faster-whisper device compatibility
        if device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            if device_id == 0:
                params["device"] = "cuda"  # faster-whisper prefers "cuda" not "cuda:0"
            # For multi-GPU systems, some models may not support specific device IDs
            # so we provide fallback options
            elif "whisper" in model_name.lower():
                params["device"] = "cuda"  # Fallback to default CUDA device
                logger.warning(
                    f"Using default CUDA device for {model_name} instead of {device}"
                )

        logger.info(f"Device optimization for {model_name}: {params}")

        if device.startswith("cuda"):
            # CUDA-specific optimizations
            try:
                # Use mixed precision for compatible GPUs
                device_id = int(device.split(":")[-1]) if ":" in device else 0
                props = torch.cuda.get_device_properties(device_id)

                # Enable float16 for newer GPUs with sufficient memory
                if props.major >= 7 and props.total_memory >= 8 * 1024**3:  # 8GB+
                    params["compute_type"] = "float16"
                    params["torch_dtype"] = torch.float16
                    logger.info(f"Using float16 precision for {model_name} on {device}")
                else:
                    logger.info(f"Using float32 precision for {model_name} on {device}")

                # Set memory allocation strategy
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

            except Exception as e:
                logger.warning(f"Could not optimize CUDA settings: {e}")

        elif device == "mps":
            # MPS-specific optimizations
            params["compute_type"] = (
                "float32"  # MPS doesn't support float16 for all models
            )
            params["torch_dtype"] = torch.float32
            logger.info(f"Using MPS-optimized settings for {model_name}")

        else:
            logger.warning(f"Unknown device '{device}', using default CPU settings")
            params["device"] = "cpu"
            params["compute_type"] = "float32"
            params["torch_dtype"] = torch.float32

        return params

    def clear_gpu_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("GPU cache cleared and garbage collected")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {"device": self.device_info["recommended_device"]}

        if torch.cuda.is_available():
            try:
                device_id = 0
                if ":" in self.device_info["recommended_device"]:
                    device_id = int(
                        self.device_info["recommended_device"].split(":")[-1]
                    )

                memory_info.update(
                    {
                        "allocated": torch.cuda.memory_allocated(device_id)
                        / (1024**3),  # GB
                        "reserved": torch.cuda.memory_reserved(device_id)
                        / (1024**3),  # GB
                        "max_allocated": torch.cuda.max_memory_allocated(device_id)
                        / (1024**3),
                        "total": torch.cuda.get_device_properties(
                            device_id
                        ).total_memory
                        / (1024**3),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get memory info: {e}")

        return memory_info

    def optimize_for_large_file(self, file_duration_minutes: float) -> Dict[str, Any]:
        """
        Get optimization recommendations for processing large audio files.

        Args:
            file_duration_minutes: Duration of audio file in minutes

        Returns:
            Dictionary of optimization recommendations
        """
        optimizations = {
            "use_chunked_processing": True,
            "chunk_size": 30,  # Default 30 second chunks
            "batch_size": 1,  # Process one chunk at a time
            "clear_cache_frequency": 10,  # Clear cache every N chunks
        }

        # Adjust based on file size
        if file_duration_minutes > 60:  # Over 1 hour
            optimizations["chunk_size"] = 60  # Use larger chunks
            optimizations["clear_cache_frequency"] = 5
            logger.info(
                f"Large file detected ({file_duration_minutes:.1f}min), using optimized chunking"
            )
        elif file_duration_minutes > 20:  # 20-60 minutes
            optimizations["chunk_size"] = 45
            optimizations["clear_cache_frequency"] = 8

        # Adjust for GPU memory constraints
        if torch.cuda.is_available():
            try:
                device_id = 0
                total_memory_gb = torch.cuda.get_device_properties(
                    device_id
                ).total_memory / (1024**3)

                if total_memory_gb < 8:  # Less than 8GB
                    optimizations["chunk_size"] = min(optimizations["chunk_size"], 20)
                    optimizations["clear_cache_frequency"] = 3
                    logger.info(
                        f"Low GPU memory ({total_memory_gb:.1f}GB), using conservative chunking"
                    )
                elif total_memory_gb >= 24:  # 24GB+ (like RTX 4090)
                    optimizations["chunk_size"] = min(optimizations["chunk_size"], 90)
                    logger.info(
                        f"High GPU memory ({total_memory_gb:.1f}GB), can use larger chunks"
                    )

            except Exception as e:
                logger.warning(f"Could not optimize based on GPU memory: {e}")

        return optimizations

    def log_device_info(self):
        """Log comprehensive device information."""
        logger.info("=== GPU Device Information ===")

        if self.device_info["cuda_available"]:
            logger.info(f"CUDA Available: Yes")
            logger.info(f"Device Count: {self.device_info['device_count']}")
            logger.info(f"Recommended Device: {self.device_info['recommended_device']}")

            for device in self.device_info["devices"]:
                memory_gb = device["total_memory"] / (1024**3)
                logger.info(
                    f"  GPU {device['id']}: {device['name']} ({memory_gb:.1f}GB, {device['compute_capability']})"
                )

        elif self.device_info["mps_available"]:
            logger.info("Apple MPS Available: Yes")
            logger.info("Recommended Device: mps")
        else:
            logger.info("GPU Available: No (CPU only)")

        # Log current memory usage
        memory_info = self.get_memory_info()
        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory: {memory_info.get('allocated', 0):.2f}GB / {memory_info.get('total', 0):.1f}GB"
            )

        logger.info("============================")

    def test_device_compatibility(self, device: str, model_name: str) -> bool:
        """
        Test if a device is compatible with specific model.

        Args:
            device: Device string to test
            model_name: Name of model (whisper, pyannote, etc.)

        Returns:
            True if device is compatible, False otherwise
        """
        try:
            if not device or device == "cpu":
                return True

            if device.startswith("cuda"):
                import torch

                if not torch.cuda.is_available():
                    return False

                # Test if we can create a tensor on device
                try:
                    test_tensor = torch.randn(1, 10).to(device)
                    del test_tensor
                    return True
                except Exception:
                    return False

            elif device == "mps":
                import torch

                if not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    return False
                try:
                    test_tensor = torch.randn(1, 10).to("mps")
                    del test_tensor
                    return True
                except Exception:
                    return False

            return False

        except Exception as e:
            logger.warning(f"Device compatibility test failed for {device}: {e}")
            return False


# Global instance for easy access
_gpu_optimizer = None


def get_gpu_optimizer() -> GPUOptimizer:
    """Get the global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer


def initialize_gpu_optimization():
    """Initialize GPU optimization and log device information."""
    optimizer = get_gpu_optimizer()
    optimizer.log_device_info()
    return optimizer
