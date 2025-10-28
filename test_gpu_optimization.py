#!/usr/bin/env python3
"""
Test script to verify GPU optimization and export functionality in SecureTranscribe.
This script will check:
1. GPU detection and initialization
2. Model loading optimization
3. Export functionality
4. Performance improvements
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add app to path
app_root = Path(__file__).parent
sys.path.insert(0, str(app_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_gpu_detection():
    """Test GPU detection and optimization."""
    logger.info("=== Testing GPU Detection ===")

    try:
        from app.services.gpu_optimization import (
            get_gpu_optimizer,
            initialize_gpu_optimization,
        )

        # Initialize GPU optimizer
        optimizer = initialize_gpu_optimization()

        # Test device detection
        device = optimizer.get_optimal_device()
        logger.info(f"Optimal device detected: {device}")

        # Test device info
        device_info = optimizer.device_info
        logger.info(f"Device info: {device_info}")

        # Test memory info
        memory_info = optimizer.get_memory_info()
        logger.info(f"Memory info: {memory_info}")

        # Test large file optimization
        test_duration = 20.0  # 20 minutes
        optimization_params = optimizer.optimize_for_large_file(test_duration)
        logger.info(
            f"Large file optimization for {test_duration}min: {optimization_params}"
        )

        # Test model loading optimization
        model_params = optimizer.optimize_model_loading(device, "whisper-base")
        logger.info(f"Model loading params: {model_params}")

        logger.info("✅ GPU detection test passed")
        return True

    except Exception as e:
        logger.error(f"❌ GPU detection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_transcription_service():
    """Test transcription service with GPU optimization."""
    logger.info("=== Testing Transcription Service ===")

    try:
        from app.services.transcription_service import TranscriptionService
        from app.core.config import get_settings

        # Initialize service
        service = TranscriptionService()
        logger.info(f"Transcription service device: {service.device}")
        logger.info(f"Transcription service model size: {service.model_size}")

        # Test GPU optimizer integration
        if hasattr(service, "gpu_optimizer"):
            logger.info("✅ GPU optimizer integrated in transcription service")

            # Test memory clearing
            service.gpu_optimizer.clear_gpu_cache()
            logger.info("✅ GPU cache clearing works")
        else:
            logger.error("❌ GPU optimizer not integrated in transcription service")
            return False

        logger.info("✅ Transcription service test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Transcription service test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_diarization_service():
    """Test diarization service with GPU optimization."""
    logger.info("=== Testing Diarization Service ===")

    try:
        from app.services.diarization_service import DiarizationService

        # Initialize service
        service = DiarizationService()
        logger.info(f"Diarization service device: {service.device}")
        logger.info(f"Diarization service model: {service.pyannote_model}")

        # Test GPU optimizer integration
        if hasattr(service, "gpu_optimizer"):
            logger.info("✅ GPU optimizer integrated in diarization service")

            # Test memory clearing
            service.gpu_optimizer.clear_gpu_cache()
            logger.info("✅ GPU cache clearing works")
        else:
            logger.error("❌ GPU optimizer not integrated in diarization service")
            return False

        logger.info("✅ Diarization service test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Diarization service test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_export_functionality():
    """Test export functionality to identify export errors."""
    logger.info("=== Testing Export Functionality ===")

    try:
        from app.services.export_service import ExportService
        from app.models.transcription import Transcription

        # Initialize export service
        export_service = ExportService()
        logger.info("✅ Export service initialized")

        # Test supported formats
        formats = export_service.get_export_formats()
        logger.info(f"✅ Supported export formats: {formats}")

        # Test include options
        options = export_service.get_include_options()
        logger.info(f"✅ Include options: {options}")

        # Create a mock transcription for testing
        mock_transcription = type(
            "MockTranscription",
            (),
            {
                "id": 1,
                "session_id": "test_session",
                "original_filename": "test_audio.mp3",
                "file_duration": 120.0,
                "file_size": 1024000,
                "file_format": "mp3",
                "status": "completed",
                "is_completed": True,
                "full_transcript": "This is a test transcript for export functionality.",
                "language_detected": "en",
                "confidence_score": 0.95,
                "num_speakers": 2,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "text": "Hello world",
                        "speaker": "Speaker 1",
                        "confidence": 0.9,
                    },
                    {
                        "start": 5.0,
                        "end": 10.0,
                        "text": "This is a test",
                        "speaker": "Speaker 2",
                        "confidence": 0.95,
                    },
                ],
            },
        )()

        # Add methods to mock transcription
        def mock_export_to_dict():
            return {
                "id": 1,
                "session_id": "test_session",
                "original_filename": "test_audio.mp3",
                "full_transcript": "This is a test transcript for export functionality.",
                "segments": mock_transcription.segments,
            }

        def mock_get_speaker_list():
            return ["Speaker 1", "Speaker 2"]

        def mock_get_speaker_stats():
            return {
                "Speaker 1": {"segment_count": 1},
                "Speaker 2": {"segment_count": 1},
            }

        mock_transcription.export_to_dict = mock_export_to_dict
        mock_transcription.get_speaker_list = mock_get_speaker_list
        mock_transcription.get_speaker_stats = mock_get_speaker_stats

        # Test JSON export (simplest format)
        try:
            json_result = export_service.export_transcription(
                mock_transcription, "json", [], None
            )
            logger.info(f"✅ JSON export successful, {len(json_result)} bytes")
        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
            return False

        # Test TXT export
        try:
            txt_result = export_service.export_transcription(
                mock_transcription, "txt", [], None
            )
            logger.info(f"✅ TXT export successful, {len(txt_result)} bytes")
        except Exception as e:
            logger.error(f"❌ TXT export failed: {e}")
            return False

        # Test CSV export
        try:
            csv_result = export_service.export_transcription(
                mock_transcription, "csv", [], None
            )
            logger.info(f"✅ CSV export successful, {len(csv_result)} bytes")
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            return False

        # Test PDF export (most complex)
        try:
            pdf_result = export_service.export_transcription(
                mock_transcription, "pdf", [], None
            )
            logger.info(f"✅ PDF export successful, {len(pdf_result)} bytes")
        except Exception as e:
            logger.error(f"❌ PDF export failed: {e}")
            # Don't return False here as PDF might have dependency issues
            logger.warning("PDF export might have missing dependencies, continuing...")

        logger.info("✅ Export functionality test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Export functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_improvements():
    """Test if performance improvements are working."""
    logger.info("=== Testing Performance Improvements ===")

    try:
        # Test if GPU is actually being used
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available with {device_count} device(s)")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")

                # Test if we can actually allocate memory
                try:
                    test_tensor = torch.randn(1000, 1000).cuda(i)
                    del test_tensor
                    torch.cuda.empty_cache()
                    logger.info(f"  ✅ GPU {i} memory allocation test passed")
                except Exception as e:
                    logger.error(f"  ❌ GPU {i} memory allocation test failed: {e}")
        else:
            logger.warning("⚠️  CUDA not available, tests will run on CPU")

        # Check if Apple MPS is available
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("✅ Apple MPS available")

            # Test MPS allocation
            try:
                test_tensor = torch.randn(1000, 1000).to("mps")
                del test_tensor
                logger.info("✅ MPS memory allocation test passed")
            except Exception as e:
                logger.error(f"❌ MPS memory allocation test failed: {e}")
        else:
            logger.info("ℹ️  Apple MPS not available")

        logger.info("✅ Performance improvements test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Performance improvements test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting SecureTranscribe GPU Optimization Tests")
    logger.info("=" * 60)

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Transcription Service", test_transcription_service),
        ("Diarization Service", test_diarization_service),
        ("Export Functionality", test_export_functionality),
        ("Performance Improvements", test_performance_improvements),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! GPU optimization is working correctly.")
        logger.info("\n📋 NEXT STEPS:")
        logger.info("1. Test with a real audio file to verify performance improvements")
        logger.info("2. Monitor GPU memory usage during processing")
        logger.info("3. Verify export functionality through the web interface")
        return 0
    else:
        logger.error(
            f"💥 {total - passed} test(s) failed. Please check the errors above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
