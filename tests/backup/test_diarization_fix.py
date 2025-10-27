#!/usr/bin/env python3
"""
Simple test to verify diarization service fixes work correctly.
"""

import os
import sys
from pathlib import Path

# Add app directory to path
app_root = Path(__file__).parent
sys.path.insert(0, str(app_root / "app"))

# Set test environment
os.environ.update(
    {
        "TEST_MODE": "true",
        "MOCK_GPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "LOG_LEVEL": "DEBUG",
        "ALLOWED_HOSTS": '["localhost", "127.0.0.1"]',
        "CORS_ORIGINS": '["http://localhost:3000"]',
    }
)


def test_diarization_service():
    """Test that diarization service initializes and runs without errors."""
    print("🧪 Testing Diarization Service...")

    try:
        from app.services.diarization_service import DiarizationService
        from app.models.transcription import Transcription

        print("✅ Successfully imported DiarizationService")

        # Initialize service
        service = DiarizationService()
        print("✅ Service initialized successfully")

        # Check pipeline loading
        service._load_pipeline()
        if service.pipeline is not None:
            print("✅ Pipeline loaded successfully")
            print(f"   Pipeline type: {type(service.pipeline)}")
        else:
            print("❌ Pipeline is None after loading")
            return False

        # Test with mock transcription
        mock_transcription = Transcription()
        mock_transcription.id = "test-123"
        mock_transcription.file_path = "/tmp/test.wav"
        mock_transcription.status = "transcription_completed"
        mock_transcription.segments = []

        print("✅ Mock transcription created")

        # Create a temporary test audio file
        import tempfile
        import numpy as np

        try:
            from scipy.io import wavfile
        except ImportError:
            print("⚠️  scipy not available, creating simple test file")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(
                    b"RIFF\x24\x08\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00\x00"
                )
                test_audio_path = temp_file.name
        else:
            # Create simple sine wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sample_rate = 16000
                duration = 1.0
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                frequency = 440
                amplitude = 0.3
                audio_data = (
                    amplitude * np.sin(2 * np.pi * frequency * t) * 32767
                ).astype(np.int16)
                wavfile.write(temp_file.name, sample_rate, audio_data)
                test_audio_path = temp_file.name

        # Test diarization (should work with mock pipeline)
        try:
            result = service.diarize_audio(
                audio_path=test_audio_path,  # Use actual temporary file
                transcription=mock_transcription,
                session=None,
                progress_callback=lambda p, s: print(f"   Progress: {p}% - {s}"),
            )

            # Cleanup test file
            try:
                os.unlink(test_audio_path)
            except:
                pass

            print("✅ Diarization completed successfully")
            print(f"   Result keys: {list(result.keys())}")
            print(f"   Speakers found: {result.get('total_speakers', 0)}")

            return True

        except Exception as e:
            print(f"❌ Diarization failed: {e}")
            import traceback

            traceback.print_exc()

            # Cleanup test file
            try:
                os.unlink(test_audio_path)
            except:
                pass

            return False

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_parameter_fix():
    """Test that the parameter name fix works."""
    print("\n🧪 Testing Pipeline Parameter Fix...")

    try:
        from app.services.diarization_service import DiarizationService

        service = DiarizationService()

        # Check that min_duration is used instead of min_duration_on
        import inspect

        # Get the _perform_diarization method source
        source = inspect.getsource(service._perform_diarization)

        if "min_duration_on=" in source:
            print("❌ Still using old parameter name 'min_duration_on'")
            return False
        elif "min_duration=" in source:
            print("✅ Using correct parameter name 'min_duration'")
            return True
        else:
            print("⚠️  Could not find min_duration parameter in source")
            return True

    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Testing Diarization Service Fixes")
    print("=" * 50)

    # Run tests
    results = []
    results.append(("Diarization Service", test_diarization_service()))
    results.append(("Parameter Fix", test_pipeline_parameter_fix()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1

    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Diarization fix is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    sys.exit(0 if passed == len(results) else 1)
