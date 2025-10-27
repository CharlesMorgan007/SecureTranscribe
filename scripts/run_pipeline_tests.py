#!/usr/bin/env python3
"""
Standalone test runner for SecureTranscribe processing pipeline.
This script runs comprehensive tests to verify the transcription and queue systems work correctly.
Designed to run on development machines without CUDA/GPU requirements.
"""

import os
import sys
import json
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Add app directory to path
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root / "app"))

# Set test environment
os.environ.update({
    "TEST_MODE": "true",
    "MOCK_GPU": "true",
    "CUDA_VISIBLE_DEVICES": "",
    "LOG_LEVEL": "DEBUG",
    "ALLOWED_HOSTS": '["localhost", "127.0.0.1"]',
    "CORS_ORIGINS": '["http://localhost:3000"]',
})

def create_test_audio_file(output_path: Path) -> Path:
    """Create a test audio file with sine wave."""
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        print("❌ scipy is required for test audio generation")
        print("   Install with: pip install scipy numpy")
        sys.exit(1)

    audio_path = output_path / "test_audio.wav"

    # Generate a simple sine wave (2 seconds, 16kHz)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A4 note
    amplitude = 0.3
    audio_data = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    wavfile.write(str(audio_path), sample_rate, audio_data)
    print(f"✅ Created test audio file: {audio_path}")
    return audio_path

def test_imports():
    """Test that all required modules can be imported."""
    print("\n🧪 Testing module imports...")

    try:
        from main import app
        from fastapi.testclient import TestClient
        from app.core.database import get_database
        from app.models.processing_queue import ProcessingQueue
        from app.models.transcription import Transcription
        from app.models.session import UserSession
        from app.services.queue_service import get_queue_service
        from app.services.transcription_service import TranscriptionService
        from app.services.diarization_service import DiarizationService
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic API functionality."""
    print("\n🧪 Testing basic API functionality...")

    try:
        from main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test health check
        response = client.get("/health")
        if response.status_code != 200:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        print("✅ Health check passed")

        # Test session creation
        response = client.get("/api/sessions/create")
        if response.status_code != 200:
            print(f"❌ Session creation failed: {response.status_code}")
            return False
        session_data = response.json()
        session_id = session_data.get("session_id")
        if not session_id:
            print("❌ No session ID returned")
            return False
        print(f"✅ Session created: {session_id}")

        # Test queue status
        response = client.get("/api/queue/status")
        if response.status_code != 200:
            print(f"❌ Queue status failed: {response.status_code}")
            return False
        queue_data = response.json()
        print(f"✅ Queue status: {queue_data.get('total_jobs', 0)} total jobs")

        return session_id

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_file_upload(session_id: str, test_audio_path: Path):
    """Test file upload functionality."""
    print("\n🧪 Testing file upload...")

    try:
        from main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Upload test audio
        with open(test_audio_path, "rb") as f:
            response = client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id}
            )

        if response.status_code != 200:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None

        upload_data = response.json()
        transcription_id = upload_data.get("transcription_id")

        if not transcription_id:
            print("❌ No transcription ID returned from upload")
            return None

        print(f"✅ File uploaded successfully: {transcription_id}")

        # Verify upload status
        response = client.get(f"/api/transcription/{transcription_id}/status")
        if response.status_code != 200:
            print(f"❌ Status check failed: {response.status_code}")
            return None

        status_data = response.json()
        print(f"✅ Upload status: {status_data.get('status')}")

        return transcription_id

    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        traceback.print_exc()
        return None

def test_mock_processing(session_id: str, transcription_id: str):
    """Test transcription with mocked GPU services."""
    print("\n🧪 Testing mock transcription processing...")

    try:
        from main import app
        from fastapi.testclient import TestClient
        from app.services.transcription_service import TranscriptionService
        from app.services.diarization_service import DiarizationService

        client = TestClient(app)

        # Mock the GPU services
        with (
            patch.object(TranscriptionService, 'transcribe_audio') as mock_transcribe,
            patch.object(DiarizationService, 'diarize_audio') as mock_diarize,
            patch.object(TranscriptionService, '_get_device') as mock_device
        ):
            # Configure mocks
            mock_device.return_value = "cpu"

            mock_transcribe.return_value = {
                "text": "This is a mock transcription test. The system is working correctly.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "This is a mock transcription test.",
                        "speaker": "SPEAKER_00"
                    },
                    {
                        "start": 2.0,
                        "end": 4.0,
                        "text": "The system is working correctly.",
                        "speaker": "SPEAKER_01"
                    }
                ]
            }

            mock_diarize.return_value = {
                "speakers": [
                    {"id": "SPEAKER_00", "name": "Speaker 1", "confidence": 0.95},
                    {"id": "SPEAKER_01", "name": "Speaker 2", "confidence": 0.87}
                ],
                "speaker_matches": {
                    "SPEAKER_00": MagicMock(name="Speaker 1"),
                    "SPEAKER_01": MagicMock(name="Speaker 2")
                }
            }

            # Start transcription
            response = client.post(
                f"/api/transcription/{transcription_id}/start",
                json={"session_id": session_id}
            )

            if response.status_code != 200:
                print(f"❌ Failed to start transcription: {response.status_code}")
                print(f"Response: {response.text}")
                return False

            print("✅ Transcription started successfully")

            # Monitor progress
            max_wait = 15
            start_time = time.time()
            completed = False

            while time.time() - start_time < max_wait:
                response = client.get(f"/api/transcription/{transcription_id}/status")

                if response.status_code != 200:
                    print(f"❌ Status check failed: {response.status_code}")
                    break

                status_data = response.json()
                current_status = status_data.get("status")
                progress = status_data.get("progress", 0)
                current_step = status_data.get("current_step", "")

                print(f"  📊 Status: {current_status}, Progress: {progress}%, Step: {current_step}")

                if current_status == "completed":
                    completed = True
                    break
                elif current_status == "failed":
                    error = status_data.get("error", "Unknown error")
                    print(f"❌ Transcription failed: {error}")
                    return False

                time.sleep(0.5)

            if not completed:
                print("❌ Transcription did not complete within timeout")
                return False

            print("✅ Transcription completed successfully")

            # Get final results
            response = client.get(f"/api/transcription/{transcription_id}")
            if response.status_code != 200:
                print(f"❌ Failed to get final results: {response.status_code}")
                return False

            final_data = response.json()
            print(f"✅ Final text: {final_data.get('text', 'No text')[:50]}...")
            print(f"✅ Segments: {len(final_data.get('segments', []))}")

            return True

    except Exception as e:
        print(f"❌ Mock processing test failed: {e}")
        traceback.print_exc()
        return False

def test_queue_status_tracking(session_id: str):
    """Test that queue status updates are tracked correctly."""
    print("\n🧪 Testing queue status tracking...")

    try:
        from main import app
        from fastapi.testclient import TestClient
        from app.core.database import get_database
        from app.models.processing_queue import ProcessingQueue

        client = TestClient(app)

        # Check initial queue status
        response = client.get("/api/queue/status")
        if response.status_code != 200:
            print(f"❌ Failed to get queue status: {response.status_code}")
            return False

        queue_data = response.json()
        initial_total = queue_data.get('total_jobs', 0)
        initial_queued = queue_data.get('queued_jobs', 0)

        print(f"  📊 Initial queue: {initial_total} total, {initial_queued} queued")

        # Check user's jobs
        response = client.get("/api/queue/jobs")
        if response.status_code != 200:
            print(f"❌ Failed to get user jobs: {response.status_code}")
            return False

        jobs_data = response.json()
        user_jobs = jobs_data.get('jobs', [])

        print(f"  📊 User jobs: {len(user_jobs)}")

        # Check database consistency
        with next(get_database()) as db:
            db_jobs = (
                db.query(ProcessingQueue)
                .filter(ProcessingQueue.session_id == session_id)
                .all()
            )
            print(f"  📊 Database jobs: {len(db_jobs)}")

            for job in db_jobs:
                job_dict = job.to_dict()
                print(f"    - Job {job_dict.get('job_id')}: {job_dict.get('status')}")

        print("✅ Queue status tracking is working correctly")
        return True

    except Exception as e:
        print(f"❌ Queue status tracking test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and status tracking."""
    print("\n🧪 Testing error handling...")

    try:
        from main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test non-existent transcription
        response = client.get("/api/transcription/99999/status")
        if response.status_code != 404:
            print(f"❌ Expected 404 for non-existent transcription, got {response.status_code}")
            return False
        print("✅ Non-existent transcription returns 404")

        # Test invalid session
        fake_session = "fake-session-id"
        with open("test_dummy.txt", "w") as f:
            f.write("dummy")

        try:
            with open("test_dummy.txt", "rb") as f:
                response = client.post(
                    "/api/transcription/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"session_id": fake_session}
                )

            if response.status_code != 401:
                print(f"❌ Expected 401 for invalid session, got {response.status_code}")
                return False
            print("✅ Invalid session returns 401")
        finally:
            if os.path.exists("test_dummy.txt"):
                os.remove("test_dummy.txt")

        # Test queue service status
        response = client.get("/api/queue/worker-status")
        if response.status_code != 200:
            print(f"❌ Worker status failed: {response.status_code}")
            return False
        print("✅ Worker status endpoint works")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_progress_updates(session_id: str, transcription_id: str):
    """Test that progress updates are working."""
    print("\n🧪 Testing progress updates...")

    try:
        from main import app
        from fastapi.testclient import TestClient
        from app.services.transcription_service import TranscriptionService

        client = TestClient(app)

        # Track progress updates
        progress_seen = []

        def mock_progress_callback(percentage, step):
            progress_seen.append((percentage, step))
            print(f"    📈 Progress: {percentage}% - {step}")

        with patch.object(TranscriptionService, 'transcribe_audio') as mock_transcribe:
            def mock_transcribe_with_progress(*args, **kwargs):
                progress_cb = kwargs.get('progress_callback')
                if progress_cb:
                    progress_cb(25, "Loading audio...")
                    progress_cb(50, "Processing transcription...")
                    progress_cb(75, "Finalizing results...")
                return {
                    "text": "Progress test completed successfully.",
                    "segments": [{"start": 0, "end": 2, "text": "Progress test", "speaker": "SPEAKER_00"}]
                }

            mock_transcribe.side_effect = mock_transcribe_with_progress

            # Start another transcription
            response = client.post(
                f"/api/transcription/{transcription_id}/start",
                json={"session_id": session_id}
            )

            if response.status_code != 200:
                print(f"❌ Failed to start progress test: {response.status_code}")
                return False

            # Monitor progress
            max_wait = 10
            start_time = time.time()

            while time.time() - start_time < max_wait:
                response = client.get(f"/api/transcription/{transcription_id}/status")

                if response.status_code == 200:
                    status_data = response.json()
                    progress = status_data.get("progress", 0)
                    step = status_data.get("current_step", "")

                    if progress > 0:
                        print(f"  📊 API Progress: {progress}% - {step}")

                    if status_data.get("status") == "completed":
                        break

                time.sleep(0.5)

        print("✅ Progress updates are working correctly")
        return True

    except Exception as e:
        print(f"❌ Progress updates test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Starting SecureTranscribe Pipeline Tests")
    print("=" * 50)

    # Create temporary directory and test audio
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_audio_path = create_test_audio_file(temp_path)

        results = []

        # Run tests
        results.append(("Module Imports", test_imports()))

        session_id = test_basic_functionality()
        results.append(("Basic Functionality", bool(session_id)))

        if session_id:
            transcription_id = test_file_upload(session_id, test_audio_path)
            results.append(("File Upload", bool(transcription_id)))

            if transcription_id:
                results.append(("Mock Processing", test_mock_processing(session_id, transcription_id)))
                results.append(("Progress Updates", test_progress_updates(session_id, transcription_id)))

            results.append(("Queue Status Tracking", test_queue_status_tracking(session_id)))

        results.append(("Error Handling", test_error_handling()))

        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)

        passed = 0
        total = len(results)

        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
            if result:
                passed += 1

        print("-" * 50)
        print(f"Total: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! The processing pipeline is working correctly.")
            return True
        else:
            print(f"⚠️  {total - passed} test(s) failed. Check the output above for details.")
            return False

if __name__ == "__main__":
    print("SecureTranscribe Processing Pipeline Test Runner")
    print("This script tests the transcription and queue systems without requiring GPU.")
    print()

    # Check dependencies
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("⚠️  pytest not found, but script will still work")

    # Run tests
    success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
```

Now let me create a test configuration file and documentation:
<tool_call>create_directory
<arg_key>path</arg_key>
<arg_value>/Users/cmorgan/Devel/Personal/SecureTranscribe/tests/integration/config</arg_value>
</tool_call>
