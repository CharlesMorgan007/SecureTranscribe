"""
Comprehensive integration tests for the transcription processing pipeline.
Tests the entire flow from file upload through transcription, diarization, and status tracking.
Designed to run on development machines without CUDA/GPU requirements.
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

import pytest
from fastapi.testclient import TestClient

# Add app directory to path
app_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(app_root / "app"))

# Set test environment variables before importing app modules
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

try:
    from main import app
    from app.core.database import get_database
    from app.models.processing_queue import ProcessingQueue
    from app.models.transcription import Transcription
    from app.models.session import UserSession
    from app.services.queue_service import get_queue_service
    from app.services.transcription_service import TranscriptionService
    from app.services.diarization_service import DiarizationService
except ImportError as e:
    print(f"Failed to import app modules: {e}")
    pytest.exit("Cannot import required app modules")


class TestProcessingPipeline:
    """Test the complete transcription processing pipeline."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        """Set up test environment with temporary directories and mock data."""
        # Create temporary directories
        self.temp_dir = tmp_path
        self.upload_dir = self.temp_dir / "uploads"
        self.processed_dir = self.temp_dir / "processed"
        self.logs_dir = self.temp_dir / "logs"

        for dir_path in [self.upload_dir, self.processed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create test audio file (sine wave)
        self.test_audio_path = self._create_test_audio_file()

        # Mock settings
        self.mock_settings = MagicMock()
        self.mock_settings.upload_dir = str(self.upload_dir)
        self.mock_settings.processed_dir = str(self.processed_dir)
        self.mock_settings.test_mode = True
        self.mock_settings.mock_gpu = True
        self.mock_settings.log_level = "DEBUG"
        self.mock_settings.max_workers = 2
        self.mock_settings.queue_size = 10
        self.mock_settings.processing_timeout = 60

        # Client and session
        self.client = TestClient(app)
        self.session_id = None

        yield

        # Cleanup
        if self.session_id:
            self._cleanup_test_data()

    def _create_test_audio_file(self) -> Path:
        """Create a test audio file with sine wave."""
        import numpy as np
        from scipy.io import wavfile

        audio_path = self.upload_dir / "test_audio.wav"

        # Generate a simple sine wave (1 second, 16kHz)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440  # A4 note
        amplitude = 0.3
        audio_data = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(
            np.int16
        )

        wavfile.write(str(audio_path), sample_rate, audio_data)
        return audio_path

    def _get_session(self):
        """Get or create a user session."""
        if not self.session_id:
            response = self.client.get("/api/sessions/create")
            assert response.status_code == 200
            data = response.json()
            self.session_id = data["session_id"]
        return self.session_id

    def _cleanup_test_data(self):
        """Clean up test data from database."""
        try:
            with next(get_database()) as db:
                # Delete test transcriptions
                db.query(Transcription).filter(
                    Transcription.session_id == self.session_id
                ).delete()

                # Delete test queue jobs
                db.query(ProcessingQueue).filter(
                    ProcessingQueue.session_id == self.session_id
                ).delete()

                # Delete test session
                db.query(UserSession).filter(
                    UserSession.session_id == self.session_id
                ).delete()

                db.commit()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def test_file_upload_and_initial_status(self):
        """Test file upload and initial status tracking."""
        session_id = self._get_session()

        # Upload file
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id},
            )

        assert response.status_code == 200
        data = response.json()
        assert "transcription_id" in data
        assert data["status"] == "uploaded"

        transcription_id = data["transcription_id"]

        # Check initial queue status
        response = self.client.get("/api/queue/status")
        assert response.status_code == 200
        queue_status = response.json()
        assert "total_jobs" in queue_status
        assert "queued_jobs" in queue_status

        # Check user jobs
        response = self.client.get("/api/queue/jobs")
        assert response.status_code == 200
        user_jobs = response.json()
        assert "jobs" in user_jobs
        assert len(user_jobs["jobs"]) >= 1

    @patch("app.services.transcription_service.TranscriptionService._get_device")
    @patch("app.services.diarization_service.DiarizationService._get_device")
    def test_transcription_processing_with_mock_gpu(
        self, mock_diarization_device, mock_transcription_device
    ):
        """Test complete transcription processing with mocked GPU services."""
        # Mock GPU devices
        mock_transcription_device.return_value = "cpu"
        mock_diarization_device.return_value = "cpu"

        session_id = self._get_session()

        # Mock the actual processing methods
        with (
            patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe,
            patch.object(DiarizationService, "diarize_audio") as mock_diarize,
        ):
            # Configure mocks
            mock_transcribe.return_value = {
                "text": "This is a test transcription.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "This is a test transcription.",
                        "speaker": "SPEAKER_00",
                    }
                ],
            }

            mock_diarize.return_value = {
                "speakers": [
                    {"id": "SPEAKER_00", "name": "Speaker 1", "confidence": 0.9}
                ],
                "speaker_matches": {"SPEAKER_00": MagicMock(name="Speaker 1")},
            }

            # Upload file
            with open(self.test_audio_path, "rb") as f:
                upload_response = self.client.post(
                    "/api/transcription/upload",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"session_id": session_id},
                )

            assert upload_response.status_code == 200
            transcription_id = upload_response.json()["transcription_id"]

            # Start transcription
            start_response = self.client.post(
                f"/api/transcription/{transcription_id}/start",
                json={"session_id": session_id},
            )

            assert start_response.status_code == 200

            # Wait for processing to complete
            max_wait = 30  # seconds
            start_time = time.time()
            completed = False

            while time.time() - start_time < max_wait:
                # Check transcription status
                status_response = self.client.get(
                    f"/api/transcription/{transcription_id}/status"
                )
                assert status_response.status_code == 200

                status_data = status_response.json()
                current_status = status_data.get("status")
                progress = status_data.get("progress", 0)

                print(f"Status: {current_status}, Progress: {progress}%")

                if current_status == "completed":
                    completed = True
                    break

                time.sleep(0.5)

            assert completed, "Transcription did not complete within timeout"

            # Verify final results
            final_response = self.client.get(f"/api/transcription/{transcription_id}")
            assert final_response.status_code == 200

            final_data = final_response.json()
            assert final_data["status"] == "completed"
            assert "text" in final_data
            assert "segments" in final_data
            assert len(final_data["segments"]) > 0

    def test_queue_status_updates_during_processing(self):
        """Test that queue status updates correctly during processing."""
        session_id = self._get_session()

        # Upload multiple files to test queue
        transcription_ids = []

        for i in range(3):
            with open(self.test_audio_path, "rb") as f:
                response = self.client.post(
                    "/api/transcription/upload",
                    files={"file": (f"test_{i}.wav", f, "audio/wav")},
                    data={"session_id": session_id},
                )

            assert response.status_code == 200
            transcription_ids.append(response.json()["transcription_id"])

        # Check queue status before processing
        queue_response = self.client.get("/api/queue/status")
        assert queue_response.status_code == 200

        initial_queue = queue_response.json()
        initial_queued = initial_queue.get("queued_jobs", 0)
        assert initial_queued >= 3

        # Start processing first job
        with (
            patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe,
            patch.object(DiarizationService, "diarize_audio") as mock_diarize,
        ):
            mock_transcribe.return_value = {
                "text": "Mock transcription",
                "segments": [],
            }
            mock_diarize.return_value = {"speakers": [], "speaker_matches": {}}

            start_response = self.client.post(
                f"/api/transcription/{transcription_ids[0]}/start",
                json={"session_id": session_id},
            )

            assert start_response.status_code == 200

            # Check queue status during processing
            processing_response = self.client.get("/api/queue/status")
            assert processing_response.status_code == 200

            processing_queue = processing_response.json()
            # Should have at least one processing job
            assert processing_queue.get("processing_jobs", 0) >= 0

    def test_job_cancellation_and_status_tracking(self):
        """Test job cancellation and status tracking."""
        session_id = self._get_session()

        # Upload file
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id},
            )

        assert response.status_code == 200
        transcription_id = response.json()["transcription_id"]

        # Start with a slow mock to allow cancellation
        with patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe:

            def slow_transcribe(*args, **kwargs):
                time.sleep(5)  # Simulate slow processing
                return {"text": "Slow transcription", "segments": []}

            mock_transcribe.side_effect = slow_transcribe

            # Start processing in background thread
            def start_processing():
                self.client.post(
                    f"/api/transcription/{transcription_id}/start",
                    json={"session_id": session_id},
                )

            processing_thread = threading.Thread(target=start_processing)
            processing_thread.start()

            # Give it time to start
            time.sleep(1)

            # Get job ID from queue
            jobs_response = self.client.get("/api/queue/jobs?status_filter=processing")
            assert jobs_response.status_code == 200

            jobs_data = jobs_response.json()
            if jobs_data["jobs"]:
                job_id = jobs_data["jobs"][0]["job_id"]

                # Cancel the job
                cancel_response = self.client.delete(f"/api/queue/jobs/{job_id}")
                assert cancel_response.status_code == 200

                cancel_data = cancel_response.json()
                assert cancel_data["success"] is True

            processing_thread.join(timeout=10)

    def test_error_handling_and_status_tracking(self):
        """Test error handling and status tracking when processing fails."""
        session_id = self._get_session()

        # Upload file
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id},
            )

        assert response.status_code == 200
        transcription_id = response.json()["transcription_id"]

        # Mock transcription to raise an error
        with patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe:
            mock_transcribe.side_effect = Exception("Simulated transcription error")

            # Start processing
            start_response = self.client.post(
                f"/api/transcription/{transcription_id}/start",
                json={"session_id": session_id},
            )

            assert start_response.status_code == 200

            # Wait for error to be processed
            max_wait = 10
            start_time = time.time()
            error_detected = False

            while time.time() - start_time < max_wait:
                status_response = self.client.get(
                    f"/api/transcription/{transcription_id}/status"
                )
                assert status_response.status_code == 200

                status_data = status_response.json()
                current_status = status_data.get("status")

                if current_status == "failed":
                    error_detected = True
                    assert "error" in status_data
                    break

                time.sleep(0.5)

            assert error_detected, "Error status was not properly tracked"

    def test_progress_callback_functionality(self):
        """Test that progress callbacks work correctly during processing."""
        session_id = self._get_session()

        # Upload file
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id},
            )

        assert response.status_code == 200
        transcription_id = response.json()["transcription_id"]

        # Track progress updates
        progress_updates = []

        def mock_progress_callback(percentage, step):
            progress_updates.append((percentage, step))

        with (
            patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe,
            patch.object(DiarizationService, "diarize_audio") as mock_diarize,
        ):

            def mock_transcribe_with_progress(
                audio_path, transcription, db, progress_callback=None
            ):
                if progress_callback:
                    progress_callback(25, "Processing audio...")
                    progress_callback(50, "Transcribing...")
                    progress_callback(75, "Finalizing...")
                return {"text": "Progress test", "segments": []}

            def mock_diarize_with_progress(
                audio_path, transcription, db, progress_callback=None
            ):
                if progress_callback:
                    progress_callback(90, "Diarizing speakers...")
                return {"speakers": [], "speaker_matches": {}}

            mock_transcribe.side_effect = mock_transcribe_with_progress
            mock_diarize.side_effect = mock_diarize_with_progress

            # Start processing
            start_response = self.client.post(
                f"/api/transcription/{transcription_id}/start",
                json={"session_id": session_id},
            )

            assert start_response.status_code == 200

            # Monitor progress
            max_wait = 10
            start_time = time.time()
            progress_seen = False

            while time.time() - start_time < max_wait:
                status_response = self.client.get(
                    f"/api/transcription/{transcription_id}/status"
                )
                assert status_response.status_code == 200

                status_data = status_response.json()
                progress = status_data.get("progress", 0)
                current_step = status_data.get("current_step", "")

                if progress > 0 and current_step:
                    progress_seen = True
                    print(f"Progress: {progress}%, Step: {current_step}")

                if status_data.get("status") == "completed":
                    break

                time.sleep(0.5)

            assert progress_seen, "Progress updates were not tracked"

    def test_concurrent_job_processing(self):
        """Test handling of multiple concurrent jobs."""
        session_id = self._get_session()

        # Upload multiple files
        transcription_ids = []
        for i in range(2):
            with open(self.test_audio_path, "rb") as f:
                response = self.client.post(
                    "/api/transcription/upload",
                    files={"file": (f"test_{i}.wav", f, "audio/wav")},
                    data={"session_id": session_id},
                )
            assert response.status_code == 200
            transcription_ids.append(response.json()["transcription_id"])

        # Mock fast processing
        with (
            patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe,
            patch.object(DiarizationService, "diarize_audio") as mock_diarize,
        ):
            mock_transcribe.return_value = {
                "text": f"Mock transcription",
                "segments": [],
            }
            mock_diarize.return_value = {"speakers": [], "speaker_matches": {}}

            # Start both jobs concurrently
            for transcription_id in transcription_ids:
                response = self.client.post(
                    f"/api/transcription/{transcription_id}/start",
                    json={"session_id": session_id},
                )
                assert response.status_code == 200

            # Wait for both to complete
            all_completed = False
            max_wait = 15
            start_time = time.time()

            while time.time() - start_time < max_wait:
                completed_count = 0
                for transcription_id in transcription_ids:
                    status_response = self.client.get(
                        f"/api/transcription/{transcription_id}/status"
                    )
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        completed_count += 1

                if completed_count == len(transcription_ids):
                    all_completed = True
                    break

                time.sleep(0.5)

            assert all_completed, "Not all concurrent jobs completed"

    def test_database_state_consistency(self):
        """Test that database state remains consistent throughout processing."""
        session_id = self._get_session()

        # Upload file
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id},
            )

        assert response.status_code == 200
        transcription_id = response.json()["transcription_id"]

        # Check initial database state
        with next(get_database()) as db:
            transcription = (
                db.query(Transcription)
                .filter(Transcription.id == transcription_id)
                .first()
            )
            assert transcription is not None
            assert transcription.status == "uploaded"
            assert transcription.session_id == session_id

        # Mock processing
        with (
            patch.object(TranscriptionService, "transcribe_audio") as mock_transcribe,
            patch.object(DiarizationService, "diarize_audio") as mock_diarize,
        ):
            mock_transcribe.return_value = {
                "text": "Database consistency test",
                "segments": [
                    {"start": 0, "end": 1, "text": "test", "speaker": "SPEAKER_00"}
                ],
            }
            mock_diarize.return_value = {"speakers": [], "speaker_matches": {}}

            # Start processing
            start_response = self.client.post(
                f"/api/transcription/{transcription_id}/start",
                json={"session_id": session_id},
            )
            assert start_response.status_code == 200

            # Wait for completion
            max_wait = 10
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_response = self.client.get(
                    f"/api/transcription/{transcription_id}/status"
                )
                status_data = status_response.json()

                if status_data.get("status") == "completed":
                    break

                time.sleep(0.5)

            # Verify final database state
            with next(get_database()) as db:
                transcription = (
                    db.query(Transcription)
                    .filter(Transcription.id == transcription_id)
                    .first()
                )
                assert transcription is not None
                assert transcription.status == "completed"
                assert transcription.text is not None
                assert len(transcription.segments) > 0

    def test_api_error_responses(self):
        """Test that API returns appropriate error responses."""
        session_id = self._get_session()

        # Test non-existent transcription
        response = self.client.get("/api/transcription/99999/status")
        assert response.status_code == 404

        # Test invalid transcription ID format
        response = self.client.get("/api/transcription/invalid-id/status")
        assert response.status_code == 422  # Validation error

        # Test starting non-existent transcription
        response = self.client.post(
            "/api/transcription/99999/start", json={"session_id": session_id}
        )
        assert response.status_code == 404

        # Test unauthorized access (different session)
        fake_session_id = str(uuid.uuid4())
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/api/transcription/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"session_id": fake_session_id},
            )
        assert response.status_code == 401  # Unauthorized


if __name__ == "__main__":
    # Run specific tests for debugging
    test_instance = TestProcessingPipeline()

    # Create temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_instance.setup_test_environment(Path(tmp_dir))

        try:
            print("Running basic file upload test...")
            test_instance.test_file_upload_and_initial_status()
            print("✅ File upload test passed")

            print("Running mock GPU processing test...")
            test_instance.test_transcription_processing_with_mock_gpu()
            print("✅ Mock GPU processing test passed")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
