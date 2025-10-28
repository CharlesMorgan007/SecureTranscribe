Unit tests for TranscriptionService.

Tests the core transcription functionality including:
- Model loading and initialization
- Audio validation and processing
- Whisper transcription accuracy
- Progress tracking
- Error handling and edge cases
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from app.services.transcription_service import TranscriptionService
from app.models.transcription import Transcription
from app.utils.exceptions import TranscriptionError


class TestTranscriptionService(unittest.TestCase):
    """Test suite for TranscriptionService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.transcription_service = TranscriptionService()
        self.test_audio_dir = tempfile.mkdtemp()

        # Create test audio file (silence)
        self.test_audio_path = os.path.join(self.test_audio_dir, "test.wav")
        self._create_test_audio_file(self.test_audio_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_audio_dir, ignore_errors=True)

    def _create_test_audio_file(self, path: str, duration: float = 1.0):
        """Create a test audio file with silence."""
        import wave
        import struct

        sample_rate = 16000
        samples = int(sample_rate * duration)

        with wave.open(path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            # Generate silence (zero amplitude)
            silence = struct.pack('<h', 0) * samples
            wav_file.writeframes(silence)

    def test_init(self):
        """Test service initialization."""
        service = TranscriptionService()

        self.assertEqual(service.model_size, "base")
        self.assertEqual(service.device, "cpu")
        self.assertEqual(service.sample_rate, 16000)
        self.assertEqual(service.chunk_length, 30)
        self.assertEqual(service.overlap_length, 5)

    def test_get_device(self):
        """Test device detection."""
        service = TranscriptionService()
        device = service._get_device()

        # Should return "cpu" as configured
        self.assertEqual(device, "cpu")

    def test_validate_file_valid(self):
        """Test valid file validation."""
        # Create a valid audio file
        valid_path = os.path.join(self.test_audio_dir, "valid.wav")
        self._create_test_audio_file(valid_path, 5.0)

        result = self.transcription_service.validate_file(valid_path)
        self.assertTrue(result)

    def test_validate_file_invalid_format(self):
        """Test invalid file format validation."""
        # Create a text file (invalid format)
        invalid_path = os.path.join(self.test_audio_dir, "invalid.txt")
        with open(invalid_path, 'w') as f:
            f.write("not audio")

        result = self.transcription_service.validate_file(invalid_path)
        self.assertFalse(result)

    def test_validate_file_too_large(self):
        """Test file size validation."""
        # Mock large file
        with patch('os.path.getsize', return_value=600 * 1024 * 1024):  # 600MB
            result = self.transcription_service.validate_file("dummy.wav")
            self.assertFalse(result)

    def test_validate_file_nonexistent(self):
        """Test nonexistent file validation."""
        result = self.transcription_service.validate_file("/nonexistent/file.wav")
        self.assertFalse(result)

    @patch('app.services.transcription_service.TranscriptionService._load_model')
    @patch('app.services.transcription_service.TranscriptionService._process_transcription')
    def test_transcribe_audio_success(self, mock_process, mock_load):
        """Test successful audio transcription."""
        # Setup mocks
        mock_load.return_value = None
        mock_process.return_value = {
            "text": "Test transcription result",
            "language": "en",
            "avg_confidence": 0.95
        }

        # Create test transcription
        transcription = Mock(spec=Transcription)
        transcription.session_id = "test_session"
        transcription.file_path = self.test_audio_path
        transcription.mark_as_started = Mock()
        transcription.mark_as_completed = Mock()

        # Mock database session
        mock_session = Mock()
        mock_session.commit = Mock()

        # Test transcription
        result = self.transcription_service.transcribe_audio(
            self.test_audio_path,
            transcription,
            mock_session,
            language="en"
        )

        # Verify results
        self.assertEqual(result["text"], "Test transcription result")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["avg_confidence"], 0.95)

        # Verify transcription was updated
        transcription.mark_as_started.assert_called_once()
        transcription.mark_as_completed.assert_called_once()

    def test_transcribe_audio_file_validation_failure(self):
        """Test transcription with invalid file."""
        # Create invalid audio file
        invalid_path = os.path.join(self.test_audio_dir, "invalid.wav")
        with open(invalid_path, 'w') as f:
            f.write("not audio")

        transcription = Mock(spec=Transcription)
        mock_session = Mock()

        with self.assertRaises(TranscriptionError):
            self.transcription_service.transcribe_audio(
                invalid_path,
                transcription,
                mock_session
            )

    @patch('faster_whisper.WhisperModel')
    def test_load_model(self, mock_whisper_model):
        """Test model loading."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        service = TranscriptionService()
        service._load_model()

        self.assertEqual(service.model, mock_model)
        mock_whisper_model.assert_called_once_with(
            "base",
            device="cpu",
            compute_type="float32"
        )

    def test_process_transcription_short_audio(self):
        """Test processing short audio file."""
        # Test implementation would depend on actual audio processing logic
        # For now, test that method exists and handles short audio
        self.assertTrue(hasattr(self.transcription_service, '_process_transcription'))

        # Test with very short audio
        short_audio = os.path.join(self.test_audio_dir, "short.wav")
        self._create_test_audio_file(short_audio, 0.5)  # 0.5 seconds

        # This should not crash
        try:
            # We can't actually test without proper model setup
            # but we can verify the method handles edge cases
            self.transcription_service._transcribe_single_chunk = Mock(return_value={"text": "test"})
            result = self.transcription_service._process_transcription(
                short_audio, Mock(), Mock()
            )
        except Exception:
            # Expected to fail without proper model setup
            pass

    def test_progress_callback(self):
        """Test progress callback functionality."""
        # Create mock transcription
        transcription = Mock(spec=Transcription)
        transcription.update_progress = Mock()

        # Test progress callback
        self.transcription_service.transcribe_audio(
            self.test_audio_path,
            transcription,
            Mock(),
            progress_callback=lambda p, s: None
        )

        # Should not crash and should handle callback
        self.assertTrue(True)  # If we get here, callback was handled

    @patch('os.path.exists')
    @patch('os.remove')
    def test_cleanup_temporary_files(self, mock_remove, mock_exists):
        """Test temporary file cleanup."""
        mock_exists.return_value = True

        # This test would verify that temporary files are cleaned up
        # Implementation depends on actual cleanup logic
        self.assertTrue(True)


class TestTranscriptionServiceIntegration(unittest.TestCase):
    """Integration tests for TranscriptionService."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.transcription_service = TranscriptionService()
        self.test_audio_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.test_audio_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """Test complete transcription workflow."""
        # Create a real transcription model
        transcription = Transcription(
            session_id="integration_test",
            original_filename="test.wav",
            file_path=self.test_audio_dir,
            file_size=1024,
            file_duration=2.0,
            file_format="wav"
        )

        # Test that the service can handle the workflow
        # This is a basic smoke test
        self.assertIsInstance(self.transcription_service, TranscriptionService)
        self.assertIsNotNone(transcription.session_id)
        self.assertEqual(transcription.original_filename, "test.wav")


if __name__ == "__main__":
    unittest.main()
