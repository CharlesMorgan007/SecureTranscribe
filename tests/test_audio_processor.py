"""
Test audio processor service for SecureTranscribe.
Tests audio file validation, conversion, and feature extraction.
"""

import os
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from app.services.audio_processor import AudioProcessor
from app.utils.exceptions import AudioProcessingError


class TestAudioProcessor:
    """Test the AudioProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_temp_audio_file(
        self, filename="test.wav", content=b"fake audio content"
    ):
        """Create a temporary audio file for testing."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    @patch("app.services.audio_processor.magic.from_file")
    @patch("app.services.audio_processor.sf.SoundFile")
    def test_validate_file_success(self, mock_soundfile, mock_magic):
        """Test successful file validation."""
        # Mock soundfile
        mock_file = MagicMock()
        mock_file.samplerate = 44100
        mock_file.channels = 2
        mock_file.__len__ = MagicMock(return_value=44100)  # 1 second
        mock_soundfile.return_value.__enter__.return_value = mock_file

        # Mock magic
        mock_magic.return_value = "audio/wav"

        # Create test file
        file_path = self.create_temp_audio_file("test.wav")

        result = self.processor.validate_file(file_path)

        assert result["is_valid"] is True
        assert result["original_filename"] == "test.wav"
        assert result["file_format"] == "wav"
        assert result["duration"] == 1.0
        assert result["channels"] == 2
        assert result["sample_rate"] == 44100
        assert result["mime_type"] == "audio/wav"

    def test_validate_file_not_exists(self):
        """Test validation of non-existent file."""
        with pytest.raises(AudioProcessingError, match="File does not exist"):
            self.processor.validate_file("nonexistent.wav")

    def test_validate_file_too_large(self):
        """Test validation of file that's too large."""
        # Create a large file (mocked)
        large_content = b"x" * (600 * 1024 * 1024)  # 600MB
        file_path = self.create_temp_audio_file("large.wav", large_content)

        with patch(
            "app.services.audio_processor.magic.from_file", return_value="audio/wav"
        ):
            with patch("app.services.audio_processor.sf.SoundFile") as mock_soundfile:
                mock_file = MagicMock()
                mock_file.samplerate = 44100
                mock_file.channels = 2
                mock_file.__len__ = MagicMock(return_value=44100)
                mock_soundfile.return_value.__enter__.return_value = mock_file

                with pytest.raises(
                    AudioProcessingError, match="File size .* exceeds maximum"
                ):
                    self.processor.validate_file(file_path)

    def test_validate_file_unsupported_format(self):
        """Test validation of unsupported file format."""
        file_path = self.create_temp_audio_file("test.txt")

        with pytest.raises(AudioProcessingError, match="Unsupported file format"):
            self.processor.validate_file(file_path)

    def test_validate_file_invalid_mime_type(self):
        """Test validation of file with invalid MIME type."""
        file_path = self.create_temp_audio_file("test.wav")

        with patch(
            "app.services.audio_processor.magic.from_file", return_value="text/plain"
        ):
            with pytest.raises(AudioProcessingError, match="Invalid file type"):
                self.processor.validate_file(file_path)

    def test_validate_file_zero_duration(self):
        """Test validation of file with zero duration."""
        file_path = self.create_temp_audio_file("empty.wav")

        with patch(
            "app.services.audio_processor.magic.from_file", return_value="audio/wav"
        ):
            with patch("app.services.audio_processor.sf.SoundFile") as mock_soundfile:
                mock_file = MagicMock()
                mock_file.samplerate = 44100
                mock_file.channels = 2
                mock_file.__len__ = MagicMock(return_value=0)  # Zero duration
                mock_soundfile.return_value.__enter__.return_value = mock_file

                with pytest.raises(
                    AudioProcessingError, match="Audio file has zero duration"
                ):
                    self.processor.validate_file(file_path)

    @patch("app.services.audio_processor.AudioSegment")
    def test_convert_to_wav_success(self, mock_audio_segment):
        """Test successful audio conversion to WAV."""
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.export = MagicMock()

        input_path = self.create_temp_audio_file("test.mp3")
        output_path = os.path.join(self.temp_dir, "output.wav")

        result = self.processor.convert_to_wav(input_path, output_path)

        assert result == output_path
        mock_audio_segment.from_file.assert_called_once_with(input_path)
        mock_audio.set_channels.assert_called_once_with(1)
        mock_audio.set_frame_rate.assert_called_once_with(16000)
        mock_audio.export.assert_called_once()

    @patch("app.services.audio_processor.AudioSegment")
    def test_convert_to_wav_auto_output_path(self, mock_audio_segment):
        """Test audio conversion with automatic output path generation."""
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.export = MagicMock()

        input_path = self.create_temp_audio_file("test.mp3")

        result = self.processor.convert_to_wav(input_path)

        assert result.endswith(".wav")
        assert "converted_test" in result
        mock_audio_segment.from_file.assert_called_once_with(input_path)

    @patch("app.services.audio_processor.AudioSegment")
    def test_convert_to_wav_failure(self, mock_audio_segment):
        """Test audio conversion failure."""
        mock_audio_segment.from_file.side_effect = Exception("Conversion failed")

        input_path = self.create_temp_audio_file("test.mp3")

        with pytest.raises(AudioProcessingError, match="Failed to convert audio file"):
            self.processor.convert_to_wav(input_path)

    @patch("app.services.audio_processor.librosa.load")
    def test_preprocess_audio_success(self, mock_librosa_load):
        """Test successful audio preprocessing."""
        # Mock librosa.load
        mock_audio_data = np.random.randn(16000)  # 1 second of audio
        mock_librosa_load.return_value = (mock_audio_data, 16000)

        file_path = self.create_temp_audio_file("test.wav")

        audio_data, sample_rate = self.processor.preprocess_audio(file_path)

        assert sample_rate == 16000
        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) == len(mock_audio_data)
        mock_librosa_load.assert_called_once_with(
            file_path, sr=16000, mono=True, dtype=np.float32
        )

    @patch("app.services.audio_processor.librosa.load")
    def test_preprocess_audio_failure(self, mock_librosa_load):
        """Test audio preprocessing failure."""
        mock_librosa_load.side_effect = Exception("Processing failed")

        file_path = self.create_temp_audio_file("test.wav")

        with pytest.raises(AudioProcessingError, match="Audio preprocessing failed"):
            self.processor.preprocess_audio(file_path)

    @patch("app.services.audio_processor.librosa.load")
    @patch("app.services.audio_processor.librosa.feature")
    @patch("app.services.audio_processor.librosa.beat")
    def test_extract_audio_features_success(
        self, mock_beat, mock_feature, mock_librosa_load
    ):
        """Test successful audio feature extraction."""
        # Mock audio data
        mock_audio_data = np.random.randn(16000)
        mock_librosa_load.return_value = (mock_audio_data, 16000)

        # Mock features
        mock_feature.spectral_centroid.return_value = np.array([[2000]])
        mock_feature.spectral_rolloff.return_value = np.array([[8000]])
        mock_feature.spectral_bandwidth.return_value = np.array([[1000]])
        mock_feature.zero_crossing_rate.return_value = np.array([[0.1]])
        mock_feature.mfcc.return_value = np.random.randn(
            13, 87
        )  # 13 MFCCs for ~1 second
        mock_beat.beat_track.return_value = (120, np.array([0, 0.5]))

        file_path = self.create_temp_audio_file("test.wav")

        features = self.processor.extract_audio_features(file_path)

        # Check that all expected features are present
        expected_features = [
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_bandwidth",
            "zero_crossing_rate",
            "mfcc_mean",
            "mfcc_std",
            "avg_pitch",
            "pitch_std",
            "pitch_min",
            "pitch_max",
            "rms_energy",
            "rms_std",
            "tempo",
        ]

        for feature in expected_features:
            assert feature in features

    @patch("app.services.audio_processor.librosa.load")
    def test_extract_audio_features_failure(self, mock_librosa_load):
        """Test audio feature extraction failure."""
        mock_librosa_load.side_effect = Exception("Feature extraction failed")

        file_path = self.create_temp_audio_file("test.wav")

        with pytest.raises(AudioProcessingError, match="Feature extraction failed"):
            self.processor.extract_audio_features(file_path)

    @patch("app.services.audio_processor.AudioSegment")
    def test_create_audio_segments_success(self, mock_audio_segment):
        """Test successful creation of audio segments."""
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.__getitem__ = MagicMock(return_value=mock_audio)
        mock_audio.export = MagicMock()

        # Mock segments
        segments = [
            {"speaker": "Speaker_1", "start_time": 0, "end_time": 5, "id": 1},
            {"speaker": "Speaker_2", "start_time": 5, "end_time": 10, "id": 2},
        ]

        file_path = self.create_temp_audio_file("test.wav")

        result = self.processor.create_audio_segments(file_path, segments)

        assert len(result) == 2
        assert "Speaker_1" in result
        assert "Speaker_2" in result

        # Check that export was called for each segment
        assert mock_audio.export.call_count == 2

    @patch("app.services.audio_processor.AudioSegment")
    def test_create_audio_segments_failure(self, mock_audio_segment):
        """Test audio segment creation failure."""
        mock_audio_segment.from_file.side_effect = Exception("Segment creation failed")

        segments = [{"speaker": "Speaker_1", "start_time": 0, "end_time": 5}]
        file_path = self.create_temp_audio_file("test.wav")

        with pytest.raises(
            AudioProcessingError, match="Failed to create audio segments"
        ):
            self.processor.create_audio_segments(file_path, segments)

    @patch("app.services.audio_processor.AudioSegment")
    def test_chunk_audio_success(self, mock_audio_segment):
        """Test successful audio chunking."""
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=60000)  # 60 seconds in ms
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.__getitem__ = MagicMock(return_value=mock_audio)
        mock_audio.export = MagicMock()

        file_path = self.create_temp_audio_file("long_audio.wav")

        chunks = self.processor.chunk_audio(file_path, chunk_length=30, overlap=5)

        # Should create 2 chunks for 60 seconds with 30s chunks and 5s overlap
        assert len(chunks) == 2
        assert all(chunk.endswith(".wav") for chunk in chunks)

    @patch("app.services.audio_processor.AudioSegment")
    def test_chunk_audio_failure(self, mock_audio_segment):
        """Test audio chunking failure."""
        mock_audio_segment.from_file.side_effect = Exception("Chunking failed")

        file_path = self.create_temp_audio_file("test.wav")

        with pytest.raises(AudioProcessingError, match="Failed to chunk audio"):
            self.processor.chunk_audio(file_path)

    def test_cleanup_temp_files_success(self):
        """Test successful cleanup of temporary files."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            file_path = self.create_temp_audio_file(f"temp_{i}.wav")
            temp_files.append(file_path)

        # Verify files exist
        assert all(os.path.exists(f) for f in temp_files)

        # Cleanup
        self.processor.cleanup_temp_files(temp_files)

        # Verify files are deleted
        assert not any(os.path.exists(f) for f in temp_files)

    def test_cleanup_temp_files_nonexistent(self):
        """Test cleanup of non-existent files (should not raise error)."""
        nonexistent_files = [
            os.path.join(self.temp_dir, "nonexistent1.wav"),
            os.path.join(self.temp_dir, "nonexistent2.wav"),
        ]

        # Should not raise an error
        self.processor.cleanup_temp_files(nonexistent_files)

    def test_get_audio_format_info(self):
        """Test audio format information retrieval."""
        format_info = AudioProcessor.get_audio_format_info()

        expected_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

        for fmt in expected_formats:
            assert fmt in format_info
            assert "name" in format_info[fmt]
            assert "description" in format_info[fmt]

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create test audio data
        audio_data = np.array([0.5, -0.8, 0.3, -0.6])
        normalized = self.processor._normalize_audio(audio_data)

        # Should be normalized to max absolute value of 1
        assert np.max(np.abs(normalized)) == 1.0
        assert normalized.shape == audio_data.shape

    def test_normalize_audio_zeros(self):
        """Test normalization of zero audio."""
        audio_data = np.zeros(100)
        normalized = self.processor._normalize_audio(audio_data)

        assert np.array_equal(normalized, audio_data)

    def test_generate_file_hash(self):
        """Test file hash generation."""
        file_path = self.create_temp_audio_file("test.wav", b"test content")

        hash1 = self.processor._generate_file_hash(file_path)
        hash2 = self.processor._generate_file_hash(file_path)

        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 length

    @pytest.mark.parametrize("file_format", ["mp3", "wav", "m4a", "flac", "ogg"])
    def test_supported_formats(self, file_format):
        """Test that all supported formats are accepted."""
        filename = f"test.{file_format}"
        file_path = self.create_temp_audio_file(filename)

        with patch(
            "app.services.audio_processor.magic.from_file",
            return_value=f"audio/{file_format}",
        ):
            with patch("app.services.audio_processor.sf.SoundFile") as mock_soundfile:
                mock_file = MagicMock()
                mock_file.samplerate = 44100
                mock_file.channels = 2
                mock_file.__len__ = MagicMock(return_value=44100)
                mock_soundfile.return_value.__enter__.return_value = mock_file

                result = self.processor.validate_file(file_path)
                assert result["file_format"] == file_format


if __name__ == "__main__":
    pytest.main([__file__])
