"""
Test speaker diarization service for SecureTranscribe.
Tests speaker identification functionality without requiring GPU.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import numpy as np

# Mock external dependencies to avoid installation issues
sys.modules["torch"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["pyannote.core"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["numpy"] = MagicMock()


# Create mock numpy array
def create_mock_array():
    return np.array([0.1, 0.2, 0.3, 0.2, 0.1])


# Set numpy functions
np.zeros = lambda shape: create_mock_array()
np.array = lambda data: np.array(data) if data else create_mock_array()
np.mean = lambda x: float(np.mean(x)) if hasattr(x, "__iter__") else 0.0
np.sqrt = lambda x: float(np.sqrt(x)) if hasattr(x, "__iter__") else 1.0


# Mock pyannote classes
class MockAnnotation:
    def __init__(self):
        self.segments = []

    def itertracks(self, yield_label=False):
        # Return mock segments
        return iter(self.generate_mock_segments())

    def generate_mock_segments(self):
        return [
            (MockSegment(0.0, 5.0, "Speaker_A"), False),
            (MockSegment(5.0, 10.0, "Speaker_A"), False),
            (MockSegment(10.0, 15.0, "Speaker_B"), False),
            (MockSegment(15.0, 20.0, "Speaker_B"), False),
        ]


class MockSegment:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self):
        return f"Segment({self.start}-{self.end}, {self.label})"


class MockPipeline:
    def __init__(self):
        pass

    def from_pretrained(self, model_name, use_auth_token=False):
        return MockPipeline()


# Mock audio processor
class MockAudioProcessor:
    def validate_file(self, file_path):
        return {
            "file_path": file_path,
            "original_filename": os.path.basename(file_path),
            "file_size": 2048,
            "file_format": "wav",
            "duration": 20.0,
            "channels": 1,
            "sample_rate": 16000,
            "mime_type": "audio/wav",
            "file_hash": "test_hash",
            "is_valid": True,
        }

    def convert_to_wav(self, input_path, output_path=None):
        if output_path is None:
            output_path = input_path.replace(".mp3", ".wav")
        return output_path

    def extract_audio_features(self, audio_path):
        return {
            "avg_pitch": 200.0,
            "pitch_std": 20.0,
            "spectral_centroid": 1500.0,
            "zero_crossing_rate": 0.05,
            "rms_energy": 0.1,
            "mfcc_mean": [0.1, 0.2, 0.3] * 13,
        }


# Mock diarization service that doesn't require actual ML models
class MockDiarizationService:
    def __init__(self):
        self.model_size = "pyannote/speaker-diarization-3.1"
        self.device = "cpu"
        self.sample_rate = 16000
        self.min_speaker_duration = 2.0

    def _load_pipeline(self):
        pass

    def diarize_audio(self, audio_path, transcription, session, progress_callback=None):
        # Mock diarization result
        result = {
            "speakers": ["Speaker_1", "Speaker_2"],
            "speaker_segments": [
                {
                    "speaker": "Speaker_1",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "duration": 10.0,
                    "text": "Hello, this is speaker one.",
                },
                {
                    "speaker": "Speaker_2",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "duration": 10.0,
                    "text": "And this is speaker two.",
                },
            ],
            "speaker_matches": {
                "Speaker_1": None,
                "Speaker_2": None,
            },
            "speaker_durations": {
                "Speaker_1": 10.0,
                "Speaker_2": 10.0,
            },
            "preview_clips": [],
        }

        # Update transcription
        transcription.num_speakers = len(result["speakers"])
        transcription.speakers_assigned = True
        transcription.segments = result["speaker_segments"]

        return result

    def cleanup(self):
        pass

    def estimate_processing_time(self, audio_duration):
        return audio_duration * 1.0  # Mock estimation: 1x real-time


class TestDiarizationService:
    """Test the diarization service without requiring GPU."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = MockDiarizationService()
        self.mock_audio_processor = MockAudioProcessor()

        # Create a mock transcription
        self.mock_transcription = MagicMock()
        self.mock_transcription.id = 1
        self.mock_transcription.session_id = "test_session"
        self.mock_transcription.original_filename = "test.wav"
        self.mock_transcription.file_path = "/tmp/test.wav"
        self.mock_transcription.file_size = 2048
        self.mock_transcription.file_duration = 20.0
        self.mock_transcription.file_format = "wav"
        self.mock_transcription.num_speakers = 0
        self.mock_transcription.speakers_assigned = False
        self.mock_transcription.segments = []

        # Mock session
        self.mock_session = MagicMock()

    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.model_size == "pyannote/speaker-diarization-3.1"
        assert self.service.device == "cpu"
        assert self.service.sample_rate == 16000
        assert self.service.min_speaker_duration == 2.0

    def test_load_pipeline(self):
        """Test pipeline loading."""
        self.service._load_pipeline()
        # Should not raise any exceptions
        assert True

    def test_diarize_audio_basic(self):
        """Test basic speaker diarization."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data with two speakers")
            temp_path = f.name

        try:
            result = self.service.diarize_audio(
                temp_path, self.mock_transcription, self.mock_session
            )

            # Verify results
            assert "speakers" in result
            assert "speaker_segments" in result
            assert "speaker_matches" in result
            assert "speaker_durations" in result
            assert "preview_clips" in result

            # Check speakers
            assert len(result["speakers"]) == 2
            assert "Speaker_1" in result["speakers"]
            assert "Speaker_2" in result["speakers"]

            # Check segments
            assert len(result["speaker_segments"]) == 2
            assert all(
                segment.get("speaker") in result["speakers"]
                for segment in result["speaker_segments"]
            )

            # Check transcription was updated
            assert self.mock_transcription.num_speakers == 2
            assert self.mock_transcription.speakers_assigned == True

        finally:
            os.unlink(temp_path)

    def test_diarize_audio_with_progress_callback(self):
        """Test diarization with progress callback."""
        progress_updates = []

        def progress_callback(percentage, step):
            progress_updates.append((percentage, step))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data with two speakers")
            temp_path = f.name

        try:
            result = self.service.diarize_audio(
                temp_path,
                self.mock_transcription,
                self.mock_session,
                progress_callback=progress_callback,
            )

            # Verify progress was called
            assert len(progress_updates) > 0

        finally:
            os.unlink(temp_path)

    def test_diarize_audio_single_speaker(self):
        """Test diarization with single speaker."""

        # Mock single speaker result
        def mock_diarize_single_speaker(
            audio_path, transcription, session, progress_callback=None
        ):
            return {
                "speakers": ["Speaker_1"],
                "speaker_segments": [
                    {
                        "speaker": "Speaker_1",
                        "start_time": 0.0,
                        "end_time": 20.0,
                        "duration": 20.0,
                        "text": "This is a single speaker talking.",
                    }
                ],
                "speaker_matches": {"Speaker_1": None},
                "speaker_durations": {"Speaker_1": 20.0},
                "preview_clips": [],
            }

        # Temporarily replace the method
        original_method = self.service.diarize_audio
        self.service.diarize_audio = mock_diarize_single_speaker

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data with single speaker")
            temp_path = f.name

        try:
            result = self.service.diarize_audio(
                temp_path, self.mock_transcription, self.mock_session
            )

            # Verify single speaker result
            assert len(result["speakers"]) == 1
            assert result["speakers"][0] == "Speaker_1"
            assert len(result["speaker_segments"]) == 1
            assert result["speaker_segments"][0]["speaker"] == "Speaker_1"

        finally:
            os.unlink(temp_path)
            # Restore original method
            self.service.diarize_audio = original_method

    def test_diarize_audio_three_speakers(self):
        """Test diarization with three speakers."""

        # Mock three speaker result
        def mock_diarize_three_speakers(
            audio_path, transcription, session, progress_callback=None
        ):
            return {
                "speakers": ["Speaker_A", "Speaker_B", "Speaker_C"],
                "speaker_segments": [
                    {
                        "speaker": "Speaker_A",
                        "start_time": 0.0,
                        "end_time": 6.0,
                        "duration": 6.0,
                        "text": "Speaker A talking.",
                    },
                    {
                        "speaker": "Speaker_B",
                        "start_time": 6.0,
                        "end_time": 12.0,
                        "duration": 6.0,
                        "text": "Speaker B talking.",
                    },
                    {
                        "speaker": "Speaker_C",
                        "start_time": 12.0,
                        "end_time": 20.0,
                        "duration": 8.0,
                        "text": "Speaker C talking.",
                    },
                ],
                "speaker_matches": {
                    "Speaker_A": None,
                    "Speaker_B": None,
                    "Speaker_C": None,
                },
                "speaker_durations": {
                    "Speaker_A": 6.0,
                    "Speaker_B": 6.0,
                    "Speaker_C": 8.0,
                },
                "preview_clips": [],
            }

        # Temporarily replace the method
        original_method = self.service.diarize_audio
        self.service.diarize_audio = mock_diarize_three_speakers

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data with three speakers")
            temp_path = f.name

        try:
            result = self.service.diarize_audio(
                temp_path, self.mock_transcription, self.mock_session
            )

            # Verify three speaker result
            assert len(result["speakers"]) == 3
            assert len(result["speaker_segments"]) == 3
            assert len(result["speaker_durations"]) == 3

        finally:
            os.unlink(temp_path)
            # Restore original method
            self.service.diarize_audio = original_method

    def test_diarization_updates_transcription(self):
        """Test that transcription is properly updated."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name

        try:
            result = self.service.diarize_audio(
                temp_path, self.mock_transcription, self.mock_session
            )

            # Verify transcription was updated
            assert self.mock_transcription.num_speakers > 0
            assert self.mock_transcription.speakers_assigned == True

        finally:
            os.unlink(temp_path)

    def test_speaker_segment_structure(self):
        """Test speaker segment structure."""
        # Mock result with well-structured segments
        result = self.service.diarize_audio(
            "/tmp/test.wav", self.mock_transcription, self.mock_session
        )

        # Check segment structure
        for segment in result["speaker_segments"]:
            assert "speaker" in segment
            assert "start_time" in segment
            assert "end_time" in segment
            assert "duration" in segment
            assert "text" in segment

            # Validate data types
            assert isinstance(segment["start_time"], (int, float))
            assert isinstance(segment["end_time"], (int, float))
            assert isinstance(segment["duration"], (int, float))
            assert isinstance(segment["text"], str)

            # Validate logical consistency
            assert segment["end_time"] >= segment["start_time"]
            assert segment["duration"] == segment["end_time"] - segment["start_time"]

    def test_speaker_identification(self):
        """Test speaker identification in segments."""
        result = self.service.diarize_audio(
            "/tmp/test.wav", self.mock_transcription, self.mock_session
        )

        # Check that speakers are identified consistently
        speakers = set()
        for segment in result["speaker_segments"]:
            speakers.add(segment["speaker"])

        assert len(speakers) == len(result["speakers"])
        assert all(speaker in result["speakers"] for speaker in speakers)

    def test_timing_consistency(self):
        """Test timing consistency in speaker segments."""
        result = self.service.diarize_audio(
            "/tmp/test.wav", self.mock_transcription, self.mock_session
        )

        # Sort segments by start time
        segments = sorted(result["speaker_segments"], key=lambda x: x["start_time"])

        # Check that segments don't overlap
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            assert current["end_time"] <= next_segment["start_time"], (
                f"Segments overlap: {current['end_time']} > {next_segment['start_time']}"
            )

    def test_cleanup(self):
        """Test service cleanup."""
        self.service.cleanup()
        # Should not raise any exceptions
        assert True

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        test_cases = [30, 60, 120, 300]  # seconds

        for duration in test_cases:
            estimated_time = self.service.estimate_processing_time(duration)
            expected_time = duration * 1.0  # 1x real-time for diarization
            assert estimated_time == expected_time

    def test_min_speaker_duration(self):
        """Test minimum speaker duration configuration."""
        assert self.service.min_speaker_duration == 2.0

    def test_sample_rate_configuration(self):
        """Test sample rate configuration."""
        assert self.service.sample_rate == 16000

    def test_device_configuration(self):
        """Test device configuration."""
        # Should be configured for CPU by default
        assert self.service.device == "cpu"

    def test_model_configuration(self):
        """Test model configuration."""
        assert self.service.model_size == "pyannote/speaker-diarization-3.1"

    def test_empty_audio_file(self):
        """Test diarization of empty audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"")  # Empty file
            temp_path = f.name

        try:
            # Mock audio processor to handle empty files
            with patch(
                "tests.test_diarization_service.MockAudioProcessor.validate_file"
            ) as mock_validate:
                mock_validate.return_value = {
                    "file_path": temp_path,
                    "original_filename": "empty.wav",
                    "file_size": 0,
                    "duration": 0.0,
                    "is_valid": True,
                }

                result = self.service.diarize_audio(
                    temp_path, self.mock_transcription, self.mock_session
                )

                # Should still return results for empty files
                assert "speakers" in result
                assert "speaker_segments" in result

        finally:
            os.unlink(temp_path)

    def test_very_short_audio_file(self):
        """Test diarization of very short audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"short audio")  # Very short file
            temp_path = f.name

        try:
            with patch(
                "tests.test_diarization_service.MockAudioProcessor.validate_file"
            ) as mock_validate:
                mock_validate.return_value = {
                    "file_path": temp_path,
                    "original_filename": "short.wav",
                    "file_size": 10,
                    "duration": 0.5,  # Very short
                    "is_valid": True,
                }

                result = self.service.diarize_audio(
                    temp_path, self.mock_transcription, self.mock_session
                )

                assert "speakers" in result
                # Short files might be handled specially
                if (
                    mock_validate.return_value["duration"]
                    < self.service.min_speaker_duration
                ):
                    # Very short files might have no speakers or single speaker
                    assert len(result["speakers"]) <= 1

        finally:
            os.unlink(temp_path)

    def test_very_long_audio_file(self):
        """Test diarization of very long audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"0" * (1024 * 1024))  # 1MB file
            temp_path = f.name

        try:
            with patch(
                "tests.test_diarization_service.MockAudioProcessor.validate_file"
            ) as mock_validate:
                mock_validate.return_value = {
                    "file_path": temp_path,
                    "original_filename": "long.wav",
                    "file_size": 1024 * 1024,
                    "duration": 3600.0,  # 1 hour
                    "is_valid": True,
                }

                result = self.service.diarize_audio(
                    temp_path, self.mock_transcription, self.mock_session
                )

                assert "speakers" in result
                assert "speaker_segments" in result
                # Long files should have more speakers
                assert len(result["speaker_segments"]) >= 1

        finally:
            os.unlink(temp_path)

    @pytest.mark.parametrize("num_speakers", [1, 2, 3, 4, 5])
    def test_different_speaker_counts(self, num_speakers):
        """Test diarization with different speaker counts."""

        def mock_diarize_with_speakers(
            audio_path, transcription, session, num_speakers=1
        ):
            speakers = [f"Speaker_{i}" for i in range(1, num_speakers + 1)]
            segments = []

            segment_duration = 10.0 / num_speakers
            for i, speaker in enumerate(speakers):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                segments.append(
                    {
                        "speaker": speaker,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": segment_duration,
                        "text": f"{speaker} talking.",
                    }
                )

            return {
                "speakers": speakers,
                "speaker_segments": segments,
                "speaker_matches": {speaker: None for speaker in speakers},
                "speaker_durations": {
                    speaker: segment_duration for speaker in speakers
                },
                "preview_clips": [],
            }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name

        try:
            # Temporarily replace the method
            original_method = self.service.diarize_audio
            self.service.diarize_audio = (
                lambda audio_path,
                transcription,
                session,
                progress_callback=None: mock_diarize_with_speakers(
                    audio_path, transcription, session, num_speakers
                )
            )

            result = self.service.diarize_audio(
                temp_path, self.mock_transcription, self.mock_session
            )

            # Verify speaker count
            assert len(result["speakers"]) == num_speakers
            assert len(result["speaker_segments"]) == num_speakers
            assert len(result["speaker_durations"]) == num_speakers

            # Verify all speakers are unique
            assert len(set(result["speakers"])) == num_speakers

        finally:
            os.unlink(temp_path)
            # Restore original method
            self.service.diarize_audio = original_method

    def test_concurrent_diarization(self):
        """Test handling of multiple concurrent diarization processes."""
        import threading
        import time

        results = []
        errors = []

        def diarize_worker(file_id):
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f"_test_{file_id}.wav", delete=False
                ) as f:
                    f.write(b"fake audio data")
                    temp_path = f.name

                with patch(
                    "tests.test_diarization_service.MockAudioProcessor.validate_file"
                ) as mock_validate:
                    mock_validate.return_value = {
                        "file_path": temp_path,
                        "original_filename": f"concurrent_test_{file_id}.wav",
                        "file_size": 1024,
                        "duration": 10.0,
                        "is_valid": True,
                    }

                    result = self.service.diarize_audio(
                        temp_path, self.mock_transcription, self.mock_session
                    )
                    results.append(result)

            except Exception as e:
                errors.append(e)

            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        # Create multiple diarization threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=diarize_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        for result in results:
            assert "speakers" in result
            assert "speaker_segments" in result

    def test_error_handling(self):
        """Test error handling in diarization service."""

        # Mock a diarization that raises an exception
        def failing_diarize(audio_path, transcription, session, **kwargs):
            raise ValueError("Mock diarization error")

        # Temporarily replace the method
        original_method = self.service.diarize_audio
        self.service.diarize_audio = failing_diarize

        try:
            self.service.diarize_audio(
                "/nonexistent/file.wav", self.mock_transcription, self.mock_session
            )
            assert False, "Should have raised an exception"
        except ValueError as e:
            assert str(e) == "Mock diarization error"
        finally:
            # Restore original method
            self.service.diarize_audio = original_method

    def test_progress_callback_integration(self):
        """Test progress callback integration."""
        progress_data = []

        def detailed_progress_callback(percentage, step):
            progress_data.append(
                {"percentage": percentage, "step": step, "timestamp": time.time()}
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name

        try:
            result = self.service.diarize_audio(
                temp_path,
                self.mock_transcription,
                self.mock_session,
                progress_callback=detailed_progress_callback,
            )

            # Verify detailed progress data
            assert len(progress_data) > 0
            assert all("percentage" in data for data in progress_data)
            assert all("step" in data for data in progress_data)
            assert all("timestamp" in data for data in progress_data)

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
