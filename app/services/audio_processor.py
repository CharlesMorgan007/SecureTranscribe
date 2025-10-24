"""
Audio processor service for handling audio files, conversion, and preparation.
Handles file validation, format conversion, and audio analysis for transcription.
"""

import os
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import magic

from core.config import get_settings, AUDIO_SETTINGS, SECURITY_SETTINGS
from utils.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor service for handling audio files.
    Manages file validation, conversion, and audio preparation for transcription.
    """

    def __init__(self):
        self.settings = get_settings()
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.supported_formats = AUDIO_SETTINGS["supported_formats"]
        self.max_file_size = self.settings.max_file_size_bytes

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate uploaded audio file.

        Args:
            file_path: Path to the uploaded file

        Returns:
            Dictionary containing file information

        Raises:
            AudioProcessingError: If file validation fails
        """
        try:
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                raise AudioProcessingError("File does not exist")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise AudioProcessingError(
                    f"File size {file_size} bytes exceeds maximum {self.max_file_size} bytes"
                )

            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                raise AudioProcessingError(
                    f"Unsupported file format: {file_path.suffix}. "
                    f"Supported formats: {', '.join(self.supported_formats)}"
                )

            # Check MIME type
            mime_type = magic.from_file(str(file_path), mime=True)
            if mime_type not in SECURITY_SETTINGS["allowed_mime_types"]:
                raise AudioProcessingError(f"Invalid file type: {mime_type}")

            # Get audio information
            duration, channels, sample_rate = self._get_audio_info(str(file_path))

            # Validate audio properties
            if duration <= 0:
                raise AudioProcessingError("Audio file has zero duration")

            if duration > 86400:  # 24 hours max
                raise AudioProcessingError("Audio file too long (max 24 hours)")

            # Generate file hash
            file_hash = self._generate_file_hash(str(file_path))

            return {
                "file_path": str(file_path),
                "original_filename": file_path.name,
                "file_size": file_size,
                "file_format": file_path.suffix.lower().lstrip("."),
                "duration": duration,
                "channels": channels,
                "sample_rate": sample_rate,
                "mime_type": mime_type,
                "file_hash": file_hash,
                "is_valid": True,
            }

        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            logger.error(f"File validation error: {e}")
            raise AudioProcessingError(f"File validation failed: {str(e)}")

    def convert_to_wav(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert audio file to WAV format for optimal processing.

        Args:
            input_path: Path to input audio file
            output_path: Optional output path. If None, creates temp file

        Returns:
            Path to converted WAV file
        """
        try:
            input_path = Path(input_path)

            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"converted_{input_path.stem}.wav")

            # Load audio using pydub for robust format support
            audio = AudioSegment.from_file(str(input_path))

            # Convert to mono and target sample rate
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(self.sample_rate)

            # Export as WAV
            audio.export(
                output_path, format="wav", parameters=["-ar", str(self.sample_rate)]
            )

            logger.info(f"Converted {input_path} to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise AudioProcessingError(f"Failed to convert audio file: {str(e)}")

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio for transcription.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa for processing
            audio_data, sample_rate = librosa.load(
                audio_path, sr=self.sample_rate, mono=True, dtype=np.float32
            )

            # Normalize audio
            audio_data = self._normalize_audio(audio_data)

            # Remove silence (optional)
            audio_data = self._remove_silence(audio_data, sample_rate)

            return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")

    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract audio features for speaker analysis.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary of audio features
        """
        try:
            audio_data, sample_rate = self.preprocess_audio(audio_path)

            features = {}

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            features["spectral_centroid"] = np.mean(spectral_centroids)

            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )
            features["spectral_rolloff"] = np.mean(spectral_rolloff)

            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate
            )
            features["spectral_bandwidth"] = np.mean(spectral_bandwidth)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features["zero_crossing_rate"] = np.mean(zcr)

            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_std"] = np.std(mfccs, axis=1).tolist()

            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features["avg_pitch"] = np.mean(pitch_values)
                features["pitch_std"] = np.std(pitch_values)
                features["pitch_min"] = np.min(pitch_values)
                features["pitch_max"] = np.max(pitch_values)
            else:
                features["avg_pitch"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_min"] = 0.0
                features["pitch_max"] = 0.0

            # Energy features
            rms = librosa.feature.rms(y=audio_data)
            features["rms_energy"] = np.mean(rms)
            features["rms_std"] = np.std(rms)

            # Tempo
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            features["tempo"] = tempo

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise AudioProcessingError(f"Feature extraction failed: {str(e)}")

    def create_audio_segments(
        self, audio_path: str, segments: List[Dict]
    ) -> Dict[str, str]:
        """
        Create audio segments for speaker preview clips.

        Args:
            audio_path: Path to full audio file
            segments: List of segment dictionaries with start_time, end_time, speaker

        Returns:
            Dictionary mapping speakers to segment file paths
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            speaker_segments = {}
            temp_dir = tempfile.gettempdir()

            for segment in segments:
                speaker = segment.get("speaker", "unknown")
                start_time = (
                    segment.get("start_time", 0) * 1000
                )  # Convert to milliseconds
                end_time = segment.get("end_time", 10) * 1000

                # Limit segment duration for preview
                max_duration = AUDIO_SETTINGS["preview_duration"] * 1000
                if end_time - start_time > max_duration:
                    end_time = start_time + max_duration

                # Extract segment
                segment_audio = audio[start_time:end_time]

                # Save segment
                segment_filename = f"{speaker}_{segment.get('id', 'preview')}.wav"
                segment_path = os.path.join(temp_dir, segment_filename)
                segment_audio.export(segment_path, format="wav")

                speaker_segments[speaker] = segment_path

            logger.info(f"Created {len(speaker_segments)} speaker preview segments")
            return speaker_segments

        except Exception as e:
            logger.error(f"Segment creation failed: {e}")
            raise AudioProcessingError(f"Failed to create audio segments: {str(e)}")

    def chunk_audio(
        self, audio_path: str, chunk_length: int = 30, overlap: int = 5
    ) -> List[str]:
        """
        Split audio into chunks for processing.

        Args:
            audio_path: Path to audio file
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds

        Returns:
            List of chunk file paths
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            chunk_length_ms = chunk_length * 1000
            overlap_ms = overlap * 1000
            step_ms = chunk_length_ms - overlap_ms

            chunks = []
            temp_dir = tempfile.gettempdir()
            audio_stem = Path(audio_path).stem

            for i, start_ms in enumerate(range(0, len(audio), step_ms)):
                end_ms = start_ms + chunk_length_ms
                if end_ms > len(audio):
                    end_ms = len(audio)

                chunk = audio[start_ms:end_ms]
                chunk_path = os.path.join(
                    temp_dir, f"{audio_stem}_chunk_{i + 1:03d}.wav"
                )
                chunk.export(chunk_path, format="wav")
                chunks.append(chunk_path)

            logger.info(f"Created {len(chunks)} audio chunks from {audio_path}")
            return chunks

        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            raise AudioProcessingError(f"Failed to chunk audio: {str(e)}")

    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """
        Clean up temporary files.

        Args:
            file_paths: List of file paths to delete
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")

    def _get_audio_info(self, file_path: str) -> Tuple[float, int, int]:
        """Get basic audio information."""
        try:
            with sf.SoundFile(file_path) as f:
                duration = float(len(f)) / f.samplerate
                channels = f.channels
                sample_rate = f.samplerate
                return duration, channels, sample_rate
        except Exception:
            # Fallback to pydub
            try:
                audio = AudioSegment.from_file(file_path)
                duration = len(audio) / 1000.0  # Convert from milliseconds
                channels = audio.channels
                sample_rate = audio.frame_rate
                return duration, channels, sample_rate
            except Exception as e:
                raise AudioProcessingError(f"Could not read audio file: {str(e)}")

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data."""
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        return audio_data

    def _remove_silence(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove silence from audio data."""
        try:
            # Simple silence removal using librosa
            intervals = librosa.effects.split(
                audio_data, top_db=20, frame_length=2048, hop_length=512
            )

            if len(intervals) == 0:
                return audio_data

            # Concatenate non-silent intervals
            non_silent = []
            for start, end in intervals:
                non_silent.append(audio_data[start:end])

            if non_silent:
                return np.concatenate(non_silent)
            else:
                return audio_data

        except Exception:
            # If silence removal fails, return original audio
            return audio_data

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    @staticmethod
    def get_audio_format_info() -> Dict[str, Dict[str, str]]:
        """Get supported audio format information."""
        return {
            ".mp3": {"name": "MP3", "description": "MPEG Audio Layer 3"},
            ".wav": {"name": "WAV", "description": "Waveform Audio File Format"},
            ".m4a": {"name": "M4A", "description": "MPEG-4 Audio"},
            ".flac": {"name": "FLAC", "description": "Free Lossless Audio Codec"},
            ".ogg": {"name": "OGG", "description": "Ogg Vorbis"},
        }
