"""
Diarization service using PyAnnote for speaker identification.
Handles speaker diarization, voice embedding extraction, and speaker matching.
"""

import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import librosa

from ..core.config import get_settings, AUDIO_SETTINGS
from ..core.database import get_database
from ..models.transcription import Transcription
from ..models.speaker import Speaker
from ..utils.exceptions import DiarizationError
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class DiarizationService:
    """
    Diarization service using PyAnnote for speaker identification.
    Handles speaker diarization, voice embedding extraction, and speaker matching.
    """

    def __init__(self):
        self.settings = get_settings()
        self.pyannote_model = self.settings.pyannote_model
        self.device = self._get_device()
        self.pipeline = None
        self.audio_processor = AudioProcessor()
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.min_speaker_duration = AUDIO_SETTINGS["min_speaker_duration"]

    def _get_device(self) -> str:
        """Determine the best device for diarization processing."""
        if self.settings.use_gpu and torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_pipeline(self) -> None:
        """Load PyAnnote diarization pipeline."""
        if self.pipeline is not None:
            return

        try:
            logger.info(
                f"Loading PyAnnote pipeline: {self.pyannote_model} on {self.device}"
            )

            # Load the diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.pyannote_model,
                use_auth_token=False,  # Set to True with Hugging Face token if needed
            )

            # Move to appropriate device
            if self.device != "cpu":
                self.pipeline.to(self.device)

            logger.info(f"PyAnnote pipeline loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise DiarizationError(f"Failed to load diarization model: {str(e)}")

    def diarize_audio(
        self,
        audio_path: str,
        transcription: Transcription,
        session,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file
            transcription: Transcription model instance
            session: Database session
            progress_callback: Optional progress callback function

        Returns:
            Dictionary containing diarization results
        """
        try:
            # Ensure pipeline is loaded
            self._load_pipeline()

            if progress_callback:
                progress_callback(5, "Loading diarization model")

            # Convert audio to optimal format
            wav_path = self.audio_processor.convert_to_wav(audio_path)

            try:
                if progress_callback:
                    progress_callback(15, "Starting diarization")

                # Perform diarization
                diarization_result = self._perform_diarization(
                    wav_path, transcription, progress_callback
                )

                # Process diarization results
                processed_result = self._process_diarization_result(
                    diarization_result, transcription, session, progress_callback
                )

                # Update transcription with diarization info
                transcription.num_speakers = len(processed_result["speakers"])
                transcription.pyannote_model = self.pyannote_model

                if progress_callback:
                    progress_callback(100, "Diarization completed")

                logger.info(f"Diarization completed for {transcription.session_id}")
                return processed_result

            finally:
                # Cleanup temporary file
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        except Exception as e:
            error_msg = f"Diarization failed: {str(e)}"
            logger.error(error_msg)
            raise DiarizationError(error_msg)

    def _perform_diarization(
        self,
        audio_path: str,
        transcription: Transcription,
        progress_callback: Optional[callable] = None,
    ) -> Annotation:
        """Perform the actual diarization using PyAnnote."""
        try:
            if progress_callback:
                progress_callback(20, "Analyzing audio segments")

            # Load audio
            waveform, sample_rate = librosa.load(
                audio_path, sr=self.sample_rate, mono=True
            )
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)

            if progress_callback:
                progress_callback(30, "Identifying speakers")

            # Perform diarization
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                num_speakers=None,  # Auto-detect
                min_duration_on=self.min_speaker_duration,
                min_duration_off=0.5,
            )

            if progress_callback:
                progress_callback(70, "Processing speaker segments")

            return diarization

        except Exception as e:
            logger.error(f"Diarization processing failed: {e}")
            raise DiarizationError(f"Diarization processing failed: {str(e)}")

    def _process_diarization_result(
        self,
        diarization: Annotation,
        transcription: Transcription,
        session,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Process diarization results and match with existing speakers."""
        try:
            # Extract speaker segments
            speaker_segments = []
            speakers = set()
            speaker_durations = {}

            for segment, track, speaker in diarization.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                duration = end_time - start_time

                # Skip very short segments
                if duration < self.min_speaker_duration:
                    continue

                segment_data = {
                    "speaker": f"Speaker_{speaker}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                }
                speaker_segments.append(segment_data)
                speakers.add(f"Speaker_{speaker}")

                # Track speaker durations
                if f"Speaker_{speaker}" not in speaker_durations:
                    speaker_durations[f"Speaker_{speaker}"] = 0
                speaker_durations[f"Speaker_{speaker}"] += duration

            if progress_callback:
                progress_callback(80, "Matching speakers")

            # Match with existing speakers
            speaker_matches = self._match_speakers(
                transcription.file_path, speaker_segments, session
            )

            # Update transcription segments with speaker information
            self._update_transcription_speakers(
                transcription, speaker_segments, speaker_matches
            )

            # Create speaker preview clips
            preview_clips = self._create_preview_clips(
                transcription.file_path, speaker_segments
            )

            if progress_callback:
                progress_callback(90, "Finalizing speaker data")

            result = {
                "speakers": list(speakers),
                "speaker_segments": speaker_segments,
                "speaker_matches": speaker_matches,
                "speaker_durations": speaker_durations,
                "preview_clips": preview_clips,
                "total_speakers": len(speakers),
            }

            return result

        except Exception as e:
            logger.error(f"Failed to process diarization result: {e}")
            raise DiarizationError(f"Failed to process diarization result: {str(e)}")

    def _match_speakers(
        self,
        audio_path: str,
        speaker_segments: List[Dict],
        session,
    ) -> Dict[str, Optional[Speaker]]:
        """Match identified speakers with existing speakers in database."""
        try:
            speaker_matches = {}

            for segment in speaker_segments:
                speaker_label = segment["speaker"]
                start_time = segment["start_time"]
                end_time = segment["end_time"]

                # Extract audio segment for this speaker
                segment_path = self._extract_speaker_segment(
                    audio_path, start_time, end_time
                )

                try:
                    # Extract voice features
                    voice_features = self.audio_processor.extract_audio_features(
                        segment_path
                    )

                    # Try to find matching speaker
                    matched_speaker = self._find_matching_speaker(
                        voice_features, session
                    )

                    speaker_matches[speaker_label] = matched_speaker

                except Exception as e:
                    logger.warning(f"Failed to match speaker {speaker_label}: {e}")
                    speaker_matches[speaker_label] = None

                finally:
                    # Cleanup segment file
                    if os.path.exists(segment_path):
                        os.remove(segment_path)

            return speaker_matches

        except Exception as e:
            logger.error(f"Speaker matching failed: {e}")
            # Return empty matches on failure
            return {}

    def _find_matching_speaker(
        self, voice_features: Dict[str, Any], session
    ) -> Optional[Speaker]:
        """Find the best matching speaker based on voice features."""
        try:
            # Get all active speakers with voice data
            speakers = (
                session.query(Speaker)
                .filter(Speaker.is_active == True, Speaker.has_voice_data == True)
                .all()
            )

            if not speakers:
                return None

            # Calculate similarity scores
            best_match = None
            best_score = 0.0

            for speaker in speakers:
                similarity = self._calculate_voice_similarity(voice_features, speaker)
                if similarity > best_score and similarity > 0.7:  # Threshold
                    best_score = similarity
                    best_match = speaker

            if best_match:
                logger.info(
                    f"Matched speaker: {best_match.name} (score: {best_score:.3f})"
                )

            return best_match

        except Exception as e:
            logger.error(f"Speaker matching failed: {e}")
            return None

    def _calculate_voice_similarity(
        self, voice_features: Dict[str, Any], speaker: Speaker
    ) -> float:
        """Calculate similarity between voice features and speaker profile."""
        try:
            similarity_score = 0.0
            factors = 0

            # Compare MFCC features
            if (
                "mfcc_mean" in voice_features
                and speaker.mfcc_features
                and len(speaker.mfcc_features) > 0
            ):
                # Simple cosine similarity on MFCC means
                mfcc_similarity = self._cosine_similarity(
                    voice_features["mfcc_mean"], speaker.mfcc_features[0]
                )
                similarity_score += mfcc_similarity * 0.4  # Weight: 40%
                factors += 1

            # Compare pitch
            if "avg_pitch" in voice_features and speaker.avg_pitch is not None:
                pitch_diff = abs(voice_features["avg_pitch"] - speaker.avg_pitch)
                pitch_similarity = max(0, 1 - pitch_diff / 100)  # Normalize
                similarity_score += pitch_similarity * 0.3  # Weight: 30%
                factors += 1

            # Compare spectral centroid
            if (
                "spectral_centroid" in voice_features
                and speaker.spectral_centroid is not None
            ):
                centroid_diff = abs(
                    voice_features["spectral_centroid"] - speaker.spectral_centroid
                )
                centroid_similarity = max(0, 1 - centroid_diff / 1000)  # Normalize
                similarity_score += centroid_similarity * 0.15  # Weight: 15%
                factors += 1

            # Compare zero crossing rate
            if (
                "zero_crossing_rate" in voice_features
                and speaker.zero_crossing_rate is not None
            ):
                zcr_diff = abs(
                    voice_features["zero_crossing_rate"] - speaker.zero_crossing_rate
                )
                zcr_similarity = max(0, 1 - zcr_diff / 0.1)  # Normalize
                similarity_score += zcr_similarity * 0.15  # Weight: 15%
                factors += 1

            # Normalize similarity score
            if factors > 0:
                return min(1.0, similarity_score)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Voice similarity calculation failed: {e}")
            return 0.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            dot_product = np.dot(vec1, vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0

    def _extract_speaker_segment(
        self, audio_path: str, start_time: float, end_time: float
    ) -> str:
        """Extract audio segment for a specific speaker."""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)

            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            segment = audio[start_ms:end_ms]

            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            segment_path = os.path.join(
                temp_dir, f"speaker_{start_time:.1f}_{end_time:.1f}.wav"
            )
            segment.export(segment_path, format="wav")

            return segment_path

        except Exception as e:
            logger.error(f"Failed to extract speaker segment: {e}")
            raise DiarizationError(f"Failed to extract speaker segment: {str(e)}")

    def _update_transcription_speakers(
        self,
        transcription: Transcription,
        speaker_segments: List[Dict],
        speaker_matches: Dict[str, Optional[Speaker]],
    ) -> None:
        """Update transcription segments with speaker information."""
        try:
            if not transcription.segments:
                return

            # Create speaker mapping
            speaker_mapping = {}
            for segment in speaker_segments:
                diarization_label = segment["speaker"]
                matched_speaker = speaker_matches.get(diarization_label)

                if matched_speaker:
                    speaker_mapping[diarization_label] = matched_speaker.name
                else:
                    speaker_mapping[diarization_label] = diarization_label

            # Update each transcription segment
            for trans_segment in transcription.segments:
                segment_start = trans_segment.get("start_time", 0)
                segment_end = trans_segment.get("end_time", segment_start)

                # Find the best matching diarization segment
                best_speaker = "Speaker"
                best_overlap = 0

                for dia_segment in speaker_segments:
                    dia_start = dia_segment["start_time"]
                    dia_end = dia_segment["end_time"]

                    # Calculate overlap
                    overlap_start = max(segment_start, dia_start)
                    overlap_end = min(segment_end, dia_end)
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = speaker_mapping[dia_segment["speaker"]]

                # Update speaker name
                trans_segment["speaker"] = best_speaker

            transcription.updated_at = time.time()

        except Exception as e:
            logger.error(f"Failed to update transcription speakers: {e}")

    def _create_preview_clips(
        self, audio_path: str, speaker_segments: List[Dict]
    ) -> List[Dict]:
        """Create preview clips for each speaker."""
        try:
            return self.audio_processor.create_audio_segments(
                audio_path, speaker_segments
            )
        except Exception as e:
            logger.error(f"Failed to create preview clips: {e}")
            return []

    def update_speaker_profile(
        self,
        speaker: Speaker,
        audio_path: str,
        session,
    ) -> None:
        """Update speaker profile with new audio data."""
        try:
            # Extract audio features
            voice_features = self.audio_processor.extract_audio_features(audio_path)

            # Update speaker characteristics
            if "avg_pitch" in voice_features:
                speaker.update_voice_characteristics(
                    pitch=voice_features["avg_pitch"],
                    pitch_variance=voice_features.get("pitch_std", 0),
                    spectral_centroid=voice_features.get("spectral_centroid"),
                    zero_crossing_rate=voice_features.get("zero_crossing_rate"),
                )

            # Update MFCC features
            if "mfcc_mean" in voice_features:
                speaker.update_mfcc_features([voice_features["mfcc_mean"]])

            # Update voice embedding (simplified version)
            if "mfcc_mean" in voice_features:
                speaker.update_voice_embedding(voice_features["mfcc_mean"])

            session.commit()
            logger.info(f"Updated speaker profile: {speaker.name}")

        except Exception as e:
            logger.error(f"Failed to update speaker profile: {e}")
            session.rollback()

    def estimate_processing_time(self, audio_duration: float) -> float:
        """Estimate diarization processing time."""
        # Diarization is typically slower than transcription
        if self.device == "cuda":
            ratio = 0.3  # GPU
        elif self.device == "mps":
            ratio = 0.8  # Apple Silicon
        else:
            ratio = 2.0  # CPU

        return audio_duration * ratio

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Diarization service cleaned up")
