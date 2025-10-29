"""
Diarization service using PyAnnote for speaker identification.
Handles speaker diarization, voice embedding extraction, and speaker matching.
"""

import logging
import os
import tempfile
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import librosa

from app.core.config import get_settings, AUDIO_SETTINGS
from app.core.database import get_database
from app.models.transcription import Transcription
from app.models.speaker import Speaker
from app.utils.exceptions import DiarizationError
from app.services.audio_processor import AudioProcessor
from app.services.gpu_optimization import get_gpu_optimizer

logger = logging.getLogger(__name__)


class MockDiarizationPipeline:
    """Mock diarization pipeline for development and testing."""

    def __init__(self):
        self.device = "cpu"

    def __call__(
        self,
        audio_dict,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        return_embeddings=False,
        hook=None,
    ):
        """Mock diarization that returns simple speaker segments."""
        from pyannote.core import Annotation, Segment

        # Get audio duration
        import librosa

        waveform, sample_rate = audio_dict["waveform"], audio_dict["sample_rate"]
        duration = len(waveform[0]) / sample_rate

        # Create mock annotation with 2 speakers
        annotation = Annotation()

        if duration <= 2:
            # Very short audio - single speaker
            annotation[Segment(0, duration)] = "SPEAKER_00"
        else:
            # Split between 2 speakers
            mid_point = duration / 2
            annotation[Segment(0, mid_point)] = "SPEAKER_00"
            annotation[Segment(mid_point, duration)] = "SPEAKER_01"

        return annotation

    def to(self, device):
        """Mock device movement."""
        return self

    def __del__(self):
        """Mock cleanup."""
        pass


class DiarizationService:
    """
    Diarization service using PyAnnote for speaker identification.
    Handles speaker diarization, voice embedding extraction, and speaker matching.
    """

    def __init__(self):
        self.settings = get_settings()
        self.pyannote_model = self.settings.pyannote_model
        self.gpu_optimizer = get_gpu_optimizer()
        self.device = self._get_device()
        self.pipeline = None
        self.audio_processor = AudioProcessor()
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.min_speaker_duration = AUDIO_SETTINGS["min_speaker_duration"]
        self._pipeline_lock = threading.RLock()

    def _get_device(self) -> str:
        """Determine the best device for processing using GPU optimizer."""
        return self.gpu_optimizer.get_optimal_device()

    def _load_pipeline(self) -> None:
        """Load PyAnnote diarization pipeline."""
        if self.pipeline is not None:
            return

        # In development/test mode, use mock pipeline directly
        if self.settings.test_mode or self.settings.mock_gpu:
            logger.info("Using mock diarization pipeline for development/testing")
            self.pipeline = MockDiarizationPipeline()
            return

        try:
            logger.info(
                f"Loading PyAnnote pipeline: {self.pyannote_model} on {self.device}"
            )

            # Load the diarization pipeline (guarded)
            with self._pipeline_lock:
                self.pipeline = Pipeline.from_pretrained(
                    self.pyannote_model,
                    use_auth_token=(
                        self.settings.huggingface_token
                        or os.environ.get("HUGGINGFACE_TOKEN")
                    ),  # Use token if available
                )

            # Move to appropriate device
            if self.device != "cpu":
                try:
                    target_device = torch.device(self.device)
                except Exception:
                    logger.warning(
                        f'Invalid device "{self.device}" for PyAnnote; falling back to CPU'
                    )
                    target_device = torch.device("cpu")
                # Guard device move with a lock to avoid concurrent .to() issues
                with self._pipeline_lock:
                    moved = False
                    try:
                        # CUDA/cuDNN preflight on target device to catch mismatches early
                        if target_device.type == "cuda":
                            try:
                                _x = torch.randn(1, 1, 64, 64, device=target_device)
                                _conv = torch.nn.Conv2d(1, 1, kernel_size=3).to(
                                    target_device
                                )
                                _y = _conv(_x)
                                _ = _y.mean().item()
                                del _x, _conv, _y
                                torch.cuda.synchronize(device=target_device.index or 0)
                                logger.info(
                                    f"CUDA/cuDNN preflight OK on {target_device} "
                                    f"(torch_cuda={getattr(torch.version, 'cuda', None)}, "
                                    f"cudnn_version={getattr(torch.backends.cudnn, 'version', lambda: None)()})"
                                )
                            except Exception as pre_err:
                                if "cudnn" in str(pre_err).lower():
                                    prev_cudnn = torch.backends.cudnn.enabled
                                    cudnn_ver = getattr(
                                        torch.backends.cudnn, "version", lambda: None
                                    )()
                                    logger.warning(
                                        "cuDNN error during preflight; attempting pipeline move with cuDNN disabled "
                                        f"(device={self.device}, torch_cuda={getattr(torch.version, 'cuda', None)}, "
                                        f"cudnn_version={cudnn_ver}, error={pre_err!r})"
                                    )
                                    try:
                                        torch.backends.cudnn.enabled = False
                                        self.pipeline.to(target_device)
                                        logger.info(
                                            "PyAnnote pipeline moved to CUDA with cuDNN disabled"
                                        )
                                        moved = True
                                    except Exception as move_err2:
                                        logger.warning(
                                            "Move with cuDNN disabled failed; falling back to CPU "
                                            f"(error={move_err2!r})"
                                        )
                                        self.pipeline.to(torch.device("cpu"))
                                        self.device = "cpu"
                                        moved = True
                                    finally:
                                        torch.backends.cudnn.enabled = prev_cudnn
                                else:
                                    logger.warning(
                                        f"Non-cuDNN error during preflight: {pre_err!r}"
                                    )
                        if not moved:
                            self.pipeline.to(target_device)
                            moved = True
                    except Exception as move_err:
                        if "cudnn" in str(move_err).lower():
                            cudnn_ver = getattr(
                                torch.backends.cudnn, "version", lambda: None
                            )()
                            logger.warning(
                                "cuDNN error while moving PyAnnote pipeline to device; retrying on CPU "
                                f"(device={self.device}, torch_cuda={getattr(torch.version, 'cuda', None)}, "
                                f"cudnn_version={cudnn_ver}, error={move_err!r})"
                            )
                            self.pipeline.to(torch.device("cpu"))
                            self.device = "cpu"
                        else:
                            raise

            logger.info(f"PyAnnote pipeline loaded successfully on {self.device}")
            # One-time warmup to ensure pipeline works on selected device
            try:
                # Fast warmup with 0.5s of silence at configured sample rate
                test_sr = self.sample_rate
                test_wave = torch.zeros(1, int(0.5 * test_sr), dtype=torch.float32)
                _ = self.pipeline({"waveform": test_wave, "sample_rate": test_sr})
                logger.info(f"PyAnnote warmup succeeded on {self.device}")
            except Exception as warm_err:
                logger.warning(f"PyAnnote warmup failed on {self.device}: {warm_err}")
                # Fallback to CPU as a safety net
                try:
                    self.pipeline.to(torch.device("cpu"))
                    self.device = "cpu"
                    # Try warmup again on CPU
                    test_sr = self.sample_rate
                    test_wave = torch.zeros(1, int(0.5 * test_sr), dtype=torch.float32)
                    _ = self.pipeline({"waveform": test_wave, "sample_rate": test_sr})
                    logger.info("PyAnnote warmup succeeded on CPU after fallback")
                except Exception as cpu_warm_err:
                    logger.error(f"PyAnnote CPU warmup also failed: {cpu_warm_err}")

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")

            # In development/test mode, use mock pipeline
            if self.settings.test_mode or self.settings.mock_gpu:
                logger.warning("Using mock diarization pipeline for development")
                self.pipeline = MockDiarizationPipeline()
                return

            # cuDNN error handling: retry on CPU
            if "cudnn" in str(e).lower():
                try:
                    logger.warning(
                        "cuDNN error detected during PyAnnote load; retrying on CPU"
                    )
                    token = self.settings.huggingface_token or os.environ.get(
                        "HUGGINGFACE_TOKEN"
                    )
                    self.device = "cpu"
                    self.pipeline = Pipeline.from_pretrained(
                        self.pyannote_model,
                        use_auth_token=token,
                    )
                    logger.info(
                        "PyAnnote pipeline loaded successfully on CPU after cuDNN error"
                    )
                    return
                except Exception as cpu_err:
                    logger.error(f"CPU fallback for PyAnnote failed: {cpu_err}")

            # Check if it's an authentication error
            if "gated" in str(e).lower() or "token" in str(e).lower():
                logger.error(
                    "PyAnnote model requires Hugging Face token. Set HUGGINGFACE_TOKEN environment variable."
                )
                logger.info("For now, using mock diarization as fallback")
                self.pipeline = MockDiarizationPipeline()
                return

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
                # Get audio duration to determine processing strategy
                duration, _, _ = self.audio_processor._get_audio_info(wav_path)

                if progress_callback:
                    progress_callback(15, "Starting diarization")

                # Use chunked processing for long files to avoid PyAnnote issues
                if duration > 600:  # 10 minutes
                    diarization_result = self._perform_chunked_diarization(
                        wav_path, transcription, duration, progress_callback
                    )
                else:
                    # Standard processing for shorter files
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

            # Load audio and validate minimum length
            waveform, sample_rate = librosa.load(
                audio_path, sr=self.sample_rate, mono=True
            )
            duration = len(waveform) / sample_rate

            # Ensure minimum duration for reliable diarization
            min_duration = 1.0  # 1 second minimum for testing
            if duration < min_duration:
                logger.warning(
                    f"Audio too short for reliable diarization: {duration:.2f}s < {min_duration}s"
                )
                # Create mock diarization for very short audio
                from pyannote.core import Annotation, Segment

                annotation = Annotation()
                annotation[Segment(0, duration)] = "SPEAKER_00"
                return annotation

            waveform = torch.from_numpy(waveform).float().unsqueeze(0)

            if progress_callback:
                progress_callback(30, "Identifying speakers")

            # Suppress PyAnnote std() warning for small chunks
            import warnings

            # Use warnings context properly - only filter std() warnings, keep important ones
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r".*std\(\): degrees of freedom is <= 0.*"
                )

                # Perform diarization with correct pyannote.audio API parameters
                # Note: min_duration is not supported by pyannote.audio, will filter in post-processing
                diarization = self.pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    num_speakers=None,  # Auto-detect
                    min_speakers=1,
                    max_speakers=self.settings.max_speakers,
                )

            if progress_callback:
                progress_callback(70, "Processing speaker segments")

            return diarization

        except Exception as e:
            logger.error(f"Diarization processing failed: {e}")
            # Fallback to simple speaker assignment for very short audio
            if "duration" in locals() and duration < min_duration:
                logger.info("Using fallback diarization for short audio")
                from pyannote.core import Annotation, Segment

                annotation = Annotation()
                annotation[Segment(0, duration)] = "SPEAKER_00"
                return annotation
            raise DiarizationError(f"Diarization processing failed: {str(e)}")

    def _perform_chunked_diarization(
        self,
        audio_path: str,
        transcription: Transcription,
        total_duration: float,
        progress_callback: Optional[callable] = None,
    ) -> Annotation:
        """
        Perform diarization on long audio files by processing in chunks.
        This helps avoid PyAnnote warnings about insufficient data for statistical calculations.
        """
        from pyannote.core import Annotation, Segment
        import librosa

        try:
            if progress_callback:
                progress_callback(20, "Preparing chunked diarization")

            # Load the full audio
            waveform, sample_rate = librosa.load(
                audio_path, sr=self.sample_rate, mono=True
            )

            # Calculate chunk parameters
            chunk_duration = 300  # 5 minutes per chunk
            overlap_duration = 30  # 30 seconds overlap
            min_chunk_duration = 60  # 1 minute minimum

            # Adjust chunk size for very long files
            if total_duration > 3600:  # > 1 hour
                chunk_duration = 600  # 10 minutes chunks
            elif total_duration < 1800:  # < 30 minutes
                chunk_duration = 180  # 3 minutes chunks

            logger.info(
                f"Processing {total_duration:.1f}s audio in {chunk_duration}s chunks "
                f"with {overlap_duration}s overlap"
            )

            # Process chunks
            full_annotation = Annotation()
            speakers_found = set()
            chunk_count = 0

            start_time = 0
            while start_time < total_duration:
                end_time = min(start_time + chunk_duration, total_duration)
                chunk_count += 1

                # Extract chunk
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                chunk_waveform = waveform[start_sample:end_sample]
                chunk_duration_actual = len(chunk_waveform) / sample_rate

                # Skip very short chunks (but process if this is the final chunk)
                is_last_chunk = end_time >= total_duration - 1e-6
                if (
                    chunk_duration_actual < min_chunk_duration
                    and start_time > 0
                    and not is_last_chunk
                ):
                    logger.warning(
                        f"Skipping short chunk: {chunk_duration_actual:.1f}s"
                    )
                    start_time = end_time
                    continue

                if progress_callback:
                    progress_callback(
                        20
                        + (
                            chunk_count
                            * 30
                            // max(1, int(total_duration / chunk_duration))
                        ),
                        f"Processing chunk {chunk_count} ({start_time:.0f}-{end_time:.0f}s)",
                    )

                try:
                    # Process chunk
                    chunk_tensor = torch.from_numpy(chunk_waveform).float().unsqueeze(0)

                    # Suppress warnings for each chunk
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=r".*std\(\): degrees of freedom is <= 0.*"
                        )

                        chunk_annotation = self.pipeline(
                            {"waveform": chunk_tensor, "sample_rate": sample_rate},
                            num_speakers=None,
                            min_speakers=1,
                            max_speakers=min(self.settings.max_speakers, 4),
                        )

                    # Map chunk annotations to full timeline
                    for segment, track, speaker in chunk_annotation.itertracks(
                        yield_label=True
                    ):
                        global_start = start_time + segment.start
                        global_end = start_time + segment.end

                        # Skip segments that overlap with previous chunks (except first chunk)
                        if start_time > 0 and segment.start < overlap_duration:
                            continue

                        # Use consistent speaker mapping
                        mapped_speaker = f"SPEAKER_{len(speakers_found):02d}"
                        if speaker not in speakers_found:
                            speakers_found.add(speaker)
                            mapped_speaker = f"SPEAKER_{len(speakers_found) - 1:02d}"
                        else:
                            speaker_index = list(speakers_found).index(speaker)
                            mapped_speaker = f"SPEAKER_{speaker_index:02d}"

                        full_annotation[Segment(global_start, global_end)] = (
                            mapped_speaker
                        )

                    logger.info(
                        f"Chunk {chunk_count} processed: {len(speakers_found)} speakers found"
                    )

                except Exception as chunk_error:
                    logger.warning(
                        f"Failed to process chunk {chunk_count}: {chunk_error}"
                    )
                    # Continue with next chunk instead of failing completely
                    pass

                # Move to next chunk with overlap
                start_time = end_time - overlap_duration
                if start_time < 0:
                    start_time = 0

            # If no segments were added, fallback to single-speaker annotation
            if not list(full_annotation.itertracks()):
                logger.warning("No valid segments found, using single speaker fallback")
                full_annotation[Segment(0, total_duration)] = "SPEAKER_00"

            logger.info(
                f"Chunked diarization completed: {len(speakers_found)} speakers "
                f"in {chunk_count} chunks"
            )

            return full_annotation

        except Exception as e:
            logger.error(f"Chunked diarization failed: {e}")
            # Fallback to simple annotation
            annotation = Annotation()
            annotation[Segment(0, total_duration)] = "SPEAKER_00"
            return annotation

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

            import time
            from datetime import datetime

            transcription.updated_at = datetime.utcnow()

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
