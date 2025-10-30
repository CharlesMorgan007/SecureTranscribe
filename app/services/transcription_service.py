"""
Transcription service using Whisper for speech-to-text processing.
Handles audio transcription with GPU acceleration and chunk-based processing.
"""

import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from faster_whisper import WhisperModel
import librosa

from app.core.config import get_settings, AUDIO_SETTINGS
from app.core.database import get_database
from app.models.transcription import Transcription
from app.utils.exceptions import TranscriptionError
from app.services.audio_processor import AudioProcessor
from app.services.gpu_optimization import get_gpu_optimizer

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Transcription service using Whisper for speech-to-text processing.
    Handles audio transcription with GPU acceleration and chunk-based processing.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_size = self.settings.whisper_model_size
        self.gpu_optimizer = get_gpu_optimizer()
        self.device = self._get_device()
        self.model = None
        self.audio_processor = AudioProcessor()
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.chunk_length = AUDIO_SETTINGS["chunk_length_s"]
        self.overlap_length = AUDIO_SETTINGS["overlap_length_s"]

    def _get_device(self) -> str:
        """Determine the best device for Whisper processing using GPU optimizer."""
        return self.gpu_optimizer.get_optimal_device()

    def _load_model(self) -> None:
        """Load Whisper model with appropriate configuration."""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")

            # Fix tqdm compatibility issues
            import tqdm

            # Ensure tqdm has the required _lock attribute
            if not hasattr(tqdm.tqdm, "_lock"):
                from threading import Lock

                tqdm.tqdm._lock = Lock()
            if not hasattr(tqdm.tqdm, "_instances"):
                tqdm.tqdm._instances = set()

            # Get optimal model loading parameters from GPU optimizer
            model_params = self.gpu_optimizer.optimize_model_loading(
                self.device, f"whisper-{self.model_size}"
            )

            logger.info(f"Using compute_type: {model_params['compute_type']}")
            self.model = WhisperModel(
                self.model_size,
                device=model_params["device"],
                compute_type=model_params["compute_type"],
            )
            self.whisper_model = self.model_size  # Add missing attribute
            self.pyannote_model = (
                "pyannote/speaker-diarization-3.1"  # Add missing attribute
            )

            logger.info(f"Whisper model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")

            # Handle CUDA device recognition errors
            if "unsupported device" in str(e) and self.device.startswith("cuda"):
                logger.warning(
                    "CUDA device not recognized by Whisper, trying fallback to CPU"
                )
                try:
                    # Fallback to CPU if CUDA device not supported
                    self.device = "cpu"
                    logger.info("Retrying Whisper model loading on CPU")

                    # Get CPU-optimized parameters
                    cpu_params = self.gpu_optimizer.optimize_model_loading(
                        "cpu", f"whisper-{self.model_size}"
                    )

                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="float32",
                    )
                    self.whisper_model = self.model_size
                    self.pyannote_model = "pyannote/speaker-diarization-3.1"

                    logger.warning(
                        "Whisper model loaded on CPU due to CUDA compatibility issue"
                    )
                    return
                except Exception as fallback_error:
                    logger.error(f"CPU fallback also failed: {fallback_error}")
                    raise TranscriptionError(
                        f"Failed to load Whisper model on both GPU and CPU: {e}"
                    )

            # Try fallback configuration if it's a tqdm-related error
            if "disabled_tqdm" in str(e) or "_lock" in str(e):
                logger.info(
                    "Attempting fallback model loading with alternative tqdm configuration..."
                )
                try:
                    # Try importing and patching tqdm differently
                    import tqdm.auto as tqdm_auto
                    import sys

                    # Create a simple tqdm patch
                    class SimpleTqdm:
                        def __init__(self, iterable=None, total=None, **kwargs):
                            self.iterable = iterable
                            self.total = total
                            self.n = 0

                        def __iter__(self):
                            if self.iterable:
                                for item in self.iterable:
                                    yield item
                                    self.update(1)
                            else:
                                for i in range(self.total):
                                    yield i
                                    self.update(1)

                        def update(self, n=1):
                            self.n += n

                        def close(self):
                            pass

                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            self.close()

                    # Monkey patch tqdm
                    tqdm_auto.tqdm = SimpleTqdm
                    tqdm.tqdm = SimpleTqdm

                    # Try loading model again with CPU fallback
                    fallback_device = (
                        "cpu" if self.device.startswith("cuda") else self.device
                    )
                    self.model = WhisperModel(
                        self.model_size,
                        device=fallback_device,
                        compute_type="float32",
                    )

                    logger.info(
                        f"Whisper model loaded successfully with fallback on {fallback_device}"
                    )
                    return

                except Exception as fallback_error:
                    logger.error(
                        f"Fallback model loading also failed: {fallback_error}"
                    )

            raise TranscriptionError(f"Failed to load transcription model: {str(e)}")

    def transcribe_audio(
        self,
        audio_path: str,
        transcription: Transcription,
        session,
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            transcription: Transcription model instance
            session: Database session
            language: Optional language code
            progress_callback: Optional progress callback function

        Returns:
            Dictionary containing transcription results
        """
        try:
            # Ensure model is loaded
            self._load_model()

            # Update transcription status
            transcription.mark_as_started()
            transcription.whisper_model = self.model_size
            transcription.device_used = self.device

            if progress_callback:
                progress_callback(10, "Loading audio file")

            # Convert audio to optimal format
            wav_path = self.audio_processor.convert_to_wav(audio_path)

            try:
                # Get audio duration for progress tracking
                duration, _, _ = self.audio_processor._get_audio_info(wav_path)
                transcription.file_duration = duration

                if progress_callback:
                    progress_callback(20, "Starting transcription")

                # Process transcription
                result = self._process_transcription(
                    wav_path, transcription, session, language, progress_callback
                )

                # Update transcription with results
                transcription.full_transcript = result["text"]
                transcription.language_detected = result.get("language", "unknown")
                transcription.confidence_score = result.get("avg_confidence", 0.0)

                # Persist final (merged) segments so exports include the complete transcript
                merged_for_db = []
                for seg in result.get("segments", []):
                    start_v = seg.get("start", seg.get("start_time", 0.0))
                    end_v = seg.get("end", seg.get("end_time", 0.0))
                    text_v = seg.get("text", "")
                    merged_for_db.append(
                        {
                            "id": len(merged_for_db) + 1,
                            "speaker": "Speaker",  # placeholder; diarization updates this later
                            "text": text_v.strip(),
                            "start_time": start_v,
                            "end_time": end_v,
                            "duration": (end_v - start_v)
                            if end_v is not None and start_v is not None
                            else 0.0,
                            "confidence": seg.get("confidence", 0.0),
                            "word_count": len(text_v.split()),
                        }
                    )
                if merged_for_db:
                    transcription.segments = merged_for_db

                transcription.mark_as_completed()

                logger.info(f"Transcription completed for {transcription.session_id}")
                return result

            finally:
                # Cleanup temporary file
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            transcription.mark_as_failed(error_msg)
            raise TranscriptionError(error_msg)

    def _process_transcription(
        self,
        audio_path: str,
        transcription: Transcription,
        session,
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process transcription with chunk-based approach for long files.
        """
        try:
            # Get audio duration
            duration, _, _ = self.audio_processor._get_audio_info(audio_path)

            # For short files (< 5 minutes), process directly
            if duration <= 300:
                return self._transcribe_single_chunk(
                    audio_path, transcription, language, progress_callback
                )

            # For longer files, use chunking
            return self._transcribe_chunks(
                audio_path, transcription, duration, language, progress_callback
            )

        except Exception as e:
            logger.error(f"Transcription processing failed: {e}")
            raise TranscriptionError(f"Transcription processing failed: {str(e)}")

    def _transcribe_single_chunk(
        self,
        audio_path: str,
        transcription: Transcription,
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Transcribe a single audio chunk."""
        try:
            if progress_callback:
                progress_callback(30, "Processing audio")

            # Perform transcription with VAD first; on certain VAD failures or empty output, retry without VAD
            segments = None
            info = None
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                # If VAD filtered everything (no segments), retry without VAD
                try:
                    segments = list(segments)
                except TypeError:
                    # Some versions already return a list-like
                    pass
                if not segments:
                    logger.warning("VAD produced no segments; retrying without VAD")
                    segments, info = self.model.transcribe(
                        audio_path,
                        language=language,
                        beam_size=5,
                        vad_filter=False,
                    )
            except Exception as vad_err:
                # Known failure pattern: max() arg is empty when VAD frames are empty
                if "max()" in str(vad_err) and "empty" in str(vad_err):
                    logger.warning(
                        "Transcribe with VAD failed due to empty iterable; retrying without VAD"
                    )
                    segments, info = self.model.transcribe(
                        audio_path,
                        language=language,
                        beam_size=5,
                        vad_filter=False,
                    )
                else:
                    raise

            if progress_callback:
                progress_callback(70, "Processing results")

            # Process segments
            # Normalize to list and handle empty case
            try:
                segments = list(segments)
            except TypeError:
                pass

            if not segments:
                if progress_callback:
                    progress_callback(90, "Finalizing results")
                result = {
                    "text": "",
                    "segments": [],
                    "language": getattr(info, "language", "unknown")
                    if info
                    else "unknown",
                    "language_probability": getattr(info, "language_probability", 0.0)
                    if info
                    else 0.0,
                    "avg_confidence": 0.0,
                    "duration": getattr(info, "duration", 0.0) if info else 0.0,
                }
                return result

            all_segments = []
            total_confidence = 0.0
            segment_count = 0

            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, "avg_logprob", 0.0),
                }
                all_segments.append(segment_data)

                if hasattr(segment, "avg_logprob"):
                    total_confidence += segment.avg_logprob
                    segment_count += 1

            # Calculate average confidence
            avg_confidence = (
                total_confidence / segment_count if segment_count > 0 else 0.0
            )

            # Combine all text
            full_text = " ".join([seg["text"] for seg in all_segments])

            logger.info(
                f"Single-chunk transcription summary: segments={len(all_segments)}"
            )
            if progress_callback:
                progress_callback(90, "Finalizing results")

            result = {
                "text": full_text,
                "segments": all_segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "avg_confidence": avg_confidence,
                "duration": info.duration,
            }

            # Store segments in transcription
            for i, seg in enumerate(all_segments):
                transcription.add_segment(
                    speaker_name="Speaker",  # Will be updated after diarization
                    text=seg["text"],
                    start_time=seg["start"],
                    end_time=seg["end"],
                    confidence=seg["confidence"],
                )

            return result

        except Exception as e:
            logger.error(f"Single chunk transcription failed: {e}")
            raise TranscriptionError(f"Single chunk transcription failed: {str(e)}")

    def _transcribe_chunks(
        self,
        audio_path: str,
        transcription: Transcription,
        duration: float,
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Transcribe long audio file by chunking."""
        try:
            # Get GPU optimization for large files
            duration_minutes = duration / 60.0
            optimization_params = self.gpu_optimizer.optimize_for_large_file(
                duration_minutes
            )

            # Apply optimization parameters with overrides for long files
            chunk_duration = optimization_params["chunk_size"]
            overlap_duration = self.overlap_length

            # For long files, prefer larger chunks and a slightly larger overlap
            if duration >= 3600:  # >= 60 minutes
                chunk_duration = max(chunk_duration, 90)
                overlap_duration = 10
            elif duration >= 600:  # >= 10 minutes
                chunk_duration = max(chunk_duration, 60)
                overlap_duration = 10

            step_duration = chunk_duration - overlap_duration
            clear_cache_frequency = optimization_params["clear_cache_frequency"]

            num_chunks = int(np.ceil((duration - overlap_duration) / step_duration))

            logger.info(
                f"Processing {duration:.2f}s audio in {num_chunks} chunks "
                f"(chunk_size: {chunk_duration}s, overlap: {overlap_duration}s, step: {step_duration}s, clear_cache every {clear_cache_frequency} chunks)"
            )

            all_segments = []
            total_confidence = 0.0
            total_segments = 0
            detected_language = None
            language_probability = 0.0

            # Process each chunk
            for i in range(num_chunks):
                start_time = i * step_duration
                end_time = min(start_time + chunk_duration, duration)

                if progress_callback:
                    progress = 30 + (i / num_chunks) * 50
                    progress_callback(
                        progress, f"Processing chunk {i + 1}/{num_chunks}"
                    )

                # Extract chunk
                chunk_path = self._extract_audio_chunk(audio_path, start_time, end_time)

                try:
                    # Clear GPU cache periodically to prevent memory issues
                    if i % clear_cache_frequency == 0 and i > 0:
                        self.gpu_optimizer.clear_gpu_cache()
                        logger.debug(f"Cleared GPU cache after processing {i} chunks")

                    # Transcribe chunk
                    chunk_result = self._transcribe_single_chunk(
                        chunk_path, transcription, language, None
                    )

                    # Adjust timestamps
                    logger.debug(
                        f"Chunk {i + 1}/{num_chunks} produced {len(chunk_result.get('segments', []))} segments"
                    )
                    for segment in chunk_result.get("segments", []):
                        segment["start"] += start_time
                        segment["end"] += start_time

                    all_segments.extend(chunk_result["segments"])

                    # Update confidence statistics
                    if chunk_result.get("avg_confidence"):
                        total_confidence += chunk_result["avg_confidence"]
                        total_segments += 1

                    # Store language info from first chunk
                    if detected_language is None:
                        detected_language = chunk_result.get("language", "unknown")
                        language_probability = chunk_result.get(
                            "language_probability", 0.0
                        )

                    # Add segments to transcription
                    for segment in chunk_result["segments"]:
                        transcription.add_segment(
                            speaker_name="Speaker",  # Will be updated after diarization
                            text=segment["text"],
                            start_time=segment["start"],
                            end_time=segment["end"],
                            confidence=segment["confidence"],
                        )

                finally:
                    # Cleanup chunk file
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

            # Merge overlapping segments
            if not all_segments:
                merged_segments = []
            else:
                merged_segments = self._merge_overlapping_segments(
                    all_segments, overlap_duration
                )

            # Combine all text
            full_text = " ".join([seg["text"] for seg in merged_segments])

            # Calculate average confidence
            avg_confidence = (
                total_confidence / total_segments if total_segments > 0 else 0.0
            )

            if progress_callback:
                progress_callback(90, "Finalizing results")

            result = {
                "text": full_text,
                "segments": merged_segments,
                "language": detected_language,
                "language_probability": language_probability,
                "avg_confidence": avg_confidence,
                "duration": duration,
            }

            return result

        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            raise TranscriptionError(f"Chunk transcription failed: {str(e)}")

    def _extract_audio_chunk(
        self, audio_path: str, start_time: float, end_time: float
    ) -> str:
        """Extract a chunk of audio from the full file."""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)

            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            chunk = audio[start_ms:end_ms]

            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            chunk_path = os.path.join(
                temp_dir, f"chunk_{start_time:.1f}_{end_time:.1f}.wav"
            )
            chunk.export(chunk_path, format="wav")

            return chunk_path

        except Exception as e:
            logger.error(f"Failed to extract audio chunk: {e}")
            raise TranscriptionError(f"Failed to extract audio chunk: {str(e)}")

    def _merge_overlapping_segments(
        self, segments: List[Dict], overlap_duration: float
    ) -> List[Dict]:
        """Merge segments that may overlap due to chunk processing."""
        if not segments:
            return []

        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            # Check for overlap
            if next_segment["start"] <= current_segment["end"] + overlap_duration:
                # Merge segments
                current_segment["end"] = max(
                    current_segment["end"], next_segment["end"]
                )
                current_segment["text"] += " " + next_segment["text"]
                current_segment["confidence"] = (
                    current_segment["confidence"] + next_segment["confidence"]
                ) / 2
            else:
                # No overlap, add current and start new
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # Add the last segment
        merged.append(current_segment)

        return merged

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages for transcription."""
        return {
            "auto": "Auto-detect",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "sv": "Swedish",
            "no": "Norwegian",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "tr": "Turkish",
        }

    def estimate_processing_time(self, audio_duration: float) -> float:
        """Estimate processing time based on audio duration and device."""
        # Base processing time ratios (seconds of processing per second of audio)
        if self.device == "cuda":
            ratio = 0.1  # GPU is much faster
        elif self.device == "mps":
            ratio = 0.3  # Apple Silicon is fast
        else:
            ratio = 1.0  # CPU is slower

        # Adjust based on model size
        model_multipliers = {
            "tiny": 0.3,
            "base": 0.5,
            "small": 0.8,
            "medium": 1.2,
            "large-v3": 2.0,
        }

        multiplier = model_multipliers.get(self.model_size, 1.0)

        return audio_duration * ratio * multiplier

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Transcription service cleaned up")
