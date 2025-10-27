"""
Speaker service for managing speaker profiles and matching.
Handles speaker creation, matching, profile updates, and voice characteristics.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from app.core.config import get_settings, AUDIO_SETTINGS
from app.core.database import get_database
from app.models.speaker import Speaker
from app.models.transcription import Transcription
from app.utils.exceptions import SpeakerError
from app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class SpeakerService:
    """
    Speaker service for managing speaker profiles and matching.
    Handles speaker creation, matching, profile updates, and voice characteristics.
    """

    def __init__(self):
        self.settings = get_settings()
        self.audio_processor = AudioProcessor()
        self.min_confidence_threshold = AUDIO_SETTINGS["confidence_threshold"]
        self.min_sample_count = 3  # Minimum samples for reliable matching

    def create_speaker(
        self,
        session,
        name: str,
        audio_path: Optional[str] = None,
        voice_features: Optional[Dict[str, Any]] = None,
        gender: Optional[str] = None,
        age_range: Optional[str] = None,
        language: str = "en",
        accent: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Speaker:
        """
        Create a new speaker profile.

        Args:
            session: Database session
            name: Speaker name
            audio_path: Optional audio file for voice analysis
            voice_features: Optional pre-extracted voice features
            gender: Optional gender information
            age_range: Optional age range
            language: Primary language
            accent: Optional accent information
            description: Optional description

        Returns:
            Created Speaker instance
        """
        try:
            # Check if speaker with this name already exists
            existing_speaker = (
                session.query(Speaker)
                .filter(Speaker.name == name, Speaker.is_active == True)
                .first()
            )

            if existing_speaker:
                raise SpeakerError(f"Speaker with name '{name}' already exists")

            # Extract voice features if audio path provided
            if audio_path and not voice_features:
                voice_features = self.audio_processor.extract_audio_features(audio_path)

            # Create speaker with voice features
            if voice_features:
                speaker = Speaker.create_from_audio_analysis(
                    session=session,
                    name=name,
                    voice_embedding=voice_features.get("mfcc_mean"),
                    pitch=voice_features.get("avg_pitch"),
                    pitch_variance=voice_features.get("pitch_std"),
                    speaking_rate=voice_features.get("tempo", 0)
                    / 60,  # Convert to words/min
                    voice_energy=voice_features.get("rms_energy"),
                    spectral_centroid=voice_features.get("spectral_centroid"),
                    spectral_rolloff=voice_features.get("spectral_rolloff"),
                    zero_crossing_rate=voice_features.get("zero_crossing_rate"),
                    gender=gender,
                    age_range=age_range,
                    language=language,
                    accent=accent,
                    description=description,
                )
            else:
                # Create speaker without voice features
                speaker = Speaker(
                    name=name,
                    gender=gender,
                    age_range=age_range,
                    language=language,
                    accent=accent,
                    description=description,
                )
                session.add(speaker)
                session.commit()
                session.refresh(speaker)

            logger.info(f"Created new speaker: {name} (ID: {speaker.id})")
            return speaker

        except Exception as e:
            logger.error(f"Failed to create speaker: {e}")
            session.rollback()
            raise SpeakerError(f"Failed to create speaker: {str(e)}")

    def update_speaker(
        self,
        session,
        speaker_id: int,
        name: Optional[str] = None,
        gender: Optional[str] = None,
        age_range: Optional[str] = None,
        language: Optional[str] = None,
        accent: Optional[str] = None,
        description: Optional[str] = None,
        is_verified: Optional[bool] = None,
    ) -> Speaker:
        """
        Update speaker profile information.

        Args:
            session: Database session
            speaker_id: Speaker ID
            name: Optional new name
            gender: Optional gender
            age_range: Optional age range
            language: Optional language
            accent: Optional accent
            description: Optional description
            is_verified: Optional verification status

        Returns:
            Updated Speaker instance
        """
        try:
            speaker = session.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                raise SpeakerError(f"Speaker with ID {speaker_id} not found")

            # Update fields if provided
            if name is not None and name != speaker.name:
                # Check for name conflicts
                existing = (
                    session.query(Speaker)
                    .filter(
                        Speaker.name == name,
                        Speaker.id != speaker_id,
                        Speaker.is_active == True,
                    )
                    .first()
                )
                if existing:
                    raise SpeakerError(f"Speaker with name '{name}' already exists")
                speaker.name = name

            if gender is not None:
                speaker.gender = gender
            if age_range is not None:
                speaker.age_range = age_range
            if language is not None:
                speaker.language = language
            if accent is not None:
                speaker.accent = accent
            if description is not None:
                speaker.description = description
            if is_verified is not None:
                speaker.is_verified = is_verified

            session.commit()
            session.refresh(speaker)

            logger.info(f"Updated speaker: {speaker.name} (ID: {speaker.id})")
            return speaker

        except Exception as e:
            logger.error(f"Failed to update speaker: {e}")
            session.rollback()
            raise SpeakerError(f"Failed to update speaker: {str(e)}")

    def add_voice_sample(self, session, speaker_id: int, audio_path: str) -> Speaker:
        """
        Add a new voice sample to improve speaker profile.

        Args:
            session: Database session
            speaker_id: Speaker ID
            audio_path: Path to audio file

        Returns:
            Updated Speaker instance
        """
        try:
            speaker = session.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                raise SpeakerError(f"Speaker with ID {speaker_id} not found")

            # Extract voice features
            voice_features = self.audio_processor.extract_audio_features(audio_path)

            # Update speaker with new voice data
            speaker.update_voice_characteristics(
                pitch=voice_features.get("avg_pitch"),
                pitch_variance=voice_features.get("pitch_std"),
                speaking_rate=voice_features.get("tempo", 0) / 60,
                voice_energy=voice_features.get("rms_energy"),
                spectral_centroid=voice_features.get("spectral_centroid"),
                spectral_rolloff=voice_features.get("spectral_rolloff"),
                zero_crossing_rate=voice_features.get("zero_crossing_rate"),
            )

            # Update MFCC features
            if "mfcc_mean" in voice_features:
                speaker.update_mfcc_features([voice_features["mfcc_mean"]])

            # Update voice embedding
            if "mfcc_mean" in voice_features:
                speaker.update_voice_embedding(voice_features["mfcc_mean"])

            session.commit()
            session.refresh(speaker)

            logger.info(
                f"Added voice sample to speaker: {speaker.name} (ID: {speaker.id})"
            )
            return speaker

        except Exception as e:
            logger.error(f"Failed to add voice sample: {e}")
            session.rollback()
            raise SpeakerError(f"Failed to add voice sample: {str(e)}")

    def find_matching_speakers(
        self,
        session,
        audio_path: str,
        min_similarity: float = 0.7,
        max_results: int = 5,
    ) -> List[Tuple[Speaker, float]]:
        """
        Find speakers that match the provided audio.

        Args:
            session: Database session
            audio_path: Path to audio file
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results to return

        Returns:
            List of (Speaker, similarity_score) tuples
        """
        try:
            # Extract voice features from audio
            voice_features = self.audio_processor.extract_audio_features(audio_path)

            # Get all active speakers with sufficient samples
            speakers = (
                session.query(Speaker)
                .filter(
                    Speaker.is_active == True,
                    Speaker.sample_count >= self.min_sample_count,
                )
                .all()
            )

            # Calculate similarity scores
            matches = []
            for speaker in speakers:
                similarity = self._calculate_speaker_similarity(voice_features, speaker)
                if similarity >= min_similarity:
                    matches.append((speaker, similarity))

            # Sort by similarity score (descending) and limit results
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:max_results]

        except Exception as e:
            logger.error(f"Failed to find matching speakers: {e}")
            raise SpeakerError(f"Failed to find matching speakers: {str(e)}")

    def _calculate_speaker_similarity(
        self, voice_features: Dict[str, Any], speaker: Speaker
    ) -> float:
        """Calculate similarity between voice features and speaker profile."""
        try:
            similarity_score = 0.0
            total_weight = 0.0

            # MFCC similarity (highest weight)
            if (
                "mfcc_mean" in voice_features
                and speaker.voice_embedding
                and len(speaker.voice_embedding) > 0
            ):
                mfcc_similarity = self._cosine_similarity(
                    voice_features["mfcc_mean"], speaker.voice_embedding
                )
                similarity_score += mfcc_similarity * 0.5
                total_weight += 0.5

            # Pitch similarity
            if "avg_pitch" in voice_features and speaker.avg_pitch is not None:
                pitch_diff = abs(voice_features["avg_pitch"] - speaker.avg_pitch)
                pitch_similarity = max(0, 1 - pitch_diff / 100)
                similarity_score += pitch_similarity * 0.2
                total_weight += 0.2

            # Spectral features similarity
            if (
                "spectral_centroid" in voice_features
                and speaker.spectral_centroid is not None
            ):
                centroid_diff = abs(
                    voice_features["spectral_centroid"] - speaker.spectral_centroid
                )
                centroid_similarity = max(0, 1 - centroid_diff / 1000)
                similarity_score += centroid_similarity * 0.15
                total_weight += 0.15

            # Zero crossing rate similarity
            if (
                "zero_crossing_rate" in voice_features
                and speaker.zero_crossing_rate is not None
            ):
                zcr_diff = abs(
                    voice_features["zero_crossing_rate"] - speaker.zero_crossing_rate
                )
                zcr_similarity = max(0, 1 - zcr_diff / 0.1)
                similarity_score += zcr_similarity * 0.15
                total_weight += 0.15

            # Normalize by total weight
            if total_weight > 0:
                return min(1.0, similarity_score / total_weight)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Speaker similarity calculation failed: {e}")
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

    def get_speaker_statistics(self, session, speaker_id: int) -> Dict[str, Any]:
        """Get detailed statistics for a speaker."""
        try:
            speaker = session.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                raise SpeakerError(f"Speaker with ID {speaker_id} not found")

            # Get transcription statistics
            transcriptions = session.query(Transcription).filter(
                Transcription.speaker_id == speaker_id
            )

            total_transcriptions = transcriptions.count()
            total_audio_duration = sum(
                [t.file_duration for t in transcriptions if t.file_duration]
            )
            avg_confidence = (
                sum(
                    [
                        t.confidence_score
                        for t in transcriptions
                        if t.confidence_score is not None
                    ]
                )
                / total_transcriptions
                if total_transcriptions > 0
                else 0.0
            )

            # Calculate confidence distribution
            confidence_levels = {"high": 0, "medium": 0, "low": 0, "very_low": 0}
            for t in transcriptions:
                if t.confidence_score is not None:
                    if t.confidence_score >= 0.9:
                        confidence_levels["high"] += 1
                    elif t.confidence_score >= 0.7:
                        confidence_levels["medium"] += 1
                    elif t.confidence_score >= 0.5:
                        confidence_levels["low"] += 1
                    else:
                        confidence_levels["very_low"] += 1

            return {
                "speaker_info": speaker.to_dict(),
                "total_transcriptions": total_transcriptions,
                "total_audio_duration": total_audio_duration,
                "average_confidence": avg_confidence,
                "confidence_distribution": confidence_levels,
                "voice_data_quality": {
                    "has_voice_embedding": speaker.voice_embedding is not None,
                    "has_mfcc_features": speaker.mfcc_features is not None,
                    "sample_count": speaker.sample_count,
                    "is_reliable": speaker.sample_count >= self.min_sample_count,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get speaker statistics: {e}")
            raise SpeakerError(f"Failed to get speaker statistics: {str(e)}")

    def merge_speakers(
        self, session, source_speaker_id: int, target_speaker_id: int
    ) -> Speaker:
        """
        Merge two speaker profiles, keeping the target as primary.

        Args:
            session: Database session
            source_speaker_id: ID of speaker to merge from
            target_speaker_id: ID of speaker to merge into

        Returns:
            Updated target Speaker instance
        """
        try:
            source_speaker = (
                session.query(Speaker).filter(Speaker.id == source_speaker_id).first()
            )
            target_speaker = (
                session.query(Speaker).filter(Speaker.id == target_speaker_id).first()
            )

            if not source_speaker:
                raise SpeakerError(
                    f"Source speaker with ID {source_speaker_id} not found"
                )
            if not target_speaker:
                raise SpeakerError(
                    f"Target speaker with ID {target_speaker_id} not found"
                )

            # Merge voice characteristics using weighted average
            total_samples = source_speaker.sample_count + target_speaker.sample_count

            if total_samples > 0:
                source_weight = source_speaker.sample_count / total_samples
                target_weight = target_speaker.sample_count / total_samples

                # Merge pitch
                if (
                    source_speaker.avg_pitch is not None
                    and target_speaker.avg_pitch is not None
                ):
                    target_speaker.avg_pitch = (
                        source_speaker.avg_pitch * source_weight
                        + target_speaker.avg_pitch * target_weight
                    )

                # Merge speaking rate
                if (
                    source_speaker.speaking_rate is not None
                    and target_speaker.speaking_rate is not None
                ):
                    target_speaker.speaking_rate = (
                        source_speaker.speaking_rate * source_weight
                        + target_speaker.speaking_rate * target_weight
                    )

                # Merge voice energy
                if (
                    source_speaker.voice_energy is not None
                    and target_speaker.voice_energy is not None
                ):
                    target_speaker.voice_energy = (
                        source_speaker.voice_energy * source_weight
                        + target_speaker.voice_energy * target_weight
                    )

                # Update sample count
                target_speaker.sample_count = total_samples

            # Merge transcriptions
            transcriptions = (
                session.query(Transcription)
                .filter(Transcription.speaker_id == source_speaker_id)
                .all()
            )
            for transcription in transcriptions:
                transcription.speaker_id = target_speaker_id

            # Deactivate source speaker
            source_speaker.is_active = False
            source_speaker.name = (
                f"{source_speaker.name} (merged into {target_speaker.name})"
            )

            session.commit()
            session.refresh(target_speaker)

            logger.info(
                f"Merged speaker {source_speaker.name} into {target_speaker.name}"
            )
            return target_speaker

        except Exception as e:
            logger.error(f"Failed to merge speakers: {e}")
            session.rollback()
            raise SpeakerError(f"Failed to merge speakers: {str(e)}")

    def delete_speaker(self, session, speaker_id: int, permanent: bool = False) -> bool:
        """
        Delete or deactivate a speaker.

        Args:
            session: Database session
            speaker_id: Speaker ID
            permanent: If True, permanently delete; if False, deactivate

        Returns:
            True if successful
        """
        try:
            speaker = session.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                raise SpeakerError(f"Speaker with ID {speaker_id} not found")

            if permanent:
                # Check for associated transcriptions
                transcription_count = (
                    session.query(Transcription)
                    .filter(Transcription.speaker_id == speaker_id)
                    .count()
                )
                if transcription_count > 0:
                    raise SpeakerError(
                        f"Cannot delete speaker with {transcription_count} transcriptions. "
                        "Deactivate instead."
                    )
                session.delete(speaker)
                logger.info(f"Permanently deleted speaker: {speaker.name}")
            else:
                speaker.is_active = False
                logger.info(f"Deactivated speaker: {speaker.name}")

            session.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to delete speaker: {e}")
            session.rollback()
            raise SpeakerError(f"Failed to delete speaker: {str(e)}")

    def get_all_speakers(
        self,
        session,
        active_only: bool = True,
        verified_only: bool = False,
        page: int = 1,
        per_page: int = 50,
    ) -> List[Speaker]:
        """Get list of speakers with optional filtering."""
        try:
            query = session.query(Speaker)

            if active_only:
                query = query.filter(Speaker.is_active == True)
            if verified_only:
                query = query.filter(Speaker.is_verified == True)

            offset = (page - 1) * per_page
            speakers = query.offset(offset).limit(per_page).all()

            return speakers

        except Exception as e:
            logger.error(f"Failed to get speakers: {e}")
            raise SpeakerError(f"Failed to get speakers: {str(e)}")

    def search_speakers(
        self, session, query: str, active_only: bool = True
    ) -> List[Speaker]:
        """Search speakers by name or description."""
        try:
            db_query = session.query(Speaker)

            if active_only:
                db_query = db_query.filter(Speaker.is_active == True)

            # Search in name and description
            db_query = db_query.filter(
                (Speaker.name.ilike(f"%{query}%"))
                | (Speaker.description.ilike(f"%{query}%"))
            )

            return db_query.all()

        except Exception as e:
            logger.error(f"Failed to search speakers: {e}")
            raise SpeakerError(f"Failed to search speakers: {str(e)}")
