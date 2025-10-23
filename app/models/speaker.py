"""
Speaker model for storing speaker traits and identification data.
Manages speaker profiles, voice characteristics, and recognition data.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from ..core.database import Base

logger = logging.getLogger(__name__)


class Speaker(Base):
    """
    Speaker model for storing speaker identification traits and profiles.
    Enables automatic speaker recognition across multiple audio files.
    """

    __tablename__ = "speakers"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Speaker characteristics
    gender = Column(String(50), nullable=True)  # male, female, unknown
    age_range = Column(String(50), nullable=True)  # child, young_adult, adult, senior
    language = Column(String(10), default="en", nullable=False)
    accent = Column(String(100), nullable=True)

    # Voice characteristics (extracted from audio analysis)
    avg_pitch = Column(Float, nullable=True)
    pitch_variance = Column(Float, nullable=True)
    speaking_rate = Column(Float, nullable=True)  # words per minute
    voice_energy = Column(Float, nullable=True)
    spectral_centroid = Column(Float, nullable=True)
    spectral_rolloff = Column(Float, nullable=True)
    zero_crossing_rate = Column(Float, nullable=True)

    # Embedding and recognition data
    voice_embedding = Column(JSON, nullable=True)  # Store voice embedding vector
    mfcc_features = Column(JSON, nullable=True)  # MFCC features for comparison
    confidence_score = Column(Float, default=0.0, nullable=False)
    sample_count = Column(Integer, default=0, nullable=False)  # Number of audio samples

    # Metadata
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(
        Boolean, default=False, nullable=False
    )  # Manually verified by user

    # Relationships
    transcriptions = relationship("Transcription", back_populates="speaker")

    def __repr__(self) -> str:
        return (
            f"<Speaker(id={self.id}, name='{self.name}', verified={self.is_verified})>"
        )

    @hybrid_property
    def display_name(self) -> str:
        """Return the display name for the speaker."""
        return self.name or f"Speaker {self.id}"

    @hybrid_property
    def has_voice_data(self) -> bool:
        """Check if speaker has voice characteristics data."""
        return (
            self.voice_embedding is not None
            or self.avg_pitch is not None
            or self.mfcc_features is not None
        )

    @hybrid_property
    def confidence_level(self) -> str:
        """Return confidence level based on confidence score."""
        if self.confidence_score >= 0.9:
            return "high"
        elif self.confidence_score >= 0.7:
            return "medium"
        elif self.confidence_score >= 0.5:
            return "low"
        else:
            return "very_low"

    def update_voice_characteristics(
        self,
        pitch: Optional[float] = None,
        pitch_variance: Optional[float] = None,
        speaking_rate: Optional[float] = None,
        voice_energy: Optional[float] = None,
        spectral_centroid: Optional[float] = None,
        spectral_rolloff: Optional[float] = None,
        zero_crossing_rate: Optional[float] = None,
    ) -> None:
        """Update voice characteristics with new measurements."""
        if pitch is not None:
            # Update running average
            if self.avg_pitch is None:
                self.avg_pitch = pitch
            else:
                self.avg_pitch = (self.avg_pitch * self.sample_count + pitch) / (
                    self.sample_count + 1
                )

        if pitch_variance is not None:
            self.pitch_variance = pitch_variance

        if speaking_rate is not None:
            if self.speaking_rate is None:
                self.speaking_rate = speaking_rate
            else:
                self.speaking_rate = (
                    self.speaking_rate * self.sample_count + speaking_rate
                ) / (self.sample_count + 1)

        if voice_energy is not None:
            if self.voice_energy is None:
                self.voice_energy = voice_energy
            else:
                self.voice_energy = (
                    self.voice_energy * self.sample_count + voice_energy
                ) / (self.sample_count + 1)

        if spectral_centroid is not None:
            self.spectral_centroid = spectral_centroid

        if spectral_rolloff is not None:
            self.spectral_rolloff = spectral_rolloff

        if zero_crossing_rate is not None:
            self.zero_crossing_rate = zero_crossing_rate

        self.sample_count += 1
        self.updated_at = datetime.utcnow()

    def update_voice_embedding(self, embedding: List[float]) -> None:
        """Update voice embedding with new measurement."""
        if self.voice_embedding is None:
            self.voice_embedding = embedding
        else:
            # Average the embeddings
            current_embedding = self.voice_embedding
            averaged_embedding = [
                (current_embedding[i] * self.sample_count + embedding[i])
                / (self.sample_count + 1)
                for i in range(len(embedding))
            ]
            self.voice_embedding = averaged_embedding

        self.sample_count += 1
        self.updated_at = datetime.utcnow()

    def update_mfcc_features(self, mfcc_features: List[List[float]]) -> None:
        """Update MFCC features with new measurement."""
        if self.mfcc_features is None:
            self.mfcc_features = mfcc_features
        else:
            # Store the most recent MFCC features
            self.mfcc_features = mfcc_features

        self.updated_at = datetime.utcnow()

    def calculate_similarity(self, other_speaker: "Speaker") -> float:
        """
        Calculate similarity score with another speaker.
        Returns a value between 0 and 1, where 1 is perfect match.
        """
        if not self.has_voice_data or not other_speaker.has_voice_data:
            return 0.0

        similarity_score = 0.0
        factors = 0

        # Compare voice embeddings if available
        if self.voice_embedding and other_speaker.voice_embedding:
            embedding_similarity = self._cosine_similarity(
                self.voice_embedding, other_speaker.voice_embedding
            )
            similarity_score += embedding_similarity * 0.5  # Weight: 50%
            factors += 1

        # Compare pitch characteristics
        if self.avg_pitch and other_speaker.avg_pitch:
            pitch_diff = abs(self.avg_pitch - other_speaker.avg_pitch)
            pitch_similarity = max(0, 1 - pitch_diff / 100)  # Normalize to 0-1
            similarity_score += pitch_similarity * 0.2  # Weight: 20%
            factors += 1

        # Compare speaking rate
        if self.speaking_rate and other_speaker.speaking_rate:
            rate_diff = abs(self.speaking_rate - other_speaker.speaking_rate)
            rate_similarity = max(0, 1 - rate_diff / 100)  # Normalize to 0-1
            similarity_score += rate_similarity * 0.15  # Weight: 15%
            factors += 1

        # Compare voice energy
        if self.voice_energy and other_speaker.voice_energy:
            energy_diff = abs(self.voice_energy - other_speaker.voice_energy)
            energy_similarity = max(0, 1 - energy_diff / 50)  # Normalize to 0-1
            similarity_score += energy_similarity * 0.15  # Weight: 15%
            factors += 1

        # Normalize similarity score
        if factors > 0:
            return min(1.0, similarity_score)
        else:
            return 0.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert speaker to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "gender": self.gender,
            "age_range": self.age_range,
            "language": self.language,
            "accent": self.accent,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "is_verified": self.is_verified,
            "is_active": self.is_active,
            "sample_count": self.sample_count,
            "has_voice_data": self.has_voice_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "description": self.description,
        }

    @classmethod
    def find_best_match(
        cls, session, voice_embedding: List[float], min_similarity: float = 0.7
    ) -> Optional["Speaker"]:
        """
        Find the best matching speaker for a given voice embedding.
        Returns the speaker with highest similarity above the threshold.
        """
        speakers = (
            session.query(cls)
            .filter(cls.is_active == True, cls.has_voice_data == True)
            .all()
        )

        best_match = None
        best_similarity = 0.0

        for speaker in speakers:
            if speaker.voice_embedding:
                similarity = speaker._cosine_similarity(
                    voice_embedding, speaker.voice_embedding
                )
                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_match = speaker

        if best_match:
            best_match.confidence_score = best_similarity
            session.commit()

        return best_match

    @classmethod
    def create_from_audio_analysis(
        cls,
        session,
        name: str,
        voice_embedding: Optional[List[float]] = None,
        pitch: Optional[float] = None,
        pitch_variance: Optional[float] = None,
        speaking_rate: Optional[float] = None,
        voice_energy: Optional[float] = None,
        **kwargs,
    ) -> "Speaker":
        """Create a new speaker from audio analysis data."""
        speaker = cls(
            name=name,
            voice_embedding=voice_embedding,
            avg_pitch=pitch,
            pitch_variance=pitch_variance,
            speaking_rate=speaking_rate,
            voice_energy=voice_energy,
            sample_count=1,
            **kwargs,
        )

        session.add(speaker)
        session.commit()
        session.refresh(speaker)

        logger.info(f"Created new speaker: {speaker.name} (ID: {speaker.id})")
        return speaker
