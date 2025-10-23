"""
Transcription model for storing audio transcriptions and metadata.
Manages transcription sessions, speaker segments, and processing results.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    JSON,
    ForeignKey,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from ..core.database import Base

logger = logging.getLogger(__name__)


class Transcription(Base):
    """
    Transcription model for storing audio transcription results and metadata.
    Links audio files with their processed text, speaker segments, and analysis.
    """

    __tablename__ = "transcriptions"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_duration = Column(Float, nullable=False)  # Duration in seconds
    file_format = Column(String(50), nullable=False)

    # Processing metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Processing status
    status = Column(
        String(50), default="pending", nullable=False, index=True
    )  # pending, processing, completed, failed
    progress_percentage = Column(Float, default=0.0, nullable=False)
    current_step = Column(
        String(100), nullable=True
    )  # Current processing step description

    # Model and configuration used
    whisper_model = Column(String(100), nullable=False)
    pyannote_model = Column(String(100), nullable=False)
    device_used = Column(String(50), nullable=True)  # cpu, cuda, mps
    processing_time = Column(Float, nullable=True)  # Total processing time in seconds

    # Results
    full_transcript = Column(Text, nullable=True)  # Complete transcribed text
    language_detected = Column(String(10), nullable=True)
    confidence_score = Column(Float, nullable=True)  # Overall confidence score

    # Speaker information
    num_speakers = Column(Integer, default=0, nullable=False)
    speakers_assigned = Column(Boolean, default=False, nullable=False)

    # Segments (JSON array of speaker segments)
    segments = Column(
        JSON, nullable=True
    )  # List of segments with speaker, text, timestamps

    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    # Export and cleanup
    export_formats = Column(JSON, nullable=True)  # List of generated export formats
    cleanup_scheduled = Column(Boolean, default=False, nullable=False)
    cleanup_at = Column(DateTime, nullable=True)

    # Relationships
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)
    speaker = relationship("Speaker", back_populates="transcriptions")

    def __repr__(self) -> str:
        return f"<Transcription(id={self.id}, session={self.session_id}, status={self.status})>"

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if transcription is completed."""
        return self.status == "completed"

    @hybrid_property
    def is_failed(self) -> bool:
        """Check if transcription failed."""
        return self.status == "failed"

    @hybrid_property
    def is_processing(self) -> bool:
        """Check if transcription is currently processing."""
        return self.status == "processing"

    @hybrid_property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None

    @hybrid_property
    def formatted_duration(self) -> str:
        """Get formatted duration string."""
        if not self.file_duration:
            return "00:00:00"

        hours = int(self.file_duration // 3600)
        minutes = int((self.file_duration % 3600) // 60)
        seconds = int(self.file_duration % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @hybrid_property
    def formatted_file_size(self) -> str:
        """Get formatted file size string."""
        if not self.file_size:
            return "0 B"

        for unit in ["B", "KB", "MB", "GB"]:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"

    def update_progress(self, percentage: float, step: Optional[str] = None) -> None:
        """Update processing progress."""
        self.progress_percentage = max(0, min(100, percentage))
        if step:
            self.current_step = step
        self.updated_at = datetime.utcnow()

    def mark_as_started(self) -> None:
        """Mark transcription as started."""
        self.status = "processing"
        self.started_at = datetime.utcnow()
        self.update_progress(0, "Initializing transcription")
        logger.info(f"Transcription {self.session_id} started")

    def mark_as_completed(self) -> None:
        """Mark transcription as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        self.current_step = "Completed"

        if self.started_at:
            self.processing_time = self.processing_duration

        logger.info(
            f"Transcription {self.session_id} completed in {self.processing_time} seconds"
        )

    def mark_as_failed(
        self, error_message: str, error_traceback: Optional[str] = None
    ) -> None:
        """Mark transcription as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if error_traceback:
            self.error_traceback = error_traceback

        if self.started_at:
            self.processing_time = self.processing_duration

        logger.error(f"Transcription {self.session_id} failed: {error_message}")

    def add_segment(
        self,
        speaker_name: str,
        text: str,
        start_time: float,
        end_time: float,
        confidence: float = 0.0,
    ) -> None:
        """Add a speaker segment to the transcription."""
        if self.segments is None:
            self.segments = []

        segment = {
            "id": len(self.segments) + 1,
            "speaker": speaker_name,
            "text": text.strip(),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "confidence": confidence,
            "word_count": len(text.split()),
        }

        self.segments.append(segment)
        self.updated_at = datetime.utcnow()

    def get_segments_by_speaker(self, speaker_name: Optional[str] = None) -> List[Dict]:
        """Get segments filtered by speaker name."""
        if not self.segments:
            return []

        if speaker_name:
            return [seg for seg in self.segments if seg.get("speaker") == speaker_name]
        return self.segments

    def get_speaker_list(self) -> List[str]:
        """Get list of unique speakers in the transcription."""
        if not self.segments:
            return []

        speakers = set()
        for segment in self.segments:
            if "speaker" in segment:
                speakers.add(segment["speaker"])

        return sorted(list(speakers))

    def get_speaker_stats(self) -> Dict[str, Dict]:
        """Get statistics for each speaker."""
        if not self.segments:
            return {}

        stats = {}
        for segment in self.segments:
            speaker = segment.get("speaker", "Unknown")
            if speaker not in stats:
                stats[speaker] = {
                    "segment_count": 0,
                    "total_duration": 0.0,
                    "total_words": 0,
                    "avg_confidence": 0.0,
                    "confidence_sum": 0.0,
                }

            stats[speaker]["segment_count"] += 1
            stats[speaker]["total_duration"] += segment.get("duration", 0)
            stats[speaker]["total_words"] += segment.get("word_count", 0)
            stats[speaker]["confidence_sum"] += segment.get("confidence", 0)

        # Calculate average confidence
        for speaker, data in stats.items():
            if data["segment_count"] > 0:
                data["avg_confidence"] = data["confidence_sum"] / data["segment_count"]
                del data["confidence_sum"]  # Remove intermediate value

        return stats

    def generate_preview_clips(self, duration: float = 10.0) -> List[Dict]:
        """Generate preview clip information for each speaker."""
        if not self.segments:
            return []

        speaker_clips = {}
        for segment in self.segments:
            speaker = segment.get("speaker", "Unknown")
            if speaker not in speaker_clips and segment.get("duration", 0) >= 2.0:
                # Create a preview clip from this segment
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", start_time + duration)

                # Limit clip duration
                if end_time - start_time > duration:
                    end_time = start_time + duration

                speaker_clips[speaker] = {
                    "speaker": speaker,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "preview_text": segment.get("text", "")[:100] + "..."
                    if len(segment.get("text", "")) > 100
                    else segment.get("text", ""),
                }

        return list(speaker_clips.values())

    def update_speaker_assignment(self, old_name: str, new_name: str) -> None:
        """Update speaker name across all segments."""
        if not self.segments:
            return

        for segment in self.segments:
            if segment.get("speaker") == old_name:
                segment["speaker"] = new_name

        self.updated_at = datetime.utcnow()
        logger.info(
            f"Updated speaker name from '{old_name}' to '{new_name}' in transcription {self.session_id}"
        )

    def export_to_dict(self, include_segments: bool = True) -> Dict[str, Any]:
        """Export transcription data to dictionary."""
        result = {
            "id": self.id,
            "session_id": self.session_id,
            "original_filename": self.original_filename,
            "file_duration": self.file_duration,
            "formatted_duration": self.formatted_duration,
            "file_size": self.file_size,
            "formatted_file_size": self.formatted_file_size,
            "file_format": self.file_format,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "processing_time": self.processing_time,
            "status": self.status,
            "full_transcript": self.full_transcript,
            "language_detected": self.language_detected,
            "confidence_score": self.confidence_score,
            "num_speakers": self.num_speakers,
            "speakers": self.get_speaker_list(),
            "speaker_stats": self.get_speaker_stats(),
            "models_used": {
                "whisper": self.whisper_model,
                "pyannote": self.pyannote_model,
                "device": self.device_used,
            },
        }

        if include_segments and self.segments:
            result["segments"] = self.segments

        return result

    def schedule_cleanup(self, delay_hours: int = 1) -> None:
        """Schedule cleanup of transcription files."""
        from datetime import timedelta

        self.cleanup_scheduled = True
        self.cleanup_at = datetime.utcnow() + timedelta(hours=delay_hours)
        self.updated_at = datetime.utcnow()

        logger.info(
            f"Scheduled cleanup for transcription {self.session_id} at {self.cleanup_at}"
        )

    @classmethod
    def get_pending_transcriptions(
        cls, session, limit: int = 10
    ) -> List["Transcription"]:
        """Get pending transcriptions for processing."""
        return (
            session.query(cls)
            .filter(cls.status == "pending")
            .order_by(cls.created_at)
            .limit(limit)
            .all()
        )

    @classmethod
    def get_processing_transcriptions(cls, session) -> List["Transcription"]:
        """Get currently processing transcriptions."""
        return session.query(cls).filter(cls.status == "processing").all()

    @classmethod
    def cleanup_expired(cls, session, hours_old: int = 24) -> int:
        """Clean up old completed transcriptions."""
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)

        expired = (
            session.query(cls)
            .filter(
                cls.status == "completed",
                cls.completed_at < cutoff_time,
                cls.cleanup_scheduled == True,
            )
            .all()
        )

        count = len(expired)
        for transcription in expired:
            session.delete(transcription)

        session.commit()
        logger.info(f"Cleaned up {count} expired transcriptions")

        return count
