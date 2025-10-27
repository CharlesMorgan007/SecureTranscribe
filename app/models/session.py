"""
User session model for managing user sessions and tracking.
Handles session management, user identification, and session cleanup.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.hybrid import hybrid_property

from app.core.database import Base
from app.core.config import get_settings, SECURITY_SETTINGS

logger = logging.getLogger(__name__)


class UserSession(Base):
    """
    User session model for managing user sessions and tracking.
    Provides session-based authentication without full user management.
    """

    __tablename__ = "user_sessions"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)

    # Session information
    user_identifier = Column(String(255), nullable=True, index=True)  # Optional user ID
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    # Session status
    is_active = Column(Boolean, default=True, nullable=False)
    is_authenticated = Column(Boolean, default=False, nullable=False)

    # Processing queue information
    queue_position = Column(Integer, default=0, nullable=False)
    current_transcription_id = Column(Integer, nullable=True)
    total_files_processed = Column(Integer, default=0, nullable=False)

    # Session data (JSON)
    session_data = Column(JSON, nullable=True)  # Store session-specific data
    preferences = Column(JSON, nullable=True)  # User preferences and settings

    # Statistics
    total_processing_time = Column(Float, default=0.0, nullable=False)  # seconds
    total_audio_duration = Column(Float, default=0.0, nullable=False)  # seconds
    average_confidence = Column(Float, default=0.0, nullable=False)

    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, session_id='{self.session_id}', active={self.is_active})>"

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at

    @hybrid_property
    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)."""
        return self.is_active and not self.is_expired

    @hybrid_property
    def session_age(self) -> float:
        """Get session age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @session_age.expression
    def session_age(cls):
        """SQL expression for session_age."""
        from sqlalchemy import extract

        return extract("epoch", datetime.utcnow() - cls.created_at)

    @hybrid_property
    def time_until_expiry(self) -> float:
        """Get time until expiry in seconds."""
        if self.is_expired:
            return 0.0
        return (self.expires_at - datetime.utcnow()).total_seconds()

    @time_until_expiry.expression
    def time_until_expiry(cls):
        """SQL expression for time_until_expiry."""
        from sqlalchemy import extract

        return extract("epoch", cls.expires_at - datetime.utcnow())

    @hybrid_property
    def formatted_session_age(self) -> str:
        """Get formatted session age string."""
        age_seconds = int(self.session_age)
        hours = age_seconds // 3600
        minutes = (age_seconds % 3600) // 60
        seconds = age_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @hybrid_property
    def is_processing(self) -> bool:
        """Check if session is currently processing a transcription."""
        return self.current_transcription_id is not None

    @hybrid_property
    def processing_efficiency(self) -> float:
        """Calculate processing efficiency (audio_duration / processing_time)."""
        if self.total_processing_time > 0:
            return self.total_audio_duration / self.total_processing_time
        return 0.0

    def update_last_accessed(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()

    def extend_session(self, hours: int = None) -> None:
        """Extend session expiry."""
        if hours is None:
            hours = SECURITY_SETTINGS["session_timeout"] // 3600

        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_last_accessed()

        logger.info(f"Extended session {self.session_id} until {self.expires_at}")

    def invalidate(self) -> None:
        """Invalidate the session."""
        self.is_active = False
        self.update_last_accessed()
        logger.info(f"Invalidated session {self.session_id}")

    def authenticate(self, user_identifier: Optional[str] = None) -> None:
        """Mark session as authenticated."""
        self.is_authenticated = True
        if user_identifier:
            self.user_identifier = user_identifier
        self.update_last_accessed()

    def update_processing_stats(
        self, processing_time: float, audio_duration: float, confidence: float
    ) -> None:
        """Update processing statistics."""
        self.total_processing_time += processing_time
        self.total_audio_duration += audio_duration
        self.total_files_processed += 1

        # Update average confidence
        if self.total_files_processed > 0:
            total_confidence = (
                self.average_confidence * (self.total_files_processed - 1) + confidence
            )
            self.average_confidence = total_confidence / self.total_files_processed

        self.update_last_accessed()

    def set_queue_position(self, position: int) -> None:
        """Set queue position for this session."""
        self.queue_position = max(0, position)
        self.update_last_accessed()

    def set_current_transcription(self, transcription_id: Optional[int]) -> None:
        """Set currently processing transcription ID."""
        self.current_transcription_id = transcription_id
        self.update_last_accessed()

    def increment_queue_position(self) -> None:
        """Increment queue position (move up in queue)."""
        if self.queue_position > 0:
            self.queue_position -= 1
            self.update_last_accessed()

    def get_session_data(self, key: str, default: Any = None) -> Any:
        """Get session data value."""
        if not self.session_data:
            return default
        return self.session_data.get(key, default)

    def set_session_data(self, key: str, value: Any) -> None:
        """Set session data value."""
        if not self.session_data:
            self.session_data = {}
        self.session_data[key] = value
        self.update_last_accessed()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference value."""
        if not self.preferences:
            return default
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference value."""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value
        self.update_last_accessed()

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert session to dictionary representation."""
        result = {
            "id": self.id,
            "session_id": self.session_id,
            "user_identifier": self.user_identifier,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat()
            if self.last_accessed
            else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "is_authenticated": self.is_authenticated,
            "is_valid": self.is_valid,
            "queue_position": self.queue_position,
            "is_processing": self.is_processing,
            "total_files_processed": self.total_files_processed,
            "session_age": self.session_age,
            "formatted_session_age": self.formatted_session_age,
            "processing_efficiency": self.processing_efficiency,
            "average_confidence": self.average_confidence,
        }

        if include_sensitive:
            result.update(
                {
                    "session_token": self.session_token,
                    "user_agent": self.user_agent,
                    "ip_address": self.ip_address,
                    "session_data": self.session_data,
                    "preferences": self.preferences,
                    "total_processing_time": self.total_processing_time,
                    "total_audio_duration": self.total_audio_duration,
                }
            )

        return result

    @classmethod
    def create_session(
        cls,
        session,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_identifier: Optional[str] = None,
        expires_hours: int = None,
    ) -> "UserSession":
        """Create a new user session."""
        if expires_hours is None:
            expires_hours = SECURITY_SETTINGS["session_timeout"] // 3600

        session_id = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

        user_session = cls(
            session_id=session_id,
            session_token=session_token,
            user_agent=user_agent,
            ip_address=ip_address,
            user_identifier=user_identifier,
            expires_at=expires_at,
        )

        session.add(user_session)
        session.commit()
        session.refresh(user_session)

        logger.info(
            f"Created new session: {session_id} for user: {user_identifier or 'anonymous'}"
        )
        return user_session

    @classmethod
    def get_by_token(cls, session, session_token: str) -> Optional["UserSession"]:
        """Get session by token."""
        return (
            session.query(cls)
            .filter(cls.session_token == session_token, cls.is_active == True)
            .first()
        )

    @classmethod
    def get_by_session_id(cls, session, session_id: str) -> Optional["UserSession"]:
        """Get session by session ID."""
        return (
            session.query(cls)
            .filter(cls.session_id == session_id, cls.is_active == True)
            .first()
        )

    @classmethod
    def get_active_sessions(cls, session) -> List["UserSession"]:
        """Get all active sessions."""
        return (
            session.query(cls)
            .filter(cls.is_active == True, cls.expires_at > datetime.utcnow())
            .all()
        )

    @classmethod
    def get_processing_sessions(cls, session) -> List["UserSession"]:
        """Get sessions currently processing transcriptions."""
        return (
            session.query(cls)
            .filter(
                cls.is_active == True,
                cls.current_transcription_id.isnot(None),
            )
            .all()
        )

    @classmethod
    def get_queue_sessions(cls, session) -> List["UserSession"]:
        """Get sessions in the processing queue."""
        return (
            session.query(cls)
            .filter(
                cls.is_active == True,
                cls.queue_position > 0,
                cls.current_transcription_id.is_(None),
            )
            .order_by(cls.queue_position)
            .all()
        )

    @classmethod
    def cleanup_expired_sessions(cls, session) -> int:
        """Clean up expired sessions."""
        expired_sessions = (
            session.query(cls)
            .filter(
                cls.expires_at < datetime.utcnow(),
            )
            .all()
        )

        count = len(expired_sessions)
        for user_session in expired_sessions:
            user_session.is_active = False

        session.commit()
        logger.info(f"Cleaned up {count} expired sessions")
        return count

    @classmethod
    def cleanup_old_sessions(cls, session, days_old: int = 7) -> int:
        """Clean up old inactive sessions."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        old_sessions = (
            session.query(cls)
            .filter(
                cls.is_active == False,
                cls.last_accessed < cutoff_date,
            )
            .all()
        )

        count = len(old_sessions)
        for user_session in old_sessions:
            session.delete(user_session)

        session.commit()
        logger.info(f"Deleted {count} old sessions")
        return count

    @classmethod
    def get_session_statistics(cls, session) -> Dict[str, Any]:
        """Get session statistics."""
        total_sessions = session.query(cls).count()
        active_sessions = (
            session.query(cls)
            .filter(cls.is_active == True, cls.expires_at > datetime.utcnow())
            .count()
        )
        processing_sessions = (
            session.query(cls)
            .filter(
                cls.is_active == True,
                cls.current_transcription_id.isnot(None),
            )
            .count()
        )
        queued_sessions = (
            session.query(cls)
            .filter(
                cls.is_active == True,
                cls.queue_position > 0,
                cls.current_transcription_id.is_(None),
            )
            .count()
        )

        # Get average processing time and efficiency
        from sqlalchemy import func

        avg_processing_time = (
            session.query(func.avg(cls.total_processing_time))
            .filter(cls.total_files_processed > 0)
            .scalar()
            or 0.0
        )

        avg_efficiency = (
            session.query(
                func.avg(cls.total_audio_duration / cls.total_processing_time)
            )
            .filter(cls.total_processing_time > 0)
            .scalar()
            or 0.0
        )

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "processing_sessions": processing_sessions,
            "queued_sessions": queued_sessions,
            "average_processing_time": avg_processing_time,
            "average_efficiency": avg_efficiency,
        }
