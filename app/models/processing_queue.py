"""
Processing queue model for managing transcription queue.
Handles job queuing, priority management, and processing coordination.
"""

import json
import logging
from datetime import datetime, timedelta
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

from app.core.database import Base
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ProcessingQueue(Base):
    """
    Processing queue model for managing transcription jobs.
    Handles job queuing, priority management, and processing coordination.
    """

    __tablename__ = "processing_queue"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    transcription_id = Column(Integer, ForeignKey("transcriptions.id"), nullable=True)

    # Queue information
    queue_position = Column(Integer, default=0, nullable=False, index=True)
    priority = Column(
        Integer, default=5, nullable=False, index=True
    )  # 1-10, 1 = highest
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Status and progress
    status = Column(
        String(50), default="queued", nullable=False, index=True
    )  # queued, processing, completed, failed, cancelled
    progress_percentage = Column(Float, default=0.0, nullable=False)
    current_step = Column(String(100), nullable=True)
    estimated_duration = Column(
        Float, nullable=True
    )  # Estimated processing time in seconds
    actual_duration = Column(Float, nullable=True)  # Actual processing time in seconds

    # Job metadata
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_duration = Column(Float, nullable=False)
    processing_options = Column(JSON, nullable=True)  # Processing configuration options

    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)

    # Resource allocation
    assigned_worker = Column(String(255), nullable=True)  # Worker ID or process name
    gpu_assigned = Column(Boolean, default=False, nullable=False)
    memory_allocated = Column(Integer, nullable=True)  # Memory in MB

    # Relationships
    transcription = relationship("Transcription", foreign_keys=[transcription_id])

    def __repr__(self) -> str:
        return f"<ProcessingQueue(id={self.id}, job_id='{self.job_id}', status={self.status})>"

    @hybrid_property
    def is_queued(self) -> bool:
        """Check if job is in queue."""
        return self.status == "queued"

    @hybrid_property
    def is_processing(self) -> bool:
        """Check if job is currently processing."""
        return self.status == "processing"

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status == "completed"

    @hybrid_property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == "failed"

    @hybrid_property
    def is_cancelled(self) -> bool:
        """Check if job was cancelled."""
        return self.status == "cancelled"

    @hybrid_property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.is_failed and self.retry_count < self.max_retries

    @hybrid_property
    def wait_time(self) -> float:
        """Get wait time in seconds."""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()

    @wait_time.expression
    def wait_time(cls):
        """SQL expression for wait_time."""
        from sqlalchemy import extract

        return cls.started_at - cls.created_at

    @hybrid_property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None

    @processing_time.expression
    def processing_time(cls):
        """SQL expression for processing_time."""
        return cls.completed_at - cls.started_at

    @hybrid_property
    def total_time(self) -> float:
        """Get total time in seconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()

    @total_time.expression
    def total_time(cls):
        """SQL expression for total_time."""
        return cls.completed_at - cls.created_at

    @hybrid_property
    def estimated_completion(self) -> Optional[datetime]:
        """Get estimated completion time."""
        if not self.estimated_duration:
            return None

        if self.started_at:
            return self.started_at + timedelta(seconds=self.estimated_duration)
        else:
            # Estimate based on queue position
            settings = get_settings()
            avg_processing_time = 300  # 5 minutes default
            queue_delay = self.queue_position * avg_processing_time
            return datetime.utcnow() + timedelta(
                seconds=queue_delay + self.estimated_duration
            )

    def mark_as_started(self, worker_id: Optional[str] = None) -> None:
        """Mark job as started processing."""
        self.status = "processing"
        self.started_at = datetime.utcnow()
        self.queue_position = 0
        self.progress_percentage = 0.0
        self.current_step = "Initializing processing"

        if worker_id:
            self.assigned_worker = worker_id

        logger.info(f"Started processing job {self.job_id}")

    def mark_as_completed(self) -> None:
        """Mark job as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        self.current_step = "Completed"

        if self.started_at:
            self.actual_duration = self.processing_time

        logger.info(f"Completed job {self.job_id} in {self.actual_duration} seconds")

    def mark_as_failed(
        self, error_message: str, error_traceback: Optional[str] = None
    ) -> None:
        """Mark job as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if error_traceback:
            self.error_traceback = error_traceback

        if self.started_at:
            self.actual_duration = self.processing_time

        self.retry_count += 1
        logger.error(
            f"Job {self.job_id} failed (attempt {self.retry_count}): {error_message}"
        )

    def mark_as_cancelled(self) -> None:
        """Mark job as cancelled."""
        self.status = "cancelled"
        self.completed_at = datetime.utcnow()
        logger.info(f"Cancelled job {self.job_id}")

    def retry_job(self) -> bool:
        """Retry a failed job."""
        if not self.can_retry:
            return False

        self.status = "queued"
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.error_traceback = None
        self.assigned_worker = None
        self.gpu_assigned = False
        self.memory_allocated = None
        self.progress_percentage = 0.0
        self.current_step = "Queued for retry"

        logger.info(f"Retrying job {self.job_id} (attempt {self.retry_count + 1})")
        return True

    def update_progress(self, percentage: float, step: Optional[str] = None) -> None:
        """Update job progress."""
        self.progress_percentage = max(0, min(100, percentage))
        if step:
            self.current_step = step

    def update_queue_position(self, position: int) -> None:
        """Update queue position."""
        self.queue_position = max(0, position)

    def set_priority(self, priority: int) -> None:
        """Set job priority (1-10, 1 = highest)."""
        self.priority = max(1, min(10, priority))

    def assign_resources(
        self, worker_id: str, gpu: bool = False, memory_mb: int = None
    ) -> None:
        """Assign processing resources to job."""
        self.assigned_worker = worker_id
        self.gpu_assigned = gpu
        if memory_mb:
            self.memory_allocated = memory_mb

    def estimate_processing_time(self) -> float:
        """Estimate processing time based on file characteristics."""
        # Base estimation: 1 second of audio = 3 seconds of processing on CPU
        # Faster on GPU: 1 second of audio = 0.5 seconds of processing
        base_factor = 3.0 if not self.gpu_assigned else 0.5

        # Adjust for file size and complexity
        size_factor = max(1.0, self.file_size / (50 * 1024 * 1024))  # 50MB base
        duration_factor = max(1.0, self.file_duration / 300)  # 5 minutes base

        estimated_time = (
            self.file_duration * base_factor * size_factor * duration_factor
        )
        self.estimated_duration = estimated_time

        return estimated_time

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        result = {
            "id": self.id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "queue_position": self.queue_position,
            "priority": self.priority,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "wait_time": self.wait_time,
            "processing_time": self.processing_time,
            "total_time": self.total_time,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "file_size": self.file_size,
            "file_duration": self.file_duration,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "can_retry": self.can_retry,
            "estimated_completion": self.estimated_completion.isoformat()
            if self.estimated_completion
            else None,
        }

        if include_sensitive:
            result.update(
                {
                    "file_path": self.file_path,
                    "processing_options": self.processing_options,
                    "error_message": self.error_message,
                    "error_traceback": self.error_traceback,
                    "assigned_worker": self.assigned_worker,
                    "gpu_assigned": self.gpu_assigned,
                    "memory_allocated": self.memory_allocated,
                }
            )

        return result

    @classmethod
    def create_job(
        cls,
        session,
        job_id: str,
        session_id: str,
        file_path: str,
        file_size: int,
        file_duration: float,
        transcription_id: Optional[int] = None,
        processing_options: Optional[Dict] = None,
        priority: int = 5,
    ) -> "ProcessingQueue":
        """Create a new processing job."""
        job = cls(
            job_id=job_id,
            session_id=session_id,
            file_path=file_path,
            file_size=file_size,
            file_duration=file_duration,
            transcription_id=transcription_id,
            processing_options=processing_options or {},
            priority=priority,
        )

        # Estimate processing time
        job.estimate_processing_time()

        session.add(job)
        session.commit()
        session.refresh(job)

        logger.info(f"Created processing job: {job_id}")
        return job

    @classmethod
    def get_next_job(
        cls, session, worker_id: Optional[str] = None
    ) -> Optional["ProcessingQueue"]:
        """Get the next job in queue."""
        return (
            session.query(cls)
            .filter(cls.status == "queued")
            .order_by(cls.priority.asc(), cls.created_at.asc())
            .first()
        )

    @classmethod
    def get_queued_jobs(cls, session, limit: int = 50) -> List["ProcessingQueue"]:
        """Get queued jobs."""
        return (
            session.query(cls)
            .filter(cls.status == "queued")
            .order_by(cls.priority.asc(), cls.created_at.asc())
            .limit(limit)
            .all()
        )

    @classmethod
    def get_processing_jobs(cls, session) -> List["ProcessingQueue"]:
        """Get currently processing jobs."""
        return session.query(cls).filter(cls.status == "processing").all()

    @classmethod
    def get_jobs_by_session(cls, session, session_id: str) -> List["ProcessingQueue"]:
        """Get jobs for a specific session."""
        return (
            session.query(cls)
            .filter(cls.session_id == session_id)
            .order_by(cls.created_at.desc())
            .all()
        )

    @classmethod
    def update_queue_positions(cls, session) -> None:
        """Update queue positions for all queued jobs."""
        queued_jobs = (
            session.query(cls)
            .filter(cls.status == "queued")
            .order_by(cls.priority.asc(), cls.created_at.asc())
            .all()
        )

        for position, job in enumerate(queued_jobs, 1):
            job.queue_position = position

        session.commit()
        logger.info(f"Updated queue positions for {len(queued_jobs)} jobs")

    @classmethod
    def cleanup_completed_jobs(cls, session, hours_old: int = 24) -> int:
        """Clean up old completed jobs."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)

        completed_jobs = (
            session.query(cls)
            .filter(
                cls.status.in_(["completed", "cancelled"]),
                cls.completed_at < cutoff_time,
            )
            .all()
        )

        count = len(completed_jobs)
        for job in completed_jobs:
            session.delete(job)

        session.commit()
        logger.info(f"Cleaned up {count} completed jobs")
        return count

    @classmethod
    def get_queue_statistics(cls, session) -> Dict[str, Any]:
        """Get queue statistics."""
        from sqlalchemy import func

        total_jobs = session.query(cls).count()
        queued_jobs = session.query(cls).filter(cls.status == "queued").count()
        processing_jobs = session.query(cls).filter(cls.status == "processing").count()
        completed_jobs = session.query(cls).filter(cls.status == "completed").count()
        failed_jobs = session.query(cls).filter(cls.status == "failed").count()

        # Average wait time
        avg_wait_time = (
            session.query(
                func.avg(func.extract("epoch", cls.started_at - cls.created_at))
            )
            .filter(cls.status.in_(["processing", "completed", "failed"]))
            .filter(cls.started_at.isnot(None))
            .scalar()
            or 0.0
        )

        # Average processing time
        avg_processing_time = (
            session.query(
                func.avg(func.extract("epoch", cls.completed_at - cls.started_at))
            )
            .filter(cls.status == "completed", cls.completed_at.isnot(None))
            .scalar()
            or 0.0
        )

        return {
            "total_jobs": total_jobs,
            "queued_jobs": queued_jobs,
            "processing_jobs": processing_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "average_wait_time": avg_wait_time,
            "average_processing_time": avg_processing_time,
            "success_rate": (completed_jobs / total_jobs * 100)
            if total_jobs > 0
            else 0,
        }
