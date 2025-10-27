"""
Queue API router for SecureTranscribe.
Handles queue status, job management, and processing coordination endpoints.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Body,
)
from sqlalchemy.orm import Session

from app.core.database import get_database
from app.models.session import UserSession
from app.models.processing_queue import ProcessingQueue
from app.services.queue_service import get_queue_service
from app.utils.exceptions import (
    SecureTranscribeError,
    ValidationError,
    QueueError,
)
from app.api.transcription import get_current_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status")
async def get_queue_status(
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Get current queue status and statistics.

    Args:
        db: Database session
        user_session: Current user session

    Returns:
        Queue status information
    """
    try:
        queue_service = get_queue_service()
        status = queue_service.get_queue_status()

        # Add user-specific information
        user_queue_position = queue_service.get_user_queue_position(
            user_session.session_id
        )
        user_wait_time = queue_service.estimate_wait_time(user_session.session_id)

        status.update(
            {
                "user_queue_position": user_queue_position,
                "estimated_wait_time": user_wait_time,
            }
        )

        return status

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise QueueError(f"Failed to get queue status: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Get status of a specific job.

    Args:
        job_id: Job ID to check
        db: Database session
        user_session: Current user session

    Returns:
        Job status information
    """
    try:
        queue_service = get_queue_service()
        job_status = queue_service.get_job_status(job_id)

        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")

        # Verify job belongs to current user
        if job_status.get("session_id") != user_session.session_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return job_status

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise QueueError(f"Failed to get job status: {str(e)}")


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Cancel a job in the queue.

    Args:
        job_id: Job ID to cancel
        db: Database session
        user_session: Current user session

    Returns:
        Cancellation result
    """
    try:
        # Verify job ownership
        job = (
            db.query(ProcessingQueue)
            .filter(
                ProcessingQueue.job_id == job_id,
                ProcessingQueue.session_id == user_session.session_id,
            )
            .first()
        )

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        queue_service = get_queue_service()
        success = queue_service.cancel_job(job_id)

        if success:
            return {
                "success": True,
                "message": "Job cancelled successfully",
                "job_id": job_id,
            }
        else:
            return {
                "success": False,
                "message": "Job could not be cancelled (may have already completed)",
                "job_id": job_id,
            }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise QueueError(f"Failed to cancel job: {str(e)}")


@router.get("/jobs")
async def list_user_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    List jobs for the current user session.

    Args:
        status_filter: Filter by job status
        limit: Maximum number of results
        offset: Results offset
        db: Database session
        user_session: Current user session

    Returns:
        List of user jobs
    """
    try:
        query = db.query(ProcessingQueue).filter(
            ProcessingQueue.session_id == user_session.session_id
        )

        if status_filter:
            query = query.filter(ProcessingQueue.status == status_filter)

        query = query.order_by(ProcessingQueue.created_at.desc())
        query = query.offset(offset).limit(limit)

        jobs = query.all()

        result = []
        for job in jobs:
            result.append(job.to_dict())

        return {
            "jobs": result,
            "total": len(result),
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
        }

    except Exception as e:
        logger.error(f"Failed to list user jobs: {e}")
        raise QueueError(f"Failed to list user jobs: {str(e)}")


@router.get("/worker-status")
async def get_worker_status(
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Get worker pool status.

    Args:
        db: Database session
        user_session: Current user session

    Returns:
        Worker status information
    """
    try:
        queue_service = get_queue_service()
        worker_status = queue_service.get_worker_status()

        return worker_status

    except Exception as e:
        logger.error(f"Failed to get worker status: {e}")
        raise QueueError(f"Failed to get worker status: {str(e)}")


@router.post("/cleanup")
async def cleanup_completed_jobs(
    hours_old: int = Body(24, embed=True),
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Clean up old completed jobs (admin function).

    Args:
        hours_old: Age in hours for jobs to clean up
        db: Database session
        user_session: Current user session

    Returns:
        Cleanup result
    """
    try:
        if hours_old < 1 or hours_old > 168:  # Max 1 week
            raise ValidationError("hours_old must be between 1 and 168")

        queue_service = get_queue_service()
        cleaned_count = queue_service.cleanup_completed_jobs(hours_old)

        return {
            "success": True,
            "cleaned_jobs": cleaned_count,
            "hours_old": hours_old,
            "message": f"Cleaned up {cleaned_count} jobs older than {hours_old} hours",
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup jobs: {e}")
        raise QueueError(f"Failed to cleanup jobs: {str(e)}")


@router.get("/estimate-wait-time")
async def estimate_wait_time(
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Estimate wait time for user's next job.

    Args:
        db: Database session
        user_session: Current user session

    Returns:
        Wait time estimate
    """
    try:
        queue_service = get_queue_service()
        wait_time = queue_service.estimate_wait_time(user_session.session_id)
        queue_position = queue_service.get_user_queue_position(user_session.session_id)

        return {
            "estimated_wait_time_seconds": wait_time,
            "estimated_wait_time_minutes": wait_time / 60 if wait_time else None,
            "queue_position": queue_position,
            "jobs_ahead": max(0, queue_position - 1) if queue_position else 0,
        }

    except Exception as e:
        logger.error(f"Failed to estimate wait time: {e}")
        raise QueueError(f"Failed to estimate wait time: {str(e)}")


@router.post("/restart-service")
async def restart_queue_service(
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Restart the queue service (admin function).

    Args:
        db: Database session
        user_session: Current user session

    Returns:
        Restart result
    """
    try:
        queue_service = get_queue_service()

        # Stop and restart the service
        queue_service.stop()
        queue_service.start()

        return {
            "success": True,
            "message": "Queue service restarted successfully",
            "timestamp": queue_service.get_queue_status().get("timestamp"),
        }

    except Exception as e:
        logger.error(f"Failed to restart queue service: {e}")
        raise QueueError(f"Failed to restart queue service: {str(e)}")
