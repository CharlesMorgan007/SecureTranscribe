"""
Queue service for managing transcription processing queue.
Handles job scheduling, priority management, and worker coordination.
"""

import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from app.core.config import get_settings
from app.core.database import get_database, get_database_manager
from app.utils.exceptions import QueueError

from app.models.processing_queue import ProcessingQueue
from app.models.transcription import Transcription
from app.services.transcription_service import TranscriptionService
from app.services.diarization_service import DiarizationService

logger = logging.getLogger(__name__)


class QueueService:
    """
    Queue service for managing transcription processing queue.
    Handles job scheduling, priority management, and worker coordination.
    """

    def __init__(self):
        self.settings = get_settings()
        self.max_workers = self.settings.max_workers
        self.queue_size = self.settings.queue_size
        self.processing_timeout = self.settings.processing_timeout

        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_jobs = {}  # job_id -> future
        self.worker_threads = {}

        # Queue management
        self.is_running = False
        self.queue_thread = None
        self.shutdown_event = threading.Event()

        # Database manager
        self.db_manager = get_database_manager()

        # Services
        self.transcription_service = None
        self.diarization_service = None

    def start(self) -> None:
        """Start the queue service."""
        try:
            if self.is_running:
                return

            self.is_running = True
            self.shutdown_event.clear()

            # Initialize services
            self.transcription_service = TranscriptionService()
            self.diarization_service = DiarizationService()

            # Start queue processor thread
            self.queue_thread = threading.Thread(
                target=self._queue_processor, daemon=True
            )
            self.queue_thread.start()

            logger.info("Queue service started")

        except Exception as e:
            logger.error(f"Failed to start queue service: {e}")
            raise QueueError(f"Failed to start queue service: {str(e)}")

    def stop(self) -> None:
        """Stop the queue service."""
        try:
            if not self.is_running:
                return

            logger.info("Stopping queue service...")

            # Signal shutdown
            self.shutdown_event.set()
            self.is_running = False

            # Wait for queue processor thread
            if self.queue_thread and self.queue_thread.is_alive():
                self.queue_thread.join(timeout=10)

            # Cancel active jobs
            for job_id, future in self.active_jobs.items():
                if not future.done():
                    future.cancel()
                    logger.info(f"Cancelled job: {job_id}")

            # Shutdown executor
            self.executor.shutdown(wait=True)

            # Cleanup services
            if self.transcription_service:
                self.transcription_service.cleanup()
            if self.diarization_service:
                self.diarization_service.cleanup()

            logger.info("Queue service stopped")

        except Exception as e:
            logger.error(f"Error stopping queue service: {e}")

    def submit_job(
        self,
        session_id: str,
        file_path: str,
        file_size: int,
        file_duration: float,
        transcription_id: int,
        priority: int = 5,
        processing_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a new transcription job to the queue.

        Args:
            session_id: User session ID
            file_path: Path to audio file
            file_size: File size in bytes
            file_duration: File duration in seconds
            transcription_id: Transcription model ID
            priority: Job priority (1-10, 1 = highest)
            processing_options: Additional processing options

        Returns:
            Job ID
        """
        try:
            # Check queue capacity
            with self.db_manager.get_session() as db:
                queued_count = (
                    db.query(ProcessingQueue)
                    .filter(ProcessingQueue.status == "queued")
                    .count()
                )

                if queued_count >= self.queue_size:
                    raise QueueError(f"Queue is full ({self.queue_size} jobs)")

                # Generate job ID
                job_id = str(uuid.uuid4())

                # Create queue entry
                job = ProcessingQueue.create_job(
                    session=db,
                    job_id=job_id,
                    session_id=session_id,
                    file_path=file_path,
                    file_size=file_size,
                    file_duration=file_duration,
                    transcription_id=transcription_id,
                    processing_options=processing_options,
                    priority=priority,
                )

                # Update queue positions
                ProcessingQueue.update_queue_positions(db)

                logger.info(f"Submitted job {job_id} to queue")
                return job_id

        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise QueueError(f"Failed to submit job: {str(e)}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        try:
            with self.db_manager.get_session() as db:
                stats = ProcessingQueue.get_queue_statistics(db)

                # Add active job information
                stats["active_jobs"] = len(self.active_jobs)
                stats["max_workers"] = self.max_workers
                stats["is_running"] = self.is_running

                return stats

        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"error": str(e)}

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        try:
            with self.db_manager.get_session() as db:
                job = (
                    db.query(ProcessingQueue)
                    .filter(ProcessingQueue.job_id == job_id)
                    .first()
                )

                if job:
                    return job.to_dict()
                return None

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job in the queue."""
        try:
            with next(get_database()) as db:
                job = (
                    db.query(ProcessingQueue)
                    .filter(ProcessingQueue.job_id == job_id)
                    .first()
                )

                if not job:
                    return False

                # Cancel if queued
                if job.status == "queued":
                    job.mark_as_cancelled()
                    ProcessingQueue.update_queue_positions(db)
                    logger.info(f"Cancelled queued job: {job_id}")
                    return True

                # Cancel if processing
                elif job.status == "processing":
                    future = self.active_jobs.get(job_id)
                    if future and not future.done():
                        future.cancel()
                        job.mark_as_cancelled()
                        logger.info(f"Cancelled processing job: {job_id}")
                        return True

                return False

        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    def _queue_processor(self) -> None:
        """Background queue processor."""
        logger.info("Queue processor started")

        while not self.shutdown_event.is_set():
            try:
                # Get next job with thread-safe database access
                def get_next_job_operation():
                    with self.db_manager.get_session() as db:
                        job = ProcessingQueue.get_next_job(db)
                        if job:
                            # Mark job as started atomically
                            job.mark_as_started(
                                worker_id=f"worker_{threading.get_ident()}"
                            )

                            # Update transcription status
                            transcription = (
                                db.query(Transcription)
                                .filter(Transcription.id == job.transcription_id)
                                .first()
                            )
                            if transcription:
                                transcription.mark_as_started()

                            db.commit()
                        return job

                job = self.db_manager.execute_with_retry(get_next_job_operation)

                if job:
                    # Submit job for processing
                    self._process_job(job)
                else:
                    # No jobs available, wait
                    self.shutdown_event.wait(1)
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                time.sleep(5)
                self.shutdown_event.wait(5)

        logger.info("Queue processor stopped")

    def _process_job(self, job: ProcessingQueue) -> None:
        """Process a job asynchronously."""
        try:
            # Job is already marked as started in _queue_processor
            # Submit to thread pool
            future = self.executor.submit(self._execute_job, job)
            self.active_jobs[job.job_id] = future

            # Add completion callback
            future.add_done_callback(lambda f: self._job_completed(job.job_id, f))

            logger.info(f"Submitted job {job.job_id} for processing")

        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")
            self._mark_job_failed(job.job_id, str(e))

    def _execute_job(self, job: ProcessingQueue) -> Dict[str, Any]:
        """Execute the actual transcription and diarization."""
        # Store job_id to refetch job in proper session context
        job_id = job.job_id

        def execute_job_operation():
            with self.db_manager.write_lock():
                with self.db_manager.get_session() as db:
                    # Re-fetch the job and transcription in this session context
                    job = (
                        db.query(ProcessingQueue)
                        .filter(ProcessingQueue.job_id == job_id)
                        .first()
                    )
                    if not job:
                        raise QueueError(f"Job not found: {job_id}")

                    transcription = (
                        db.query(Transcription)
                        .filter(Transcription.id == job.transcription_id)
                        .first()
                    )

                    if not transcription:
                        raise QueueError(
                            f"Transcription not found: {job.transcription_id}"
                        )

                    # Progress callback
                    def progress_callback(percentage: float, step: str) -> None:
                        job.update_progress(percentage, step)
                        transcription.update_progress(percentage, step)
                        # Commit progress updates immediately to avoid connection issues
                        try:
                            db.commit()
                        except Exception as commit_error:
                            logger.warning(
                                f"Failed to commit progress update: {commit_error}"
                            )
                            db.rollback()

                    # Step 1: Transcription (40% of progress)
                    progress_callback(0, "Starting transcription")
                    transcription_result = self.transcription_service.transcribe_audio(
                        job.file_path,
                        transcription,
                        db,
                        progress_callback=progress_callback,
                    )

                    # Commit after transcription
                    try:
                        db.commit()
                    except Exception as commit_error:
                        logger.error(f"Failed to commit transcription: {commit_error}")
                        db.rollback()
                        raise QueueError(
                            f"Failed to commit transcription: {str(commit_error)}"
                        )

                    # Step 2: Diarization (40% of progress)
                    progress_callback(40, "Starting speaker diarization")
                    diarization_result = self.diarization_service.diarize_audio(
                        job.file_path,
                        transcription,
                        db,
                        progress_callback=progress_callback,
                    )

                    # Commit after diarization
                    try:
                        db.commit()
                    except Exception as commit_error:
                        logger.error(f"Failed to commit diarization: {commit_error}")
                        db.rollback()
                        raise QueueError(
                            f"Failed to commit diarization: {str(commit_error)}"
                        )

                    # Step 3: Final processing (20% of progress)
                    progress_callback(80, "Finalizing results")

                    # Update speaker assignments in transcription
                    if diarization_result.get("speaker_matches"):
                        for segment in transcription.segments or []:
                            speaker_label = segment.get("speaker", "")
                            matched_speaker = diarization_result["speaker_matches"].get(
                                speaker_label
                            )
                            if matched_speaker:
                                segment["speaker"] = matched_speaker.name

                    # Mark job and transcription as completed
                    job.mark_as_completed()
                    transcription.mark_as_completed()

                    # Commit the completion
                    db.commit()
                    logger.info(f"Job {job.job_id} committed successfully")

                    progress_callback(100, "Completed")

                    result = {
                        "transcription": transcription_result,
                        "diarization": diarization_result,
                        "status": "completed",
                    }

                    logger.info(f"Job {job.job_id} completed successfully")
                    return result

        return self.db_manager.execute_with_retry(execute_job_operation)

    def _job_completed(self, job_id: str, future) -> None:
        """Handle job completion."""
        try:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            # The job is already completed and committed in _execute_job
            # Just log the completion and handle any exceptions from the future
            try:
                result = future.result()
                logger.info(f"Job {job_id} completed successfully")
            except Exception as e:
                logger.error(f"Job {job_id} failed in execution: {e}")
                # Mark job as failed in database if not already completed
                self._mark_job_failed_from_future(job_id, str(e))

        except Exception as e:
            logger.error(f"Error handling job completion: {e}")

    def _mark_job_failed(self, job_id: str, error_message: str) -> None:
        """Mark job as failed with thread-safe database access."""

        def mark_failed_operation():
            with self.db_manager.write_lock():
                with self.db_manager.get_session() as db:
                    # Re-fetch the job to ensure we have the latest state
                    job = (
                        db.query(ProcessingQueue)
                        .filter(ProcessingQueue.job_id == job_id)
                        .first()
                    )
                    if job:
                        job.mark_as_failed(error_message)

                        # Update transcription
                        transcription = (
                            db.query(Transcription)
                            .filter(Transcription.id == job.transcription_id)
                            .first()
                        )
                        if transcription:
                            transcription.mark_as_failed(error_message)

                        db.commit()
                        logger.info(f"Job {job_id} marked as failed")
                        return True
                    return False

        try:
            result = self.db_manager.execute_with_retry(mark_failed_operation)
            if not result:
                logger.warning(f"Failed to mark job {job_id} as failed - job not found")
        except Exception as e:
            logger.error(f"Failed to mark job as failed: {e}")

    def _mark_job_failed_from_future(self, job_id: str, error_message: str) -> None:
        """Mark job as failed when called from future result."""

        def mark_failed_operation():
            with self.db_manager.write_lock():
                with self.db_manager.get_session() as db:
                    job = (
                        db.query(ProcessingQueue)
                        .filter(ProcessingQueue.job_id == job_id)
                        .first()
                    )
                    if job and not job.is_completed():
                        job.mark_as_failed(error_message)

                        # Update transcription
                        transcription = (
                            db.query(Transcription)
                            .filter(Transcription.id == job.transcription_id)
                            .first()
                        )
                        if transcription:
                            transcription.mark_as_failed(error_message)

                        db.commit()
                        logger.info(f"Job {job_id} marked as failed from future")
                        return True
                    return False

        try:
            result = self.db_manager.execute_with_retry(mark_failed_operation)
            if not result:
                logger.warning(
                    f"Failed to mark job {job_id} as failed from future - job not found or already completed"
                )
        except Exception as e:
            logger.error(f"Failed to mark job as failed from future: {e}")

    def get_user_queue_position(self, session_id: str) -> Optional[int]:
        """Get queue position for a user session."""
        try:
            with next(get_database()) as db:
                jobs = (
                    db.query(ProcessingQueue)
                    .filter(
                        ProcessingQueue.session_id == session_id,
                        ProcessingQueue.status == "queued",
                    )
                    .order_by(ProcessingQueue.queue_position)
                    .all()
                )

                if jobs:
                    return jobs[0].queue_position
                return None

        except Exception as e:
            logger.error(f"Failed to get user queue position: {e}")
            return None

    def estimate_wait_time(self, session_id: str) -> Optional[float]:
        """Estimate wait time for user's next job."""
        try:
            position = self.get_user_queue_position(session_id)
            if position is None:
                return None

            # Average processing time estimate (5 minutes)
            avg_processing_time = 300  # seconds

            # Calculate estimated wait time
            estimated_wait = (position - 1) * avg_processing_time

            return estimated_wait

        except Exception as e:
            logger.error(f"Failed to estimate wait time: {e}")
            return None

    def cleanup_completed_jobs(self, hours_old: int = 24) -> int:
        """Clean up old completed jobs."""
        try:
            with self.db_manager.get_session() as db:
                return ProcessingQueue.cleanup_completed_jobs(db, hours_old)

        except Exception as e:
            logger.error(f"Failed to cleanup completed jobs: {e}")
            return 0

    def get_worker_status(self) -> Dict[str, Any]:
        """Get worker pool status."""
        return {
            "max_workers": self.max_workers,
            "active_jobs": len(self.active_jobs),
            "available_workers": self.max_workers - len(self.active_jobs),
            "queue_size": self.queue_size,
            "is_running": self.is_running,
        }


# Global queue service instance
queue_service = QueueService()


def get_queue_service() -> QueueService:
    """Get the global queue service instance."""
    return queue_service
