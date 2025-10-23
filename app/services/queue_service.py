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

from ..core.config import get_settings
from ..core.database import get_database
from ..models.processing_queue import ProcessingQueue
from ..models.transcription import Transcription
from ..models.session import UserSession
from ..utils.exceptions import QueueError
from .transcription_service import TranscriptionService
from .diarization_service import DiarizationService

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
            with next(get_database()) as db:
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
            with next(get_database()) as db:
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
            with next(get_database()) as db:
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
        """Main queue processor loop."""
        logger.info("Queue processor started")

        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get next job
                with next(get_database()) as db:
                    job = ProcessingQueue.get_next_job(db)

                    if job:
                        # Submit job for processing
                        self._process_job(job)
                    else:
                        # No jobs available, wait
                        self.shutdown_event.wait(1)

            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                self.shutdown_event.wait(5)

        logger.info("Queue processor stopped")

    def _process_job(self, job: ProcessingQueue) -> None:
        """Process a job asynchronously."""
        try:
            # Mark job as started
            with next(get_database()) as db:
                job.mark_as_started(worker_id=f"worker_{threading.get_ident()}")

                # Update transcription status
                transcription = (
                    db.query(Transcription)
                    .filter(Transcription.id == job.transcription_id)
                    .first()
                )
                if transcription:
                    transcription.mark_as_started()

                db.commit()

            # Submit to thread pool
            future = self.executor.submit(self._execute_job, job)
            self.active_jobs[job.job_id] = future

            # Add completion callback
            future.add_done_callback(lambda f: self._job_completed(job.job_id, f))

            logger.info(f"Submitted job {job.job_id} for processing")

        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")
            self._mark_job_failed(job, str(e))

    def _execute_job(self, job: ProcessingQueue) -> Dict[str, Any]:
        """Execute the actual transcription and diarization."""
        try:
            logger.info(f"Starting execution of job {job.job_id}")

            with next(get_database()) as db:
                # Get transcription
                transcription = (
                    db.query(Transcription)
                    .filter(Transcription.id == job.transcription_id)
                    .first()
                )

                if not transcription:
                    raise QueueError(f"Transcription not found: {job.transcription_id}")

                # Progress callback
                def progress_callback(percentage: float, step: str) -> None:
                    job.update_progress(percentage, step)
                    transcription.update_progress(percentage, step)
                    db.commit()

                # Step 1: Transcription (40% of progress)
                progress_callback(0, "Starting transcription")
                transcription_result = self.transcription_service.transcribe_audio(
                    job.file_path,
                    transcription,
                    db,
                    progress_callback=progress_callback,
                )

                # Step 2: Diarization (40% of progress)
                progress_callback(40, "Starting speaker diarization")
                diarization_result = self.diarization_service.diarize_audio(
                    job.file_path,
                    transcription,
                    db,
                    progress_callback=progress_callback,
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

                progress_callback(100, "Completed")

                result = {
                    "transcription": transcription_result,
                    "diarization": diarization_result,
                    "status": "completed",
                }

                logger.info(f"Job {job.job_id} completed successfully")
                return result

        except Exception as e:
            logger.error(f"Job execution failed {job.job_id}: {e}")
            raise QueueError(f"Job execution failed: {str(e)}")

    def _job_completed(self, job_id: str, future) -> None:
        """Handle job completion."""
        try:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            # Update job status in database
            with next(get_database()) as db:
                job = (
                    db.query(ProcessingQueue)
                    .filter(ProcessingQueue.job_id == job_id)
                    .first()
                )

                if job:
                    try:
                        result = future.result()
                        job.mark_as_completed()

                        # Update transcription
                        transcription = (
                            db.query(Transcription)
                            .filter(Transcription.id == job.transcription_id)
                            .first()
                        )
                        if transcription:
                            transcription.mark_as_completed()

                        logger.info(f"Job {job_id} marked as completed")

                    except Exception as e:
                        self._mark_job_failed(job, str(e))

                db.commit()

        except Exception as e:
            logger.error(f"Error handling job completion: {e}")

    def _mark_job_failed(self, job: ProcessingQueue, error_message: str) -> None:
        """Mark job as failed."""
        try:
            with next(get_database()) as db:
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

        except Exception as e:
            logger.error(f"Failed to mark job as failed: {e}")

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
            with next(get_database()) as db:
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
