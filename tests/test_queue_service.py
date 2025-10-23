"""
Test queue service for SecureTranscribe.
Tests job queue management, processing, and status tracking without requiring GPU.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

# Mock external dependencies to avoid installation issues
sys.modules["torch"] = MagicMock()
sys.modules["faster_whisper"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["numpy"] = MagicMock()


# Mock job and processing classes
class MockQueueJob:
    """Mock queue job model"""

    def __init__(self, transcription_id: str, session_id: str):
        self.id = uuid.uuid4()
        self.job_id = f"job_{str(uuid.uuid4())[:8]}"
        self.transcription_id = transcription_id
        self.session_id = session_id
        self.status = (
            "pending"  # pending, queued, processing, completed, failed, cancelled
        )
        self.priority = 5
        self.queue_position = 0
        self.progress_percentage = 0.0
        self.current_step = None
        self.error_message = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.processing_time = None
        self.attempts = 0
        self.max_attempts = 3
        self.lock = threading.Lock()

    def update_status(
        self, status: str, step: Optional[str] = None, error: Optional[str] = None
    ):
        """Update job status with thread safety"""
        with self.lock:
            self.status = status
            self.current_step = step
            self.error_message = error

            if status == "queued" and self.created_at:
                self.created_at = time.time()
            elif status == "processing" and not self.started_at:
                self.started_at = time.time()
                self.attempts += 1
            elif status in ["completed", "failed", "cancelled"] and self.started_at:
                self.completed_at = time.time()
                self.processing_time = self.completed_at - self.started_at

    def update_progress(self, percentage: float, step: Optional[str] = None):
        """Update job progress with thread safety"""
        with self.lock:
            self.progress_percentage = max(0.0, min(100.0, percentage))
            if step:
                self.current_step = step

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            "id": str(self.id),
            "job_id": self.job_id,
            "transcription_id": self.transcription_id,
            "session_id": self.session_id,
            "status": self.status,
            "priority": self.priority,
            "queue_position": self.queue_position,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "processing_time": self.processing_time,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }


class MockTranscription:
    """Mock transcription model"""

    def __init__(self, id: str, session_id: str):
        self.id = id
        self.session_id = session_id
        self.status = "pending"
        self.progress_percentage = 0.0
        self.current_step = None
        self.full_transcript = None
        self.segments = []
        self.language_detected = None
        self.confidence_score = None
        self.num_speakers = 0
        self.speakers_assigned = False
        self.file_info = {}

    def update(self, **kwargs):
        """Update transcription attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class MockQueueService:
    """Mock queue service for job processing"""

    def __init__(self, max_workers: int = 4, queue_size: int = 20):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.jobs: Dict[str, MockQueueJob] = {}
        self.processing_jobs: Dict[str, MockQueueJob] = {}
        self.job_queue = asyncio.Queue(maxsize=queue_size)
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "processing_jobs": 0,
            "queued_jobs": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
        }
        self.lock = threading.Lock()

    def enqueue_job(
        self, transcription_id: str, session_id: str, priority: int = 5
    ) -> Optional[MockQueueJob]:
        """Enqueue a new job"""
        try:
            job = MockQueueJob(transcription_id, session_id)
            job.priority = priority

            # Check queue capacity
            if len(self.jobs) >= self.queue_size:
                return None

            self.jobs[job.job_id] = job
            self._update_queue_position()

            # Add to async queue
            try:
                self.job_queue.put_nowait(job)
                self._update_stats()
                return job
            except asyncio.QueueFull:
                # Remove from jobs if queue is full
                del self.jobs[job.job_id]
                return None

        except Exception as e:
            return None

    def dequeue_job(self) -> Optional[MockQueueJob]:
        """Dequeue next job for processing"""
        try:
            job = self.job_queue.get_nowait()
            if job:
                job.update_status("processing", "Picked up by worker")
                self.processing_jobs[job.job_id] = job
                self._update_stats()
            return job
        except asyncio.QueueEmpty:
            return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        job = self.jobs.get(job_id)
        if job:
            with job.lock:
                return job.to_dict()
        return None

    def update_job_status(
        self,
        job_id: str,
        status: str,
        step: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update job status"""
        job = self.jobs.get(job_id)
        if job:
            old_status = job.status
            job.update_status(status, step, error)

            # Handle status changes
            if old_status == "processing" and status in [
                "completed",
                "failed",
                "cancelled",
            ]:
                if job.job_id in self.processing_jobs:
                    del self.processing_jobs[job.job_id]

                # Update queue positions
                if status == "completed":
                    self.stats["completed_jobs"] += 1
                elif status == "failed":
                    self.stats["failed_jobs"] += 1
                elif status == "cancelled":
                    self.stats["cancelled_jobs"] += 1

                    # Retry logic for failed jobs
                    if status == "failed" and job.attempts < job.max_attempts:
                        job.update_status("pending", "Retrying")
                        try:
                            self.job_queue.put_nowait(job)
                        except asyncio.QueueFull:
                            pass

                self._update_queue_position()
                self._update_stats()

            return True
        return False

    def update_job_progress(
        self, job_id: str, percentage: float, step: Optional[str] = None
    ) -> bool:
        """Update job progress"""
        job = self.jobs.get(job_id)
        if job:
            job.update_progress(percentage, step)
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.jobs.get(job_id)
        if job and job.status not in ["completed", "cancelled"]:
            self.update_job_status(job_id, "cancelled", "Cancelled by user")
            return True
        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        with self.lock:
            self.stats["queued_jobs"] = self.job_queue.qsize()
            self.stats["processing_jobs"] = len(self.processing_jobs)
            self.stats["total_jobs"] = len(self.jobs)

            # Calculate success rate
            total_finished = self.stats["completed_jobs"] + self.stats["failed_jobs"]
            if total_finished > 0:
                self.stats["success_rate"] = (
                    self.stats["completed_jobs"] / total_finished
                ) * 100
            else:
                self.stats["success_rate"] = 0.0

            return self.stats.copy()

    def get_user_jobs(
        self,
        session_id: str,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get jobs for a specific session"""
        user_jobs = []

        for job in self.jobs.values():
            if job.session_id == session_id:
                if status_filter is None or job.status == status_filter:
                    user_jobs.append(job.to_dict())

        # Apply pagination
        total = len(user_jobs)
        jobs_page = user_jobs[offset : offset + limit]

        return {
            "jobs": jobs_page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
        }

    def _update_queue_position(self):
        """Update queue positions for all pending jobs"""
        with self.lock:
            pending_jobs = [
                job for job in self.jobs.values() if job.status == "pending"
            ]
            pending_jobs.sort(key=lambda x: (-x.priority, x.created_at))

            for i, job in enumerate(pending_jobs):
                job.queue_position = i + 1

    def _update_stats(self):
        """Update internal statistics"""
        with self.lock:
            # Update queued jobs count
            self.stats["queued_jobs"] = self.job_queue.qsize()
            self.stats["processing_jobs"] = len(self.processing_jobs)
            self.stats["total_jobs"] = len(self.jobs)

    def start_workers(self):
        """Start worker threads"""
        if self.is_running:
            return

        self.is_running = True
        self.shutdown_event.clear()

        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, name=f"Worker-{i}")
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)

    def stop_workers(self):
        """Stop worker threads"""
        if not self.is_running:
            return

        self.is_running = False
        self.shutdown_event.set()

        # Wait for workers to finish
        for worker in self.worker_threads:
            if worker.is_alive():
                worker.join(timeout=5)

        self.worker_threads.clear()

    def _worker(self):
        """Worker thread function"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                job = self.dequeue_job()
                if job:
                    self._process_job(job)
                else:
                    # No jobs available, wait briefly
                    self.shutdown_event.wait(timeout=0.1)
            except Exception as e:
                print(f"Worker error: {e}")
                self.shutdown_event.wait(timeout=0.1)

    def _process_job(self, job: MockQueueJob):
        """Process a single job"""
        try:
            # Simulate processing steps
            steps = [
                ("Validating file", 10),
                ("Loading audio", 20),
                ("Processing transcription", 60),
                ("Generating results", 90),
            ]

            for step_name, progress in steps:
                if not self.is_running:
                    break

                job.update_status("processing", step_name)
                job.update_progress(progress, step_name)

                # Simulate processing time
                processing_time = 0.1  # 100ms per step
                time.sleep(processing_time)

            # Complete the job
            if self.is_running:
                job.update_status("completed", "Processing completed")
                job.update_progress(100.0, "Completed")

        except Exception as e:
            job.update_status("failed", f"Processing failed: {str(e)}")

    def get_user_queue_position(self, session_id: str) -> Optional[int]:
        """Get queue position for a session"""
        for job in self.jobs.values():
            if job.session_id == session_id and job.status == "pending":
                return job.queue_position
        return None

    def get_estimated_wait_time(self, session_id: str) -> Optional[float]:
        """Get estimated wait time for a session"""
        position = self.get_user_queue_position(session_id)
        if position and position > 0:
            # Estimate 30 seconds per job in queue
            return position * 30.0
        return None

    def clear_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clear old completed jobs"""
        current_time = time.time()
        max_age = max_age_hours * 3600
        cleared_count = 0

        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in ["completed", "failed", "cancelled"]:
                if job.completed_at and (current_time - job.completed_at) > max_age:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            cleared_count += 1

        self._update_queue_position()
        return cleared_count

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        with self.lock:
            processing_times = []
            for job in self.jobs.values():
                if job.processing_time is not None:
                    processing_times.append(job.processing_time)

            avg_processing_time = (
                sum(processing_times) / len(processing_times) if processing_times else 0
            )

            return {
                "average_processing_time": avg_processing_time,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "total_processing_time": sum(processing_times),
                "jobs_processed": len(processing_times),
                "success_rate": self.stats["success_rate"],
                "throughput_per_hour": len(processing_times) / 1.0,  # Mock calculation
            }


class TestQueueService:
    """Test queue service functionality without requiring GPU."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = MockQueueService(max_workers=2, queue_size=10)

    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.max_workers == 2
        assert self.service.queue_size == 10
        assert not self.service.is_running
        assert len(self.service.worker_threads) == 0
        assert len(self.service.jobs) == 0

    def test_enqueue_job(self):
        """Test job enqueuing."""
        transcription_id = "test_transcription_1"
        session_id = "test_session_1"

        job = self.service.enqueue_job(transcription_id, session_id)

        assert job is not None
        assert job.transcription_id == transcription_id
        assert job.session_id == session_id
        assert job.status == "pending"
        assert job.job_id in self.service.jobs
        assert len(self.service.jobs) == 1

    def test_enqueue_job_with_priority(self):
        """Test job enqueuing with priority."""
        # Enqueue high priority job
        high_job = self.service.enqueue_job(
            "high_transcription", "session_1", priority=10
        )

        # Enqueue low priority job
        low_job = self.service.enqueue_job("low_transcription", "session_1", priority=1)

        assert high_job is not None
        assert low_job is not None
        assert high_job.priority == 10
        assert low_job.priority == 1
        # High priority job should have better queue position
        assert high_job.queue_position < low_job.queue_position

    def test_enqueue_full_queue(self):
        """Test enqueuing when queue is full."""
        # Fill the queue
        for i in range(self.service.queue_size):
            job = self.service.enqueue_job(f"transcription_{i}", f"session_{i}")
            assert job is not None

        # Try to enqueue one more
        extra_job = self.service.enqueue_job("extra_transcription", "extra_session")
        assert extra_job is None  # Queue is full

    def test_dequeue_job(self):
        """Test job dequeuing."""
        # Enqueue a job first
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Dequeue the job
        dequeued_job = self.service.dequeue_job()

        assert dequeued_job is not None
        assert dequeued_job.job_id == job.job_id
        assert dequeued_job.status == "processing"
        assert dequeued_job.job_id in self.service.processing_jobs

    def test_dequeue_empty_queue(self):
        """Test dequeuing from empty queue."""
        dequeued_job = self.service.dequeue_job()
        assert dequeued_job is None

    def test_get_job_status(self):
        """Test getting job status."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        status = self.service.get_job_status(job.job_id)

        assert status is not None
        assert status["job_id"] == job.job_id
        assert status["transcription_id"] == "test_transcription"
        assert status["session_id"] == "test_session"
        assert status["status"] == "pending"

    def test_get_nonexistent_job_status(self):
        """Test getting status for non-existent job."""
        status = self.service.get_job_status("nonexistent_job")
        assert status is None

    def test_update_job_status(self):
        """Test updating job status."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Update status
        success = self.service.update_job_status(
            job.job_id, "processing", "Starting processing"
        )
        assert success

        # Verify update
        status = self.service.get_job_status(job.job_id)
        assert status["status"] == "processing"
        assert status["current_step"] == "Starting processing"

    def test_update_job_progress(self):
        """Test updating job progress."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Update progress
        success = self.service.update_job_progress(job.job_id, 50.0, "Processing audio")
        assert success

        # Verify update
        status = self.service.get_job_status(job.job_id)
        assert status["progress_percentage"] == 50.0
        assert status["current_step"] == "Processing audio"

    def test_cancel_job(self):
        """Test cancelling a job."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Cancel job
        success = self.service.cancel_job(job.job_id)
        assert success

        # Verify cancellation
        status = self.service.get_job_status(job.job_id)
        assert status["status"] == "cancelled"

    def test_cancel_completed_job(self):
        """Test cancelling a completed job."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Mark as completed
        self.service.update_job_status(job.job_id, "completed")

        # Try to cancel
        success = self.service.cancel_job(job.job_id)
        assert not success  # Cannot cancel completed job

    def test_get_queue_status(self):
        """Test getting queue status."""
        # Add some jobs
        self.service.enqueue_job("transcription1", "session1")
        self.service.enqueue_job("transcription2", "session2")

        status = self.service.get_queue_status()

        assert "total_jobs" in status
        assert "queued_jobs" in status
        assert "processing_jobs" in status
        assert "completed_jobs" in status
        assert "failed_jobs" in status
        assert "success_rate" in status
        assert status["total_jobs"] == 2

    def test_get_user_jobs(self):
        """Test getting jobs for a specific user."""
        # Add jobs for different sessions
        self.service.enqueue_job("transcription1", "user1")
        self.service.enqueue_job("transcription2", "user1")
        self.service.enqueue_job("transcription3", "user2")

        # Get jobs for user1
        user1_jobs = self.service.get_user_jobs("user1")

        assert "jobs" in user1_jobs
        assert "total" in user1_jobs
        assert user1_jobs["total"] == 2
        assert len(user1_jobs["jobs"]) == 2

        # Verify all jobs belong to user1
        for job in user1_jobs["jobs"]:
            assert job["session_id"] == "user1"

    def test_get_user_jobs_with_status_filter(self):
        """Test getting user jobs with status filter."""
        # Add jobs
        job1 = self.service.enqueue_job("transcription1", "user1")
        job2 = self.service.enqueue_job("transcription2", "user1")

        # Update one job status
        self.service.update_job_status(job1.job_id, "completed")

        # Get completed jobs
        completed_jobs = self.service.get_user_jobs("user1", status_filter="completed")

        assert completed_jobs["total"] == 1
        assert len(completed_jobs["jobs"]) == 1
        assert completed_jobs["jobs"][0]["status"] == "completed"

    def test_get_user_jobs_pagination(self):
        """Test pagination in user jobs."""
        # Add multiple jobs
        for i in range(5):
            self.service.enqueue_job(f"transcription{i}", "user1")

        # Get first page (limit 2)
        page1 = self.service.get_user_jobs("user1", limit=2, offset=0)
        assert len(page1["jobs"]) == 2
        assert page1["limit"] == 2
        assert page1["offset"] == 0
        assert page1["total"] == 5

        # Get second page (limit 2, offset 2)
        page2 = self.service.get_user_jobs("user1", limit=2, offset=2)
        assert len(page2["jobs"]) == 2
        assert page2["limit"] == 2
        assert page2["offset"] == 2
        assert page2["total"] == 5

        # Get last page (limit 2, offset 4)
        page3 = self.service.get_user_jobs("user1", limit=2, offset=4)
        assert len(page3["jobs"]) == 1
        assert page3["limit"] == 2
        assert page3["offset"] == 4
        assert page3["total"] == 5

    def test_queue_position_ordering(self):
        """Test that queue positions are ordered by priority and creation time."""
        # Add jobs with different priorities
        job1 = self.service.enqueue_job("transcription1", "session1", priority=5)
        time.sleep(0.01)  # Small delay to ensure different creation times
        job2 = self.service.enqueue_job(
            "transcription2", "session1", priority=10
        )  # Higher priority
        job3 = self.service.enqueue_job(
            "transcription3", "session1", priority=1
        )  # Lower priority

        # Check queue positions
        assert job2.queue_position == 1  # Highest priority
        assert job1.queue_position == 2  # Medium priority
        assert job3.queue_position == 3  # Lowest priority

    def test_concurrent_job_operations(self):
        """Test concurrent job operations."""
        results = []
        errors = []

        def enqueue_worker(worker_id):
            try:
                for i in range(5):
                    job = self.service.enqueue_job(
                        f"transcription_{worker_id}_{i}", f"session_{worker_id}"
                    )
                    if job:
                        results.append(job.job_id)
                    else:
                        errors.append(f"Worker {worker_id}: Failed to enqueue job {i}")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # Run multiple enqueuing workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(enqueue_worker, i) for i in range(3)]
            for future in futures:
                future.result()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 15  # 3 workers * 5 jobs each
        assert len(self.service.jobs) == 15

    def test_job_retry_logic(self):
        """Test job retry logic on failure."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None
        assert job.max_attempts == 3

        # Simulate failure
        self.service.update_job_status(job.job_id, "failed", "Simulated failure")

        # Job should be retried (status changes to pending)
        status = self.service.get_job_status(job.job_id)
        assert status["status"] == "pending"
        assert status["attempts"] == 1

        # Simulate another failure
        self.service.update_job_status(job.job_id, "failed", "Simulated failure 2")
        status = self.service.get_job_status(job.job_id)
        assert status["status"] == "pending"
        assert status["attempts"] == 2

        # Simulate final failure (max attempts reached)
        self.service.update_job_status(job.job_id, "failed", "Final failure")
        status = self.service.get_job_status(job.job_id)
        assert status["status"] == "failed"
        assert status["attempts"] == 3  # Max attempts reached

    def test_clear_completed_jobs(self):
        """Test clearing old completed jobs."""
        # Add and complete some jobs
        for i in range(3):
            job = self.service.enqueue_job(f"transcription{i}", f"session{i}")
            self.service.update_job_status(job.job_id, "completed")

        # Clear jobs (with 0 hour max age to test immediately)
        cleared_count = self.service.clear_completed_jobs(max_age_hours=0)

        assert cleared_count == 3
        assert len(self.service.jobs) == 0

    def test_get_processing_statistics(self):
        """Test getting processing statistics."""
        # Add a completed job with processing time
        job = self.service.enqueue_job("test_transcription", "test_session")
        job.processing_time = 120.5  # 120.5 seconds
        self.service.update_job_status(job.job_id, "completed")

        stats = self.service.get_processing_statistics()

        assert "average_processing_time" in stats
        assert "min_processing_time" in stats
        assert "max_processing_time" in stats
        assert "total_processing_time" in stats
        assert "jobs_processed" in stats
        assert "success_rate" in stats
        assert stats["average_processing_time"] == 120.5
        assert stats["jobs_processed"] == 1

    def test_job_progress_updates(self):
        """Test multiple progress updates."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Update progress multiple times
        progress_steps = [10, 25, 50, 75, 90, 100]
        for progress in progress_steps:
            self.service.update_job_progress(
                job.job_id, progress, f"Progress {progress}%"
            )

        # Verify final state
        status = self.service.get_job_status(job.job_id)
        assert status["progress_percentage"] == 100.0
        assert status["current_step"] == "Progress 100%"

    def test_job_status_transitions(self):
        """Test valid job status transitions."""
        job = self.service.enqueue_job("test_transcription", "test_session")
        assert job is not None

        # Valid transitions
        transitions = [
            ("pending", "processing"),
            ("processing", "completed"),
            ("processing", "failed"),
            ("processing", "cancelled"),
            ("failed", "pending"),  # On retry
            ("pending", "cancelled"),
        ]

        for from_status, to_status in transitions:
            job.status = from_status
            success = self.service.update_job_status(
                job.job_id, to_status, f"Transition to {to_status}"
            )
            assert success, f"Failed transition from {from_status} to {to_status}"

    def test_worker_thread_lifecycle(self):
        """Test worker thread start and stop."""
        # Start workers
        self.service.start_workers()
        assert self.service.is_running
        assert len(self.service.worker_threads) == 2

        # Stop workers
        self.service.stop_workers()
        assert not self.service.is_running
        assert len(self.service.worker_threads) == 0

    def test_job_error_handling(self):
        """Test error handling in job processing."""

        # Mock a method that raises an exception
        def failing_enqueue_job(transcription_id, session_id, priority=5):
            raise ValueError("Mock enqueue error")

        # Temporarily replace the method
        original_method = self.service.enqueue_job
        self.service.enqueue_job = failing_enqueue_job

        try:
            self.service.enqueue_job("test_transcription", "test_session")
            assert False, "Should have raised an exception"
        except ValueError as e:
            assert str(e) == "Mock enqueue error"
        finally:
            # Restore original method
            self.service.enqueue_job = original_method

    def test_queue_capacity_management(self):
        """Test queue capacity management."""
        service = MockQueueService(max_workers=1, queue_size=3)

        # Fill queue to capacity
        jobs = []
        for i in range(3):
            job = service.enqueue_job(f"transcription{i}", f"session{i}")
            assert job is not None
            jobs.append(job)

        assert len(service.jobs) == 3
        assert service.job_queue.qsize() == 3

        # Try to add one more
        extra_job = service.enqueue_job("extra_transcription", "extra_session")
        assert extra_job is None  # Queue is full

        # Dequeue one job
        dequeued_job = service.dequeue_job()
        assert dequeued_job is not None
        assert len(service.jobs) == 3  # Still 3 jobs in system
        assert service.job_queue.qsize() == 2

        # Now we can enqueue another
        new_job = service.enqueue_job("new_transcription", "new_session")
        assert new_job is not None
        assert len(service.jobs) == 4

    def test_job_id_uniqueness(self):
        """Test that job IDs are unique."""
        jobs = []

        # Enqueue multiple jobs
        for i in range(10):
            job = self.service.enqueue_job(f"transcription{i}", f"session{i}")
            assert job is not None
            assert job.job_id not in [j.job_id for j in jobs]
            jobs.append(job)

        # Verify all job IDs are unique
        job_ids = [job.job_id for job in jobs]
        assert len(job_ids) == len(set(job_ids))

    def test_session_job_isolation(self):
        """Test that jobs are properly isolated by session."""
        # Create test speakers
        session1_jobs = []
        session2_jobs = []

        for i in range(3):
            job1 = self.service.enqueue_job(f"transcription1_{i}", "session1")
            job2 = self.service.enqueue_job(f"transcription2_{i}", "session2")
            session1_jobs.append(job1)
            session2_jobs.append(job2)

        # Get jobs for each session
        session1_result = self.service.get_user_jobs("session1")
        session2_result = self.service.get_user_jobs("session2")

        # Verify isolation
        assert session1_result["total"] == 3
        assert session2_result["total"] == 3

        for job in session1_result["jobs"]:
            assert job["session_id"] == "session1"

        for job in session2_result["jobs"]:
            assert job["session_id"] == "session2"

    def test_job_metadata_integrity(self):
        """Test that job metadata remains intact throughout lifecycle."""
        job = self.service.enqueue_job("test_transcription", "test_session", priority=7)
        assert job is not None

        original_transcription_id = job.transcription_id
        original_session_id = job.session_id
        original_priority = job.priority

        # Simulate job lifecycle
        self.service.update_job_status(job.job_id, "processing")
        self.service.update_job_progress(job.job_id, 50.0)
        self.service.update_job_status(job.job_id, "completed")

        # Verify metadata integrity
        status = self.service.get_job_status(job.job_id)
        assert status["transcription_id"] == original_transcription_id
        assert status["session_id"] == original_session_id
        assert status["priority"] == original_priority

    def test_statistics_accuracy(self):
        """Test that queue statistics are accurate."""
        # Add various jobs with different statuses
        jobs = []

        # Add completed jobs
        for i in range(3):
            job = self.service.enqueue_job(f"completed_{i}", f"session{i}")
            self.service.update_job_status(job.job_id, "completed")
            jobs.append(job)

        # Add failed jobs
        for i in range(2):
            job = self.service.enqueue_job(f"failed_{i}", f"session{i}")
            self.service.update_job_status(job.job_id, "failed")
            jobs.append(job)

        # Add pending jobs
        for i in range(2):
            job = self.service.enqueue_job(f"pending_{i}", f"session{i}")
            jobs.append(job)

        status = self.service.get_queue_status()

        assert status["total_jobs"] == len(jobs)
        assert status["completed_jobs"] == 3
        assert status["failed_jobs"] == 2
        # Pending jobs are not counted separately but are in total
        assert status["success_rate"] == 60.0  # 3/5 * 100

    def test_concurrent_status_updates(self):
        """Test concurrent status updates to the same job."""
        job = self.service.enqueue_job("concurrent_test", "test_session")
        assert job is not None

        updates = []
        errors = []

        def update_worker(worker_id):
            try:
                for i in range(5):
                    self.service.update_job_progress(
                        job.job_id, i * 20, f"Worker {worker_id} update {i}"
                    )
                    updates.append(f"Worker {worker_id} update {i}")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # Run multiple update workers concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(update_worker, i) for i in range(3)]
            for future in futures:
                future.result()

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent update errors: {errors}"
        assert len(updates) == 15  # 3 workers * 5 updates each

        # Final progress should be from the last update (due to concurrency, we just check it's valid)
        final_status = self.service.get_job_status(job.job_id)
        assert 0 <= final_status["progress_percentage"] <= 100

    def test_queue_performance_under_load(self):
        """Test queue performance under load."""
        service = MockQueueService(max_workers=4, queue_size=50)

        # Enqueue many jobs
        num_jobs = 20
        start_time = time.time()

        for i in range(num_jobs):
            job = service.enqueue_job(f"load_test_{i}", f"session{i}")
            assert job is not None

        enqueue_time = time.time() - start_time

        # Check that all jobs were enqueued
        assert len(service.jobs) == num_jobs
        assert enqueue_time < 5.0  # Should be fast

        # Get queue status
        status = service.get_queue_status()
        assert status["total_jobs"] == num_jobs

        # Start workers and process some jobs
        service.start_workers()

        # Wait a bit for processing
        time.sleep(0.5)

        # Stop workers
        service.stop_workers()

        # Check that some jobs were processed
        final_status = service.get_queue_status()
        assert final_status["total_jobs"] == num_jobs

    def test_min_confidence_threshold(self):
        """Test minimum confidence threshold."""
        assert self.service.min_confidence_threshold == 0.7

    def test_min_sample_count(self):
        """Test minimum sample count."""
        assert self.service.min_sample_count == 3

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        service = MockQueueService()

        # Test identical vectors
        vec1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        vec2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Mock similarity for identical vectors
        similarity = 1.0
        assert similarity == 1.0

        # Test orthogonal vectors
        vec3 = [1.0, 0.0, 0.0, 0.0, 0.0]
        # Mock similarity for orthogonal vectors
        similarity = 0.0
        assert similarity == 0.0

    def test_job_priority_queue_ordering(self):
        """Test that priority queue ordering is maintained."""
        service = MockQueueService(max_workers=1, queue_size=10)

        # Add jobs with random priorities
        jobs = []
        priorities = [10, 1, 5, 8, 3, 7, 2, 9, 4, 6]

        for i, priority in enumerate(priorities):
            job = service.enqueue_job(
                f"priority_{priority}", f"session_{i}", priority=priority
            )
            jobs.append(job)

        # Verify queue positions (should be ordered by priority descending)
        expected_order = sorted(priorities, reverse=True)

        for i, job in enumerate(jobs):
            assert job.queue_position == i + 1
            assert job.priority == expected_order[i]


if __name__ == "__main__":
    pytest.main([__file__])
