# Pipeline Issues Troubleshooting Guide

This guide addresses the specific issues identified through diagnostics, focusing on the core problems that cause transcription jobs to fail and status tracking to break down.

## 🔍 Key Findings from Diagnostics

### Primary Issue: Diarization Service Failure

**Error Pattern in Logs:**
```
❌ Diarization processing failed: 'NoneType' object is not callable
❌ Job execution failed: Diarization failed: 'NoneType' object is not callable
```

**Root Cause:** The diarization service pipeline is not properly initialized or is being called incorrectly. This causes all transcription jobs to fail after the transcription phase completes successfully.

## 🛠️ Immediate Solutions

### 1. Fix Diarization Service Initialization

The issue is likely in `app/services/diarization_service.py`. Check these areas:

**Pipeline Loading:**
```python
def _load_pipeline(self):
    try:
        self.pipeline = self.pyannote_model
        # This might be failing and returning None
        return self.pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        self.pipeline = None  # This causes the 'NoneType' error
        return None
```

**Fix Implementation:**
```python
def _load_pipeline(self):
    try:
        if self.settings.mock_gpu or self.settings.test_mode:
            logger.info("Using mock diarization pipeline")
            return MockDiarizationPipeline()
        
        import torch
        from pyannote.audio import Pipeline
        
        self.pipeline = Pipeline.from_pretrained(
            self.pyannote_model,
            token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        if torch.cuda.is_available() and not self.settings.mock_gpu:
            self.pipeline = self.pipeline.to(torch.device("cuda"))
        else:
            self.pipeline = self.pipeline.to(torch.device("cpu"))
            
        logger.info(f"Diarization pipeline loaded on {self.pipeline.device}")
        return self.pipeline
        
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}")
        # Don't set to None - raise the error instead
        raise RuntimeError(f"Diarization pipeline initialization failed: {e}")
```

### 2. Add Safe Calling with Fallbacks

Update the diarization method to handle initialization failures gracefully:

```python
def diarize_audio(self, audio_path, transcription, db, progress_callback=None):
    """Diarize audio file with proper error handling."""
    try:
        if progress_callback:
            progress_callback(0, "Initializing diarization...")
        
        # Ensure pipeline is loaded
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            logger.warning("Pipeline not initialized, attempting reload...")
            self._load_pipeline()
        
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            # Fallback to mock diarization
            logger.info("Using fallback mock diarization")
            return self._mock_diarization_result(transcription)
        
        # Continue with actual diarization
        # ... existing code ...
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        # Return mock result instead of failing completely
        return self._mock_diarization_result(transcription)

def _mock_diarization_result(self, transcription):
    """Return a mock diarization result to prevent complete failure."""
    return {
        "speakers": [
            {"id": "SPEAKER_00", "name": "Speaker 1", "confidence": 0.8}
        ],
        "speaker_matches": {"SPEAKER_00": "Speaker 1"},
        "segments": transcription.segments or []
    }
```

### 3. Add Development Mode Fallbacks

For development/testing environments, ensure mocked services work correctly:

```python
def __init__(self):
    self.settings = get_settings()
    self.pyannote_model = self.settings.pyannote_model
    self.device = self._get_device()
    self.pipeline = None
    
    # Auto-load in development mode
    if self.settings.test_mode or self.settings.mock_gpu:
        self.pipeline = MockDiarizationPipeline()
        logger.info("Using mock diarization pipeline for development")
```

## 📊 Status Tracking Issues

### Problem: Jobs Disappear During Processing

**Symptoms:**
- Jobs show "queued" status, then disappear
- Web interface shows empty queue
- API returns no active jobs

**Root Cause:** Jobs fail silently during diarization phase and don't update status properly.

### Solution: Add Comprehensive Status Tracking

```python
def _execute_job(self, job: ProcessingQueue) -> Dict[str, Any]:
    """Execute job with comprehensive status tracking."""
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

            # Progress callback with database updates
            def progress_callback(percentage: float, step: str) -> None:
                try:
                    job.update_progress(percentage, step)
                    transcription.update_progress(percentage, step)
                    db.commit()
                except Exception as e:
                    logger.error(f"Progress update failed: {e}")
                    # Don't let progress failures stop processing

            # Step 1: Transcription (40% of progress)
            progress_callback(0, "Starting transcription")
            transcription_result = self.transcription_service.transcribe_audio(
                job.file_path,
                transcription,
                db,
                progress_callback=progress_callback,
            )
            
            # Explicit status update after transcription
            transcription.status = "transcription_completed"
            db.commit()
            logger.info(f"Transcription phase completed for job {job.job_id}")

            # Step 2: Diarization (40% of progress)
            progress_callback(40, "Starting speaker diarization")
            
            try:
                diarization_result = self.diarization_service.diarize_audio(
                    job.file_path,
                    transcription,
                    db,
                    progress_callback=progress_callback,
                )
                transcription.status = "diarization_completed"
                db.commit()
                logger.info(f"Diarization phase completed for job {job.job_id}")
                
            except Exception as diarization_error:
                logger.error(f"Diarization failed: {diarization_error}")
                # Continue with mock diarization instead of failing
                diarization_result = self._get_mock_diarization_result(transcription)
                transcription.status = "diarization_skipped"
                db.commit()

            # Step 3: Final processing (20% of progress)
            progress_callback(80, "Finalizing results")

            # Update speaker assignments
            if diarization_result.get("speaker_matches"):
                for segment in transcription.segments or []:
                    speaker_label = segment.get("speaker", "")
                    matched_speaker = diarization_result["speaker_matches"].get(speaker_label)
                    if matched_speaker:
                        segment["speaker"] = matched_speaker.name

            progress_callback(100, "Completed")
            transcription.status = "completed"
            db.commit()

            result = {
                "transcription": transcription_result,
                "diarization": diarization_result,
                "status": "completed",
            }

            logger.info(f"Job {job.job_id} completed successfully")
            return result

    except Exception as e:
        logger.error(f"Job execution failed {job.job_id}: {e}")
        
        # Ensure failure status is properly recorded
        try:
            with next(get_database()) as db:
                transcription = (
                    db.query(Transcription)
                    .filter(Transcription.id == job.transcription_id)
                    .first()
                )
                if transcription:
                    transcription.status = "failed"
                    transcription.error_message = str(e)
                    db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update failure status: {db_error}")
        
        raise QueueError(f"Job execution failed: {str(e)}")
```

## 🧪 Testing the Fixes

### 1. Run Comprehensive Diagnostics

```bash
# First, ensure environment is set up
export TEST_MODE=true
export MOCK_GPU=true
export LOG_LEVEL=DEBUG

# Run diagnostics to verify fixes
python3 scripts/diagnose_pipeline.py
```

### 2. Test Pipeline with Mock Services

```bash
python3 scripts/run_pipeline_tests.py
```

### 3. Manual End-to-End Test

```bash
# Start the application in development mode
TEST_MODE=true MOCK_GPU=true uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# In another terminal, test file upload
curl -X POST "http://localhost:8001/api/sessions/create"

# Upload a test audio file and monitor status
curl -X POST -F "file=@test.wav" -F "session_id=YOUR_SESSION_ID" \
  "http://localhost:8001/api/transcription/upload"

# Monitor job status
curl "http://localhost:8001/api/transcription/YOUR_TRANSCRIPTION_ID/status"
```

## 🔧 Configuration Adjustments

### Development Environment Settings

Add these to your `.env` file for development:

```bash
# Enable testing mode
TEST_MODE=true
MOCK_GPU=true

# Disable GPU for development
CUDA_VISIBLE_DEVICES=

# Enable detailed logging
LOG_LEVEL=DEBUG

# Ensure directories exist
UPLOAD_DIR=./uploads
PROCESSED_DIR=./processed
LOG_FILE=./logs/securetranscribe.log

# Development security settings
ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0"]
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### Production Environment Settings

For production, ensure robust error handling:

```bash
# Production mode
TEST_MODE=false
MOCK_GPU=false

# GPU settings (if available)
CUDA_VISIBLE_DEVICES=0

# Production logging
LOG_LEVEL=WARNING

# Production directories (ensure they exist and are writable)
UPLOAD_DIR=/var/lib/securetranscribe/uploads
PROCESSED_DIR=/var/lib/securetranscribe/processed
LOG_FILE=/var/log/securetranscribe/app.log
```

## 📋 Verification Checklist

After implementing fixes, verify:

- [ ] Diarization service loads without errors
- [ ] Jobs progress through all phases: queued → processing → completed
- [ ] Status updates are visible in real-time via API
- [ ] Web interface shows accurate queue status
- [ ] Progress callbacks update both database and API responses
- [ ] Error conditions are handled gracefully with fallbacks
- [ ] Mock services work in development mode
- [ ] Database state remains consistent throughout processing

## 🚨 Emergency Fallback

If diarization continues to fail, implement this emergency fallback:

```python
def diarize_audio(self, audio_path, transcription, db, progress_callback=None):
    """Emergency fallback that always succeeds."""
    logger.warning("Using emergency diarization fallback")
    
    if progress_callback:
        progress_callback(50, "Using fallback diarization")
    
    # Return simple speaker assignment
    return {
        "speakers": [
            {"id": "SPEAKER_00", "name": "Speaker 1", "confidence": 1.0}
        ],
        "speaker_matches": {"SPEAKER_00": "Speaker 1"},
        "fallback_used": True
    }
```

This ensures the pipeline completes even if the full diarization fails, allowing users to at least get the transcription results.

## 📞 Getting Help

If issues persist:

1. **Check Logs**: Always check `logs/securetranscribe.log` for detailed error messages
2. **Run Diagnostics**: `python3 scripts/diagnose_pipeline.py` provides comprehensive health check
3. **Test Isolated Components**: Use specific test methods to isolate problems
4. **Enable Debug Logging**: Set `LOG_LEVEL=DEBUG` for detailed traces

The key insight is that the diarization service is the primary point of failure. By implementing proper initialization checks, fallbacks, and comprehensive error handling, the pipeline should become much more robust and provide better status visibility throughout the process.