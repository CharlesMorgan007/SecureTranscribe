# SecureTranscribe Testing Guide

This guide provides comprehensive testing tools to verify that your transcription and diarization processing pipeline works correctly, especially focusing on the queue management and status tracking issues you've experienced.

## Quick Start

### 1. Run Comprehensive Tests
```bash
cd SecureTranscribe
python3 scripts/run_tests.py
```

### 2. Run Pipeline Diagnostics (Recommended First Step)
```bash
python3 scripts/diagnose_pipeline.py
```

### 3. Quick Test for Fast Feedback
```bash
python3 scripts/run_tests.py --quick
```

## Problem Diagnosis

The tests are specifically designed to help identify issues where:

- Jobs disappear from the queue during processing
- Web interface shows empty queue when jobs are processing
- Transcription never completes
- Status updates aren't propagated correctly
- Progress tracking isn't working

## What the Tests Check

### 1. Environment & Configuration
- ✅ Required environment variables are set correctly
- ✅ Directory permissions and structure
- ✅ Database connectivity
- ✅ Import dependencies

### 2. Queue Management
- ✅ Jobs are properly queued
- ✅ Status transitions work: queued → processing → completed
- ✅ Queue statistics are accurate
- ✅ Worker pool status
- ✅ Database consistency during processing

### 3. API Endpoints
- ✅ All endpoints respond correctly
- ✅ Session management works
- ✅ File upload process
- ✅ Status reporting endpoints
- ✅ Error handling

### 4. Processing Pipeline
- ✅ File upload to transcription start
- ✅ Progress callback functionality
- ✅ Mock GPU services (works without CUDA)
- ✅ Error handling and recovery
- ✅ Concurrent job processing

### 5. Status Tracking
- ✅ Real-time status updates via API
- ✅ Progress percentage tracking
- ✅ Database state consistency
- ✅ Queue position calculations
- ✅ Wait time estimations

## Running Specific Tests

### Diagnose Specific Issues

**If jobs disappear from queue:**
```bash
python3 scripts/diagnose_pipeline.py
```
Focus on:
- Queue service status
- Database consistency checks
- Recent job entries

**If web interface shows empty queue:**
```bash
python3 scripts/run_tests.py --specific tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_queue_status_tracking
```

**If transcription never completes:**
```bash
python3 scripts/run_pipeline_tests.py
```
This runs with mocked GPU services to isolate the issue.

**If progress tracking isn't working:**
```bash
python3 scripts/run_tests.py --specific tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_progress_callback_functionality
```

## Test Environment

The tests automatically configure a safe testing environment:
```bash
TEST_MODE=true              # Enable test-specific behaviors
MOCK_GPU=true              # Use mocked GPU services (no CUDA needed)
CUDA_VISIBLE_DEVICES=        # Disable GPU detection
LOG_LEVEL=DEBUG             # Detailed logging for debugging
ALLOWED_HOSTS=["localhost", "127.0.0.1"]
CORS_ORIGINS=["http://localhost:3000"]
```

## Expected Test Results

### Successful Test Run
```
🚀 SecureTranscribe Master Test Runner
============================================================
📅 Started: 2024-01-15 10:30:00
📁 Project Root: /path/to/SecureTranscribe
============================================================

🧪 Checking Dependencies...
✅ fastapi
✅ sqlalchemy
✅ pydantic
✅ All required dependencies are available

🧪 Running: Pipeline Diagnostics
💻 Command: python3 scripts/diagnose_pipeline.py
============================================================
🔍 Environment Check
✅ DATABASE_URL: sqlite:///./securetranscribe.db
✅ UPLOAD_DIR: ./uploads
✅ LOG_LEVEL: DEBUG
✅ All required environment variables set

🔍 Import Check
✅ main - FastAPI application
✅ app.core.config - Configuration module
✅ All modules imported successfully

🔍 Database Check
✅ Database connection successful
✅ processing_queue: 0 records
✅ Database is accessible

🔍 Queue Service Check
✅ Queue service instantiated successfully
📊 Total jobs: 0
📊 Queued jobs: 0
📊 Processing jobs: 0
✅ Queue service is running

✅ PASS: Pipeline Diagnostics

🧪 Running: Integration Tests
💻 Command: python3 scripts/run_pipeline_tests.py
🚀 Starting SecureTranscribe Pipeline Tests
✅ Created test audio file: /tmp/tmpXXXXXX/test_audio.wav
✅ All modules imported successfully
✅ Health check passed
✅ Session created: abc-123-def-456
✅ File uploaded: 789
✅ Transcription started successfully
  📊 Status: processing, Progress: 25%, Step: Loading audio...
  📊 Status: processing, Progress: 50%, Step: Transcribing...
  📊 Status: processing, Progress: 75%, Step: Finalizing results...
✅ Transcription completed successfully
✅ Queue status tracking is working correctly
============================================================
📊 TEST SUMMARY
============================================================
✅ PASS: Module Imports
✅ PASS: Basic Functionality
✅ PASS: File Upload
✅ PASS: Mock Processing
✅ PASS: Progress Updates
✅ PASS: Queue Status Tracking
--------------------------------------------------
Total: 6/6 tests passed
🎉 All tests passed! Your processing pipeline is working correctly.
✅ PASS: Integration Tests

============================================================
📊 FINAL TEST REPORT
============================================================
📅 Completed: 2024-01-15 10:32:15
⏱️  Duration: 135.42 seconds
📈 Results: 4/4 test suites passed
  ✅ PASS: dependencies
  ✅ PASS: environment_setup
  ✅ PASS: diagnostics
  ✅ PASS: integration_tests
🎯 Overall Status: PASSED

🎉 All tests passed! Your SecureTranscribe pipeline is working correctly.
```

## Troubleshooting Failed Tests

### Common Issues and Solutions

**Import Errors:**
```
❌ Import failed: No module named 'app.main'
```
- Ensure you're in project root directory
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**Permission Errors:**
```
❌ ./uploads does not exist
```
- Create directories: `mkdir -p uploads processed logs`
- Check permissions: `chmod 755 uploads processed logs`

**Database Issues:**
```
❌ Database connection failed
```
- Check file permissions: `ls -la securetranscribe.db`
- Ensure directory is writable: `chmod 755 .`

**GPU/CUDA Issues:**
- Tests automatically use `MOCK_GPU=true`
- No CUDA required for testing
- For production GPU issues, check GPU service configuration

**API Endpoint Failures:**
```
❌ GET /api/queue/status: HTTP 500
```
- Check application logs: `tail -f logs/securetranscribe.log`
- Ensure application is running: `uvicorn app.main:app --reload`

## Test Scripts Overview

### `scripts/run_tests.py` - Master Test Runner
Main entry point for all testing. Use this for comprehensive testing.

**Options:**
- `--quick` - Fast test suite for development
- `--diagnostics-only` - Run only diagnostic checks
- `--specific <path>` - Run specific test file/method
- `--skip-diagnostics` - Skip diagnostic checks
- `--skip-unit` - Skip unit tests

### `scripts/diagnose_pipeline.py` - Pipeline Diagnostics
Best first step when troubleshooting issues. Checks:
- Environment configuration
- Module imports
- Database connectivity
- Queue service status
- API functionality
- File upload workflow
- GPU service configuration
- Recent log analysis

### `scripts/run_pipeline_tests.py` - Integration Tests
Standalone integration tests with mocked GPU services. Tests:
- Complete transcription workflow
- Queue status tracking
- Progress callbacks
- Error handling
- Database consistency
- Concurrent processing

## Development Workflow Integration

### Before Committing Changes
```bash
python3 scripts/run_tests.py --quick
```

### Before Merging to Main
```bash
python3 scripts/run_tests.py
```

### When Investigating Production Issues
```bash
python3 scripts/diagnose_pipeline.py
```

### During Development
```bash
# Test specific functionality
python3 scripts/run_tests.py --specific tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_file_upload_and_initial_status

# Enable debug logging
LOG_LEVEL=DEBUG python3 scripts/diagnose_pipeline.py
```

## Advanced Usage

### Testing with Real GPU (if available)
```bash
unset MOCK_GPU
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_tests.py --skip-diagnostics
```

### Running Individual Test Methods
```bash
python3 -m pytest tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_concurrent_job_processing -v -s
```

### Continuous Integration
All tests are designed for CI/CD:
```yaml
# GitHub Actions example
- name: Run Pipeline Tests
  run: python3 scripts/run_tests.py
```

## Interpreting Test Output

### Status Indicators
- ✅ **PASS** - Component working correctly
- ❌ **FAIL** - Issue found, check detailed output
- ⚠️ **WARNING** - Non-critical issue
- ℹ️ **INFO** - Informational message

### Key Sections to Watch
- **Database consistency** - Should show matching counts across tables
- **Queue transitions** - Jobs should move through states correctly
- **Progress updates** - Should show real-time percentage updates
- **API responses** - All endpoints should return 200 OK
- **Error handling** - Failed operations should report proper error messages

## Performance Considerations

- Tests use small audio files (1-2 seconds) for speed
- Mocked services run quickly without GPU overhead
- Database operations use optimized queries
- Tests have appropriate timeouts to prevent hanging
- Parallel test execution supported where possible

## Security Notes

- Tests use temporary sessions with no real user data
- File uploads isolated to temporary directories
- Database operations are automatically cleaned up
- No external network calls during tests
- Mocked services prevent access to real ML models

## Getting Help

### After Running Diagnostics
1. Look for ❌ symbols in output
2. Check "Recommendations" section at the end
3. Review specific error messages provided

### For Persistent Issues
1. Run with debug logging: `LOG_LEVEL=DEBUG python3 scripts/diagnose_pipeline.py`
2. Check application logs: `tail -f logs/securetranscribe.log`
3. Run specific failing tests with verbose output: `python3 -m pytest -v -s`

## Contributing Tests

When adding new tests:
1. Use descriptive names indicating what's being tested
2. Include proper cleanup to prevent pollution
3. Mock external dependencies for environment independence
4. Test both success and failure scenarios
5. Verify database state consistency
6. Add progress tracking verification for status-related tests

This comprehensive testing suite should help identify and resolve the status tracking issues you're experiencing with the transcription pipeline, providing clear feedback about where the process might be failing and verifying that all components work together correctly.