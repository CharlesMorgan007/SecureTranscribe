# Testing Suite for SecureTranscribe

This document describes the comprehensive testing suite designed to verify the transcription and diarization processing pipeline works correctly, especially focusing on queue management and status tracking issues.

## Overview

The testing suite is designed to run on development machines without CUDA/GPU requirements by using mocked services. It helps identify where the transcription process might be failing and verifies that status tracking works correctly throughout the entire pipeline.

## Quick Start

### Run All Tests
```bash
cd SecureTranscribe
python scripts/run_tests.py
```

### Quick Test Suite (Fast Feedback)
```bash
python scripts/run_tests.py --quick
```

### Run Diagnostics Only
```bash
python scripts/run_tests.py --diagnostics-only
```

### Run Specific Test
```bash
python scripts/run_tests.py --specific tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_file_upload_and_initial_status
```

## Test Components

### 1. Pipeline Diagnostics (`scripts/diagnose_pipeline.py`)

Comprehensive diagnostic tool that checks:
- Environment variables and configuration
- Module imports and dependencies
- Database connectivity and state
- Queue service status
- API endpoint functionality
- File upload workflow
- GPU service configuration
- Recent log entries for errors

**Usage:**
```bash
python scripts/diagnose_pipeline.py
```

### 2. Integration Tests (`scripts/run_pipeline_tests.py`)

Standalone integration test runner that verifies:
- File upload and initial status tracking
- Transcription processing with mocked GPU services
- Queue status updates during processing
- Progress callback functionality
- Error handling and status propagation
- Database state consistency
- Concurrent job processing

**Usage:**
```bash
python scripts/run_pipeline_tests.py
```

### 3. Unit Tests (`tests/unit/`)

Traditional unit tests for individual components:
- API endpoint testing
- Service layer testing
- Model validation testing
- Utility function testing

**Usage:**
```bash
python -m pytest tests/unit/ -v
```

### 4. Integration Test Suite (`tests/integration/test_processing_pipeline.py`)

Comprehensive pytest-based integration tests that cover:
- Complete transcription workflow
- Queue management and status tracking
- Database consistency throughout processing
- Error scenarios and recovery
- Progress tracking and callbacks

**Usage:**
```bash
python -m pytest tests/integration/test_processing_pipeline.py -v
```

## Common Issues Diagnosed by Tests

### Status Tracking Problems

The tests help identify these common status tracking issues:

1. **Jobs disappear from queue during processing**
   - Checks that job status transitions correctly from `queued` → `processing` → `completed`
   - Verifies database consistency throughout the process

2. **Progress updates not working**
   - Tests progress callback functionality
   - Monitors API status endpoints for real-time progress updates

3. **Web interface shows empty queue**
   - Verifies queue status API endpoints return correct data
   - Checks database state matches API responses

4. **Transcription never completes**
   - Tests with mocked GPU services to isolate the issue
   - Monitors job execution and error handling

### Queue Management Issues

1. **Jobs not being processed**
   - Checks queue service status and worker availability
   - Verifies job submission and processing workflow

2. **Incorrect queue positions**
   - Tests queue position calculations
   - Verifies wait time estimations

3. **Database inconsistencies**
   - Checks that queue, transcription, and session tables remain consistent
   - Verifies foreign key relationships

## Test Environment Setup

The tests automatically configure the environment for development/testing:

```python
# Automatically set before test execution
os.environ.update({
    "TEST_MODE": "true",           # Enable test mode
    "MOCK_GPU": "true",            # Use mocked GPU services
    "CUDA_VISIBLE_DEVICES": "",    # Disable GPU detection
    "LOG_LEVEL": "DEBUG",          # Enable debug logging
    "ALLOWED_HOSTS": '["localhost", "127.0.0.1"]',
    "CORS_ORIGINS": '["http://localhost:3000"]',
})
```

## Mock Services

The tests use mocked versions of GPU-dependent services to run without CUDA:

### TranscriptionService Mock
- Simulates Whisper transcription
- Returns configurable text and segments
- Supports progress callbacks
- Can simulate errors for testing

### DiarizationService Mock
- Simulates speaker diarization
- Returns configurable speaker data
- Supports speaker matching
- Can simulate failures

## Test Data

The tests automatically generate test audio files using sine waves. No manual test files are required. Generated files are:
- 1-2 seconds in duration
- 16kHz sample rate
- Sine wave at 440Hz (A4 note)
- Automatically cleaned up after tests

## Expected Test Output

### Successful Run
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
✅ pytest
✅ All required dependencies are available

🧪 Setting up test environment...
📝 TEST_MODE=true
📝 MOCK_GPU=true
📝 CUDA_VISIBLE_DEVICES=
📝 LOG_LEVEL=DEBUG
✅ Test environment configured

🧪 Running: Pipeline Diagnostics
💻 Command: python scripts/diagnose_pipeline.py
...
✅ PASS: Pipeline Diagnostics

🧪 Running: Integration Tests
💻 Command: python scripts/run_pipeline_tests.py
...
✅ PASS: Integration Tests

============================================================
📊 FINAL TEST REPORT
============================================================
📅 Completed: 2024-01-15 10:32:15
⏱️  Duration: 135.42 seconds

📈 Results: 5/5 test suites passed
  ✅ PASS: dependencies
  ✅ PASS: environment_setup
  ✅ PASS: diagnostics
  ✅ PASS: unit_tests
  ✅ PASS: integration_tests

🎯 Overall Status: PASSED

🎉 All tests passed! Your SecureTranscribe pipeline is working correctly.
```

## Troubleshooting Failed Tests

### Import Errors
```
❌ Import failed: No module named 'app.main'
```
**Solution**: Ensure you're running from the project root directory and the virtual environment is active.

### scipy Not Found
```
❌ scipy is required for test audio generation
```
**Solution**: Install scipy: `pip install scipy numpy`

### Database Connection Issues
```
❌ Database check failed: permission denied
```
**Solution**: Ensure the application has write permissions to create/modify the database file.

### GPU Service Issues
```
❌ GPU services check failed: CUDA device not found
```
**Solution**: The tests should run with `MOCK_GPU=true`. If you're testing with real GPU, ensure CUDA is properly installed.

### API Endpoint Failures
```
❌ GET /api/queue/status: HTTP 500
```
**Solution**: Check the application logs for specific error messages and ensure the application can start successfully.

## Running Tests in CI/CD

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
name: Test Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest scipy numpy
      - name: Run tests
        run: python scripts/run_tests.py
```

## Performance Considerations

- Tests use small audio files (1-2 seconds) for speed
- Mocked services run quickly without GPU overhead
- Database operations use temporary databases when possible
- Tests have appropriate timeouts to prevent hanging
- Parallel test execution is supported where possible

## Contributing to Tests

When adding new tests:

1. **Use descriptive test names** that clearly indicate what's being tested
2. **Include proper cleanup** in teardown to prevent test pollution
3. **Mock external dependencies** to ensure tests run in any environment
4. **Test both success and failure scenarios** for comprehensive coverage
5. **Verify database state consistency** throughout the test
6. **Add progress tracking verification** for status-related tests
7. **Include appropriate assertions** for all expected behaviors

## Debugging Test Failures

### Enable Verbose Logging
```bash
LOG_LEVEL=DEBUG python scripts/run_tests.py
```

### Run Individual Tests
```bash
python -m pytest tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_file_upload_and_initial_status -v -s
```

### Inspect Test Data
Modify test fixtures to use permanent directories for inspection:
```python
# In test setup
temp_path = Path("/tmp/test_debug")  # Permanent instead of temporary
```

### Database Inspection
Connect directly to the test database to inspect state:
```bash
sqlite3 securetranscribe.db
.tables
SELECT * FROM processing_queue ORDER BY created_at DESC LIMIT 5;
```

## Security Notes

- Tests use temporary sessions with no real user data
- File uploads are isolated to temporary directories and cleaned up
- Database operations are isolated and automatically cleaned up
- No external network calls are made during tests
- Mocked services prevent access to real ML models

## Integration with Development Workflow

### Before Committing
```bash
python scripts/run_tests.py --quick
```

### Before Merging
```bash
python scripts/run_tests.py
```

### After Production Issues
```bash
python scripts/diagnose_pipeline.py
```

### Performance Testing
```bash
python scripts/run_tests.py --specific tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_concurrent_job_processing
```

This comprehensive testing suite should help identify and resolve the status tracking issues you're experiencing with the transcription pipeline. The tests provide clear feedback about where the process is failing and verify that all components work together correctly.