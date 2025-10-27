# Integration Tests for SecureTranscribe Processing Pipeline

This directory contains comprehensive integration tests for the SecureTranscribe transcription and diarization processing pipeline.

## Overview

The integration tests verify that the entire processing pipeline works correctly, from file upload through transcription, diarization, and status tracking. These tests are designed to run on development machines without CUDA/GPU requirements by using mocked services.

## Test Coverage

### Core Functionality Tests
- **File Upload**: Verifies audio files can be uploaded and registered
- **Session Management**: Tests user session creation and validation
- **Queue Management**: Validates job queuing and status tracking
- **API Endpoints**: Confirms all API endpoints respond correctly

### Processing Pipeline Tests
- **Mock GPU Processing**: Tests transcription with mocked GPU services
- **Progress Tracking**: Verifies progress callbacks and status updates
- **Error Handling**: Tests error scenarios and status propagation
- **Database Consistency**: Ensures database state remains consistent

### Advanced Tests
- **Concurrent Processing**: Tests multiple jobs running simultaneously
- **Job Cancellation**: Verifies jobs can be cancelled mid-processing
- **Queue Status Updates**: Confirms queue statistics update correctly

## Running Tests

### Quick Start (Recommended)

Use the standalone test runner:
```bash
cd SecureTranscribe
python scripts/run_pipeline_tests.py
```

### Running with pytest

For more detailed test output:
```bash
cd SecureTranscribe
python -m pytest tests/integration/test_processing_pipeline.py -v
```

### Running Individual Test Classes

```bash
# Run specific test class
python -m pytest tests/integration/test_processing_pipeline.py::TestProcessingPipeline -v

# Run specific test method
python -m pytest tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_file_upload_and_initial_status -v
```

## Environment Setup

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

## Test Data

The tests automatically generate a simple sine wave audio file for testing. No manual test files are required.

## Mock Services

The tests use mocked versions of GPU-dependent services:

### TranscriptionService Mock
- Simulates Whisper transcription
- Returns configurable text and segments
- Supports progress callbacks

### DiarizationService Mock
- Simulates speaker diarization
- Returns configurable speaker data
- Supports speaker matching

## Test Structure

### TestProcessingPipeline Class

Main test class containing all integration tests:

```python
class TestProcessingPipeline:
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        # Sets up temporary directories, test audio, etc.
    
    def test_file_upload_and_initial_status(self):
        # Tests file upload and initial status tracking
    
    def test_transcription_processing_with_mock_gpu(self):
        # Tests complete transcription pipeline with mocked GPU
```

### Key Test Methods

#### Core Tests
- `test_file_upload_and_initial_status()`: Basic upload functionality
- `test_transcription_processing_with_mock_gpu()`: Complete pipeline test
- `test_queue_status_updates_during_processing()`: Queue management

#### Advanced Tests
- `test_concurrent_job_processing()`: Multiple jobs simultaneously
- `test_job_cancellation_and_status_tracking()`: Job cancellation
- `test_error_handling_and_status_tracking()`: Error scenarios
- `test_progress_callback_functionality()`: Progress updates
- `test_database_state_consistency()`: Database integrity

## Expected Behavior

### Successful Test Run

When all tests pass, you should see output like:
```
🚀 Starting SecureTranscribe Pipeline Tests
==================================================
✅ Created test audio file: /tmp/tmpXXXXXX/test_audio.wav
✅ All modules imported successfully
✅ Health check passed
✅ Session created: abc-123-def-456
✅ File uploaded successfully: 789
✅ Transcription started successfully
  📊 Status: processing, Progress: 25%, Step: Loading audio...
  📊 Status: processing, Progress: 50%, Step: Transcribing...
  📊 Status: processing, Progress: 75%, Step: Finalizing results...
✅ Transcription completed successfully
✅ Final text: This is a mock transcription test...
✅ Queue status tracking is working correctly
✅ Error handling is working correctly
==================================================
📊 TEST SUMMARY
==================================================
✅ PASS: Module Imports
✅ PASS: Basic Functionality
✅ PASS: File Upload
✅ PASS: Mock Processing
✅ PASS: Progress Updates
✅ PASS: Queue Status Tracking
✅ PASS: Error Handling
--------------------------------------------------
Total: 7/7 tests passed
🎉 All tests passed! The processing pipeline is working correctly.
```

### Common Issues and Solutions

#### Import Errors
```
❌ Import failed: No module named 'app.main'
```
**Solution**: Ensure you're running from the project root directory and the virtual environment is active.

#### scipy Not Found
```
❌ scipy is required for test audio generation
```
**Solution**: Install scipy: `pip install scipy numpy`

#### Database Connection Issues
```
❌ Database connection failed
```
**Solution**: Ensure the database file is writable and the application can create it.

## Troubleshooting

### Debug Mode

To get more detailed output, set the log level:
```bash
LOG_LEVEL=DEBUG python scripts/run_pipeline_tests.py
```

### Individual Test Debugging

Run a single test with detailed output:
```bash
python -m pytest tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_file_upload_and_initial_status -v -s
```

### Test Data Inspection

The tests use temporary directories that are automatically cleaned up. To inspect test data, modify the `setup_test_environment` fixture to use a permanent directory.

## Continuous Integration

These tests are designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Pipeline Tests
  run: |
    python scripts/run_pipeline_tests.py
```

## Contributing

When adding new tests:

1. Use descriptive test method names
2. Include proper cleanup in teardown
3. Mock external dependencies
4. Test both success and failure scenarios
5. Verify database state consistency

## Performance Considerations

- Tests use small audio files (1-2 seconds)
- Mocked services run quickly
- Database operations use in-memory SQLite when possible
- Tests have appropriate timeouts to prevent hanging

## Security Notes

- Tests use temporary sessions with no real user data
- File uploads are isolated to temporary directories
- Database operations are isolated and cleaned up
- No external network calls are made during tests