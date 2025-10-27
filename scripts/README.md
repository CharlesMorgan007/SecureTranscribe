# SecureTranscribe Testing Scripts

This directory contains utility scripts for testing and diagnosing the SecureTranscribe transcription pipeline.

## Available Scripts

### `run_tests.py` (Master Test Runner)
The main entry point for running all tests. Provides comprehensive testing of the transcription pipeline.

**Usage:**
```bash
# Run all tests
python scripts/run_tests.py

# Quick test suite for fast feedback
python scripts/run_tests.py --quick

# Run only diagnostics
python scripts/run_tests.py --diagnostics-only

# Run specific test
python scripts/run_tests.py --specific tests/unit/test_api.py

# Skip certain test types
python scripts/run_tests.py --skip-diagnostics --skip-unit
```

### `diagnose_pipeline.py` (Pipeline Diagnostics)
Comprehensive diagnostic tool that checks all components of the transcription pipeline. This is the best first step when troubleshooting issues.

**Usage:**
```bash
python scripts/diagnose_pipeline.py
```

**What it checks:**
- Environment variables and configuration
- Module imports and dependencies
- Database connectivity and state
- Queue service status
- API endpoint functionality
- File upload workflow
- GPU service configuration
- Recent log entries for errors

### `run_pipeline_tests.py` (Integration Tests)
Standalone integration test runner that verifies the complete transcription workflow with mocked GPU services.

**Usage:**
```bash
python scripts/run_pipeline_tests.py
```

**What it tests:**
- File upload and initial status tracking
- Transcription processing with mocked services
- Queue status updates during processing
- Progress callback functionality
- Error handling and status propagation
- Database state consistency

## Quick Troubleshooting Guide

### If transcription jobs disappear from queue:
```bash
python scripts/diagnose_pipeline.py
```
Look for:
- Queue service status
- Database consistency checks
- Recent job entries and their status

### If web interface shows empty queue:
```bash
python scripts/run_tests.py --specific tests/integration/test_processing_pipeline.py::TestProcessingPipeline::test_queue_status_tracking
```

### If transcription never completes:
```bash
python scripts/run_pipeline_tests.py
```
This will test with mocked GPU services to isolate the issue.

### If you're seeing errors but don't know why:
```bash
python scripts/diagnose_pipeline.py
```
Check the "Recent Logs Check" section for error patterns.

## Common Issues and Solutions

### Import Errors
- Ensure you're running from the project root directory
- Activate your virtual environment: `source venv/bin/activate`
- Install missing dependencies: `pip install -r requirements.txt`

### Permission Errors
- Check that uploads/, processed/, and logs/ directories are writable
- Ensure database file permissions allow read/write access

### GPU/CUDA Issues
- The tests automatically use `MOCK_GPU=true` to bypass GPU requirements
- For production debugging, check GPU service configuration section in diagnostics

### Database Issues
- Ensure the database file is not corrupted: `sqlite3 securetranscribe.db ".tables"`
- Check that the application has proper file permissions

## Running in Different Modes

### Development Mode (Default)
Tests run with mocked GPU services and temporary directories:
```bash
python scripts/run_tests.py
```

### Production-Like Testing
Run with real GPU services (if available):
```bash
unset MOCK_GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run_tests.py --skip-diagnostics
```

### Continuous Integration
All tests are designed to run in CI/CD environments without manual intervention:
```bash
python scripts/run_tests.py
```

## Interpreting Test Results

### ✅ PASS
Everything is working correctly for this component.

### ❌ FAIL
There's an issue with this component. Check the detailed output for specific error messages.

### ⚠️ WARNING
Non-critical issue that should be addressed but doesn't prevent operation.

### ℹ️ INFO
Informational message about the system state.

## Environment Variables for Testing

The test scripts automatically set these variables, but you can override them:

```bash
# Set to true to enable test-specific behavior
TEST_MODE=true

# Set to true to use mocked GPU services
MOCK_GPU=true

# Override GPU visibility
CUDA_VISIBLE_DEVICES=

# Set logging level for detailed output
LOG_LEVEL=DEBUG

# Security settings for testing
ALLOWED_HOSTS=["localhost", "127.0.0.1"]
CORS_ORIGINS=["http://localhost:3000"]
```

## Getting Help

### After running diagnostics
1. Look for ❌ symbols in the output
2. Check the "Recommendations" section at the end
3. Review the specific error messages provided

### For persistent issues
1. Run with debug logging: `LOG_LEVEL=DEBUG python scripts/diagnose_pipeline.py`
2. Check the application logs: `tail -f logs/securetranscribe.log`
3. Run specific failing test with verbose output: `python -m pytest -v -s`

## Integration with Development Workflow

### Before committing changes:
```bash
python scripts/run_tests.py --quick
```

### Before merging to main:
```bash
python scripts/run_tests.py
```

### When investigating production issues:
```bash
python scripts/diagnose_pipeline.py
```

## Performance Tips

- Use `--quick` for rapid feedback during development
- Mocked GPU services run much faster than real ones
- Tests use small audio files to minimize runtime
- Database operations use optimized queries