"""
Integration test configuration for SecureTranscribe.
Provides test-specific settings and mock configurations.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

# Test environment flag
TEST_MODE = True

# Mock GPU configuration
MOCK_GPU = True
CUDA_VISIBLE_DEVICES = ""

# Test database configuration
TEST_DATABASE_URL = "sqlite:///./test_securetranscribe.db"

# Test directories configuration
TEST_BASE_DIR = Path(tempfile.gettempdir()) / "securetranscribe_tests"
TEST_UPLOAD_DIR = TEST_BASE_DIR / "uploads"
TEST_PROCESSED_DIR = TEST_BASE_DIR / "processed"
TEST_LOGS_DIR = TEST_BASE_DIR / "logs"

# Test logging configuration
TEST_LOG_LEVEL = "DEBUG"
TEST_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Test API configuration
TEST_HOST = "127.0.0.1"
TEST_PORT = 8001  # Different from default to avoid conflicts
TEST_BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"

# Test session configuration
TEST_SESSION_SECRET = "test-secret-key-for-integration-tests"
TEST_SESSION_TIMEOUT = 3600  # 1 hour

# Test audio processing configuration
TEST_SAMPLE_RATE = 16000
TEST_CHUNK_LENGTH = 10  # Short chunks for faster testing
TEST_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Test queue configuration
TEST_MAX_WORKERS = 2
TEST_QUEUE_SIZE = 5
TEST_CLEANUP_DELAY = 60  # 1 minute for tests

# Mock transcription settings
MOCK_TRANSCRIPTION_DURATION = 2.0  # seconds
MOCK_DIARIZATION_SPEAKERS = 2
MOCK_TRANSCRIPT_TEXT = "This is a mock transcription for testing purposes."

# Test file configurations
TEST_AUDIO_DURATION = 5  # seconds
TEST_AUDIO_FORMAT = "wav"
TEST_AUDIO_SAMPLE_RATE = 16000
TEST_AUDIO_CHANNELS = 1

# Security test configurations
TEST_ALLOWED_HOSTS = ["localhost", "127.0.0.1", TEST_HOST]
TEST_CORS_ORIGINS = ["http://localhost:3000", f"http://{TEST_HOST}:{TEST_PORT}"]

# Performance test thresholds
MAX_RESPONSE_TIME = 5.0  # seconds
MAX_UPLOAD_TIME = 10.0  # seconds
MAX_PROCESSING_TIME = 30.0  # seconds

# Test data configurations
TEST_SPEAKERS = [
    {"id": "speaker_0", "name": "Speaker 1", "gender": "unknown"},
    {"id": "speaker_1", "name": "Speaker 2", "gender": "unknown"},
]

# Mock response templates
MOCK_TRANSCRIPT_RESPONSE = {
    "text": MOCK_TRANSCRIPT_TEXT,
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "This is a mock",
            "speaker": "speaker_0",
        },
        {
            "start": 2.0,
            "end": 5.0,
            "text": "transcription for testing purposes.",
            "speaker": "speaker_1",
        },
    ],
    "language": "en",
    "duration": TEST_AUDIO_DURATION,
}


def get_test_environment() -> Dict[str, Any]:
    """Get complete test environment configuration."""
    return {
        "TEST_MODE": TEST_MODE,
        "MOCK_GPU": MOCK_GPU,
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        "DATABASE_URL": TEST_DATABASE_URL,
        "UPLOAD_DIR": str(TEST_UPLOAD_DIR),
        "PROCESSED_DIR": str(TEST_PROCESSED_DIR),
        "LOGS_DIR": str(TEST_LOGS_DIR),
        "LOG_LEVEL": TEST_LOG_LEVEL,
        "HOST": TEST_HOST,
        "PORT": TEST_PORT,
        "SESSION_SECRET": TEST_SESSION_SECRET,
        "ALLOWED_HOSTS": TEST_ALLOWED_HOSTS,
        "CORS_ORIGINS": TEST_CORS_ORIGINS,
    }


def setup_test_environment():
    """Set up test environment variables and directories."""
    env_config = get_test_environment()

    # Set environment variables
    for key, value in env_config.items():
        os.environ[key] = str(value)

    # Create test directories
    for directory in [TEST_UPLOAD_DIR, TEST_PROCESSED_DIR, TEST_LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    return env_config


def cleanup_test_environment():
    """Clean up test environment after tests."""
    import shutil

    # Clean up test directories
    if TEST_BASE_DIR.exists():
        shutil.rmtree(TEST_BASE_DIR, ignore_errors=True)

    # Clean up test database
    test_db_path = Path(TEST_DATABASE_URL.replace("sqlite:///", ""))
    if test_db_path.exists():
        test_db_path.unlink()


# Test data fixtures
def get_test_audio_file_path() -> Path:
    """Get path to a test audio file."""
    return TEST_UPLOAD_DIR / f"test_audio.{TEST_AUDIO_FORMAT}"


def get_test_audio_config() -> Dict[str, Any]:
    """Get configuration for generating test audio."""
    return {
        "duration": TEST_AUDIO_DURATION,
        "sample_rate": TEST_AUDIO_SAMPLE_RATE,
        "channels": TEST_AUDIO_CHANNELS,
        "format": TEST_AUDIO_FORMAT,
    }


# Mock service configurations
def get_mock_transcription_config() -> Dict[str, Any]:
    """Get mock transcription service configuration."""
    return {
        "model": "mock-model",
        "device": "cpu",
        "language": "en",
        "task": "transcribe",
        "duration": MOCK_TRANSCRIPTION_DURATION,
    }


def get_mock_diarization_config() -> Dict[str, Any]:
    """Get mock diarization service configuration."""
    return {
        "model": "mock-diarization-model",
        "device": "cpu",
        "num_speakers": MOCK_DIARIZATION_SPEAKERS,
        "min_speaker_duration": 0.5,
    }
