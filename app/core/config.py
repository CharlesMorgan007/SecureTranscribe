"""
Core configuration management for SecureTranscribe.
Handles environment variables, application settings, and audio processing parameters.
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database
    database_url: str = "sqlite:///./securetranscribe.db"

    # Application
    secret_key: str = "dev-secret-key-change-in-production"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # GPU and Model Settings
    cuda_visible_devices: str = "0"
    torch_cuda_arch_list: str = "8.6"
    whisper_model_size: str = "base"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"

    # File Storage
    upload_dir: str = "./uploads"
    processed_dir: str = "./processed"
    max_file_size: str = "500MB"
    cleanup_delay: int = 3600

    # Audio Processing Settings
    sample_rate: int = 16000
    chunk_length_s: int = 30
    overlap_length_s: int = 5
    max_speakers: int = 10
    min_speaker_duration: float = 2.0
    confidence_threshold: float = 0.8

    # Queue and Processing
    max_workers: int = 4
    queue_size: int = 10
    processing_timeout: int = 3600

    # Security
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/securetranscribe.log"

    # Development
    test_mode: bool = False
    mock_gpu: bool = False

    @validator("whisper_model_size")
    def validate_whisper_model(cls, v):
        valid_models = ["tiny", "base", "small", "medium", "large-v3"]
        if v not in valid_models:
            raise ValueError(f"whisper_model_size must be one of {valid_models}")
        return v

    @validator("max_file_size")
    def validate_file_size(cls, v):
        if not v.endswith(("MB", "GB")):
            raise ValueError("max_file_size must end with MB or GB")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @property
    def max_file_size_bytes(self) -> int:
        """Convert max_file_size to bytes."""
        size_str = self.max_file_size.upper()
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        return 500 * 1024 * 1024  # Default 500MB

    @property
    def use_gpu(self) -> bool:
        """Determine if GPU acceleration should be used."""
        if self.test_mode or self.mock_gpu:
            return False
        return os.environ.get("CUDA_VISIBLE_DEVICES") != ""

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Audio processing configuration
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "chunk_length_s": 30,
    "overlap_length_s": 5,
    "max_speakers": 10,
    "min_speaker_duration": 2.0,
    "confidence_threshold": 0.8,
    "supported_formats": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    "preview_duration": 10,  # seconds for speaker preview clips
    "min_clip_duration": 2,  # seconds for speaker preview
}

# Export configuration
EXPORT_SETTINGS = {
    "formats": ["pdf", "csv", "txt", "json"],
    "include_options": [
        "meeting_summary",
        "action_items",
        "next_steps",
        "recommendations",
    ],
    "pdf_template": "default",
    "csv_delimiter": ",",
    "json_indent": 2,
}

# Database settings
DATABASE_SETTINGS = {
    "echo": False,  # Set to True for SQL debugging
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}

# Security settings
SECURITY_SETTINGS = {
    "session_timeout": 3600,  # 1 hour
    "max_upload_size": 500 * 1024 * 1024,  # 500MB
    "allowed_mime_types": [
        "audio/mpeg",
        "audio/wav",
        "audio/mp4",
        "audio/flac",
        "audio/ogg",
    ],
    "rate_limit": {"requests_per_minute": 60, "burst_size": 10},
}
