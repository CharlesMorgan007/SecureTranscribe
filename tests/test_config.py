"""
Test configuration module for SecureTranscribe.
Tests configuration loading, validation, and environment handling.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from app.core.config import Settings, get_settings, AUDIO_SETTINGS, EXPORT_SETTINGS


class TestSettings:
    """Test the Settings class."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()

        assert settings.database_url == "sqlite:///./securetranscribe.db"
        assert settings.secret_key == "dev-secret-key-change-in-production"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.whisper_model_size == "base"
        assert settings.pyannote_model == "pyannote/speaker-diarization-3.1"
        assert settings.sample_rate == 16000
        assert settings.max_workers == 4
        assert settings.queue_size == 10

    def test_whisper_model_validation(self):
        """Test Whisper model validation."""
        # Valid models
        for model in ["tiny", "base", "small", "medium", "large-v3"]:
            settings = Settings(whisper_model_size=model)
            assert settings.whisper_model_size == model

        # Invalid model
        with pytest.raises(ValueError, match="whisper_model_size must be one of"):
            Settings(whisper_model_size="invalid")

    def test_file_size_validation(self):
        """Test file size validation."""
        # Valid sizes
        for size in ["500MB", "1GB", "2.5GB"]:
            settings = Settings(max_file_size=size)
            assert settings.max_file_size == size

        # Invalid size
        with pytest.raises(ValueError, match="max_file_size must end with MB or GB"):
            Settings(max_file_size="invalid")

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Invalid level
        with pytest.raises(ValueError, match="log_level must be one of"):
            Settings(log_level="invalid")

    def test_max_file_size_bytes_property(self):
        """Test max_file_size_bytes property conversion."""
        # Test MB conversion
        settings = Settings(max_file_size="500MB")
        assert settings.max_file_size_bytes == 500 * 1024 * 1024

        # Test GB conversion
        settings = Settings(max_file_size="2GB")
        assert settings.max_file_size_bytes == 2 * 1024 * 1024 * 1024

    def test_use_gpu_property(self):
        """Test use_gpu property logic."""
        # Test mode without GPU
        settings = Settings(test_mode=True)
        assert settings.use_gpu is False

        settings = Settings(mock_gpu=True)
        assert settings.use_gpu is False

        # Test with GPU available (mocked)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            settings = Settings(test_mode=False, mock_gpu=False)
            assert settings.use_gpu is True

    def test_environment_file_loading(self):
        """Test loading configuration from environment file."""
        # This would test loading from .env file
        # In a real scenario, you'd create a temporary .env file
        pass


class TestGetSettings:
    """Test the get_settings function."""

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_instance(self):
        """Test that get_settings returns Settings instance."""
        settings = get_settings()

        assert isinstance(settings, Settings)


class TestAudioSettings:
    """Test audio settings configuration."""

    def test_audio_settings_structure(self):
        """Test audio settings have required keys."""
        required_keys = [
            "sample_rate",
            "chunk_length_s",
            "overlap_length_s",
            "max_speakers",
            "min_speaker_duration",
            "confidence_threshold",
            "supported_formats",
            "preview_duration",
            "min_clip_duration",
        ]

        for key in required_keys:
            assert key in AUDIO_SETTINGS

    def test_audio_settings_values(self):
        """Test audio settings have valid values."""
        assert AUDIO_SETTINGS["sample_rate"] == 16000
        assert AUDIO_SETTINGS["chunk_length_s"] == 30
        assert AUDIO_SETTINGS["overlap_length_s"] == 5
        assert AUDIO_SETTINGS["max_speakers"] == 10
        assert AUDIO_SETTINGS["min_speaker_duration"] == 2.0
        assert AUDIO_SETTINGS["confidence_threshold"] == 0.8
        assert isinstance(AUDIO_SETTINGS["supported_formats"], list)
        assert AUDIO_SETTINGS["preview_duration"] == 10
        assert AUDIO_SETTINGS["min_clip_duration"] == 2


class TestExportSettings:
    """Test export settings configuration."""

    def test_export_settings_structure(self):
        """Test export settings have required keys."""
        required_keys = [
            "formats",
            "include_options",
            "pdf_template",
            "csv_delimiter",
            "json_indent",
        ]

        for key in required_keys:
            assert key in EXPORT_SETTINGS

    def test_export_settings_values(self):
        """Test export settings have valid values."""
        assert isinstance(EXPORT_SETTINGS["formats"], list)
        assert "pdf" in EXPORT_SETTINGS["formats"]
        assert "csv" in EXPORT_SETTINGS["formats"]
        assert "txt" in EXPORT_SETTINGS["formats"]
        assert "json" in EXPORT_SETTINGS["formats"]

        assert isinstance(EXPORT_SETTINGS["include_options"], list)
        assert "meeting_summary" in EXPORT_SETTINGS["include_options"]
        assert "action_items" in EXPORT_SETTINGS["include_options"]

        assert EXPORT_SETTINGS["csv_delimiter"] == ","
        assert EXPORT_SETTINGS["json_indent"] == 2


class TestConfigurationIntegration:
    """Integration tests for configuration."""

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "sqlite:///./test.db",
            "SECRET_KEY": "test-secret-key",
            "DEBUG": "true",
            "WHISPER_MODEL_SIZE": "small",
            "MAX_WORKERS": "2",
        },
    )
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        # Clear cached settings
        get_settings.cache_clear()

        settings = get_settings()

        assert settings.database_url == "sqlite:///./test.db"
        assert settings.secret_key == "test-secret-key"
        assert settings.debug is True
        assert settings.whisper_model_size == "small"
        assert settings.max_workers == 2

    def test_configuration_validation_chain(self):
        """Test multiple configuration validations."""
        # Test all validations work together
        settings = Settings(
            whisper_model_size="medium", max_file_size="1GB", log_level="WARNING"
        )

        assert settings.whisper_model_size == "medium"
        assert settings.max_file_size_bytes == 1024 * 1024 * 1024
        assert settings.log_level == "WARNING"

    @pytest.mark.parametrize(
        "model_size", ["tiny", "base", "small", "medium", "large-v3"]
    )
    def test_all_valid_whisper_models(self, model_size):
        """Test all valid Whisper model sizes."""
        settings = Settings(whisper_model_size=model_size)
        assert settings.whisper_model_size == model_size

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        with pytest.raises(ValueError):
            Settings(whisper_model_size="invalid_model")

        with pytest.raises(ValueError):
            Settings(max_file_size="invalid_size")

        with pytest.raises(ValueError):
            Settings(log_level="invalid_level")


if __name__ == "__main__":
    pytest.main([__file__])
