"""
Test script to verify logging configuration works correctly with .env settings.
Run this with: python -m tests.test_logging_config
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the app directory to the path so we can import app modules
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root))

from app.core.config import get_settings


def test_logging_levels():
    """Test that logging levels are properly configured from settings."""

    # Test different log levels
    test_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level_name in test_levels:
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(f"LOG_LEVEL={level_name}\n")
            f.write(f'ALLOWED_HOSTS=["localhost", "127.0.0.1"]\n')
            f.write(f'CORS_ORIGINS=["http://localhost:3000"]\n')
            temp_env_path = f.name

        try:
            # Set the env file path
            os.environ["DOTENV_PATH"] = temp_env_path

            # Clear cached settings to force reload
            if (
                "app.core.config.get_settings"
                in sys.modules["app.core.config"].__dict__
            ):
                sys.modules["app.core.config"].get_settings.cache_clear()

            # Get settings and verify
            settings = get_settings()
            expected_level = getattr(logging, level_name)

            print(f"Testing LOG_LEVEL={level_name}:")
            print(f"  Settings.log_level: {settings.log_level}")
            print(f"  Expected logging level: {expected_level}")
            print(f"  Match: {settings.log_level == level_name}")
            print()

            assert settings.log_level == level_name, (
                f"Expected {level_name}, got {settings.log_level}"
            )

        finally:
            # Cleanup
            os.unlink(temp_env_path)
            if "DOTENV_PATH" in os.environ:
                del os.environ["DOTENV_PATH"]

    print("All log level tests passed!")


if __name__ == "__main__":
    test_logging_levels()
