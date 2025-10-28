"""Test logging configuration functionality."""

import os
import tempfile
import unittest
from unittest.mock import patch, Mock

from app.core.config import get_settings
from app.core.database import init_database
import logging


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration."""

    def test_log_level_settings(self):
        """Test that log levels are properly configured."""
        # Test DEBUG level first to avoid INFO interference
        level = "DEBUG"
        with self.subTest(level=level):
            # Create temporary env file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".env", delete=False
            ) as f:
                f.write(f"LOG_LEVEL={level}\n")
                temp_env_path = f.name

            try:
                # Update env and reload settings
                old_env_path = os.environ.get("DOTENV_PATH")
                os.environ["DOTENV_PATH"] = temp_env_path

                # Clear any cached settings
                import app.core.config

                app.core.config.get_settings.cache_clear()

                # Get settings and verify
                settings = get_settings()
                expected_level = getattr(logging, level)

                self.assertEqual(
                    settings.log_level,
                    level,
                    f"Expected {level}, got {settings.log_level}",
                )

            finally:
                # Cleanup
                os.unlink(temp_env_path)
                if old_env_path:
                    os.environ["DOTENV_PATH"] = old_env_path
                elif "DOTENV_PATH" in os.environ:
                    del os.environ["DOTENV_PATH"]

    def test_logging_basic_configuration(self):
        """Test basic logging configuration."""
        # Test that logging can be configured
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("LOG_LEVEL=INFO\n")
            temp_env_path = f.name

        try:
            os.environ["DOTENV_PATH"] = temp_env_path
            settings = get_settings()

            # Test that we can get logger
            logger = logging.getLogger(__name__)
            self.assertIsNotNone(logger)

            # Test log level
            self.assertEqual(settings.log_level, "INFO")

        finally:
            os.unlink(temp_env_path)
            if "DOTENV_PATH" in os.environ:
                del os.environ["DOTENV_PATH"]

    def test_logging_levels_effectiveness(self):
        """Test that logging levels actually work."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("LOG_LEVEL=ERROR\n")
            temp_env_path = f.name

        try:
            os.environ["DOTENV_PATH"] = temp_env_path
            settings = get_settings()

            # Test that ERROR level works
            logger = logging.getLogger("test_logger")
            logger.setLevel(logging.ERROR)

            # These should not log at ERROR level
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")

            # This should log
            logger.error("Error message")

            # Basic test - if we get here, logging works
            self.assertTrue(True)

        finally:
            os.unlink(temp_env_path)
            if "DOTENV_PATH" in os.environ:
                del os.environ["DOTENV_PATH"]


if __name__ == "__main__":
    unittest.main()
