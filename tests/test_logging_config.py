"""
Test script to verify logging configuration works correctly with .env settings.
Run this with: python -m tests.test_logging_config
"""

import os
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(f"LOG_LEVEL={level_name}\n")
            f.write(f'ALLOWED_HOSTS=["localhost", "127.0.0.1"]\n')
            f.write(f'CORS_ORIGINS=["http://localhost:3000"]\n')
            temp_env_path = f.name

        try:
            # Set the env file path
            os.environ['DOTENV_PATH'] = temp_env_path

            # Clear cached settings to force reload
            if 'app.core.config.get_settings' in sys.modules['app.core.config'].__dict__:
                sys.modules['app.core.config'].get_settings.cache_clear()

            # Get settings and verify
            settings = get_settings()
            expected_level = getattr(logging, level_name)

            print(f"Testing LOG_LEVEL={level_name}:")
            print(f"  Settings.log_level: {settings.log_level}")
            print(f"  Expected logging level: {expected_level}")
            print(f"  Match: {settings.log_level == level_name}")
            print()

            assert settings.log_level == level_name, f"Expected {level_name}, got {settings.log_level}"

        finally:
            # Cleanup
            os.unlink(temp_env_path)
            if 'DOTENV_PATH' in os.environ:
                del os.environ['DOTENV_PATH']

    print("All log level tests passed!")


if __name__ == "__main__":
    test_logging_levels()
```



The problem was in `app/main.py` where the logging configuration was set before loading the settings. I moved the `settings = get_settings()` call before the `logging.basicConfig()` and changed the hardcoded `level=logging.INFO` to `level=getattr(logging, settings.log_level)`.

### 2. Updated .env.example with correct values

I updated the `.env.example` file to use the proper JSON array format for list fields:

```bash
ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0"]
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### Important Note

When you set `LOG_LEVEL=WARNING` (or `WARN`), you need to be aware of two things:

1. **Use "WARNING" not "WARN"**: While Python's logging accepts both, the validation in the config only accepts "WARNING" (the official name).

2. **Uvicorn's own logs**: The INFO messages you see at startup from uvicorn (like "Uvicorn running on http://0.0.0.0:8001") are controlled by uvicorn's own logging, not your application's logging. To control uvicorn's log level, you need to pass it as a parameter:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001 --log-level warning
```

Your application's logger will now respect the LOG_LEVEL setting from your `.env` file. The uvicorn startup messages will still appear at INFO level unless you specify the `--log-level` parameter to uvicorn.
