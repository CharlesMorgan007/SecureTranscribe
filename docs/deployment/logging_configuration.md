# Logging Configuration Guide

## Overview

This guide explains how to properly configure logging for SecureTranscribe, including common issues and solutions.

## Environment Variables

### Core Logging Settings

```bash
# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=WARNING

# Log file location
LOG_FILE=./logs/securetranscribe.log
```

### Required Security Settings

The following settings MUST be in JSON array format for proper parsing:

```bash
# Hostnames allowed to access the application
ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0"]

# CORS allowed origins
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

## Common Issues and Solutions

### Issue: LOG_LEVEL Setting Not Working

**Problem**: Setting `LOG_LEVEL=WARNING` in `.env` file doesn't affect the log output.

**Solution**: Ensure you're using the correct log level name:
- Use `WARNING` instead of `WARN`
- Valid levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

### Issue: JSON Parsing Error on Startup

**Problem**: Application fails with JSON parsing error related to `allowed_hosts`.

**Solution**: Make sure list-type environment variables use proper JSON format:

✅ **Correct**:
```bash
ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0"]
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

❌ **Incorrect**:
```bash
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Issue: Uvicorn Startup Messages Still Show INFO

**Problem**: Uvicorn startup messages appear at INFO level even when `LOG_LEVEL=WARNING`.

**Explanation**: Uvicorn has its own logging configuration that's separate from the application logging.

**Solution**: Control uvicorn's logging with the `--log-level` parameter:

```bash
# Application logs will use LOG_LEVEL from .env
# Uvicorn logs will use the --log-level parameter
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001 --log-level warning
```

## Production Deployment

### Complete .env File Example

```bash
# Database
DATABASE_URL=sqlite:///./securetranscribe.db

# Security (CRITICAL: Change in production!)
SECRET_KEY=your-very-secure-secret-key-here

# Host Configuration
ALLOWED_HOSTS=["your-domain.com", "www.your-domain.com"]
CORS_ORIGINS=["https://your-domain.com", "https://www.your-domain.com"]

# Logging
LOG_LEVEL=WARNING
LOG_FILE=/var/log/securetranscribe/app.log

# GPU Settings (if applicable)
CUDA_VISIBLE_DEVICES=0
WHISPER_MODEL_SIZE=base

# File Storage
UPLOAD_DIR=/var/lib/securetranscribe/uploads
PROCESSED_DIR=/var/lib/securetranscribe/processed
MAX_FILE_SIZE=500MB

# Performance
MAX_WORKERS=4
QUEUE_SIZE=10
PROCESSING_TIMEOUT=3600
```

### Log Rotation (Optional)

For production, consider setting up log rotation to prevent log files from growing too large:

```bash
# Example logrotate configuration for /etc/logrotate.d/securetranscribe
/var/log/securetranscribe/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 www-data www-data
    postrotate
        systemctl reload securetranscribe || true
    endscript
}
```

## Debugging Logging Issues

### Test Your Configuration

Create a simple test script to verify logging is working:

```python
import logging
from app.core.config import get_settings

settings = get_settings()
print(f"Configured log level: {settings.log_level}")
print(f"Python logging level: {getattr(logging, settings.log_level)}")

# Test logging at different levels
logger = logging.getLogger(__name__)
logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")
logger.error("This is an ERROR message")
logger.critical("This is a CRITICAL message")
```

### Verifying Log Output

1. Check console output when running the application
2. Check the log file specified in `LOG_FILE`
3. Verify only messages at or above your configured level appear

## Troubleshooting Checklist

- [ ] Verify `.env` file exists and is readable
- [ ] Check that list variables use JSON array format
- [ ] Ensure `LOG_LEVEL` uses valid level names
- [ ] Verify log directory exists and is writable
- [ ] Test with different log levels to isolate the issue
- [ ] Check both console and file outputs
- [ ] Use `--log-level` parameter for uvicorn-specific logs

## Security Considerations

- Never commit `.env` files with sensitive data to version control
- Use strong `SECRET_KEY` values in production
- Restrict `ALLOWED_HOSTS` to your actual domains
- Set appropriate `CORS_ORIGINS` for your frontend
- Consider using environment-specific configuration files