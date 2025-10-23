"""
Utility helper functions for common operations.
Provides helper functions for file handling, validation, formatting, and other common tasks.
"""

import os
import hashlib
import mimetypes
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import uuid
import re
import json

logger = logging.getLogger(__name__)


def generate_unique_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return secrets.token_urlsafe(32)


def generate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (md5, sha1, sha256, etc.)

    Returns:
        Hexadecimal hash string
    """
    hash_func = getattr(hashlib, algorithm)()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in HH:MM:SS format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(timestamp: Union[datetime, str, float]) -> str:
    """
    Format timestamp to ISO format.

    Args:
        timestamp: Timestamp as datetime, string, or Unix timestamp

    Returns:
        ISO formatted timestamp string
    """
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.isoformat()
        except ValueError:
            return timestamp
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
        return dt.isoformat()
    else:
        return str(timestamp)


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_filename(filename: str) -> bool:
    """
    Validate filename for security.

    Args:
        filename: Filename to validate

    Returns:
        True if safe, False otherwise
    """
    # Check for path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        return False

    # Check for invalid characters
    invalid_chars = '<>:"|?*'
    if any(char in filename for char in invalid_chars):
        return False

    # Check for reserved names (Windows)
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        return False

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Remove path traversal
    filename = os.path.basename(filename)

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[: 255 - len(ext)] + ext

    return filename


def get_file_mime_type(file_path: str) -> Optional[str]:
    """
    Get MIME type of a file.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string or None
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def is_audio_file(file_path: str) -> bool:
    """
    Check if file is an audio file based on MIME type.

    Args:
        file_path: Path to the file

    Returns:
        True if audio file, False otherwise
    """
    mime_type = get_file_mime_type(file_path)
    audio_mime_types = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/wave",
        "audio/mp4",
        "audio/m4a",
        "audio/flac",
        "audio/ogg",
        "audio/x-wav",
        "audio/x-mpeg",
        "audio/x-flac",
    }

    return mime_type in audio_mime_types


def create_temp_file(
    suffix: str = "", prefix: str = "securetranscribe_", directory: Optional[str] = None
) -> str:
    """
    Create a temporary file.

    Args:
        suffix: File suffix/extension
        prefix: File prefix
        directory: Directory for temp file

    Returns:
        Path to created temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
    os.close(fd)  # Close file descriptor, we'll open it later
    return temp_path


def create_temp_directory(prefix: str = "securetranscribe_") -> str:
    """
    Create a temporary directory.

    Args:
        prefix: Directory prefix

    Returns:
        Path to created temporary directory
    """
    return tempfile.mkdtemp(prefix=prefix)


def safe_remove_file(file_path: str) -> bool:
    """
    Safely remove a file.

    Args:
        file_path: Path to file to remove

    Returns:
        True if removed, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
        return False


def safe_remove_directory(dir_path: str, recursive: bool = True) -> bool:
    """
    Safely remove a directory.

    Args:
        dir_path: Path to directory to remove
        recursive: Whether to remove recursively

    Returns:
        True if removed, False otherwise
    """
    try:
        if os.path.exists(dir_path):
            if recursive:
                import shutil

                shutil.rmtree(dir_path)
            else:
                os.rmdir(dir_path)
            logger.debug(f"Removed directory: {dir_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to remove directory {dir_path}: {e}")
        return False


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Ensure directory exists, create if necessary.

    Args:
        dir_path: Path to directory

    Returns:
        True if directory exists or was created
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False


def get_available_disk_space(path: str) -> int:
    """
    Get available disk space in bytes.

    Args:
        path: Path to check

    Returns:
        Available space in bytes
    """
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except Exception as e:
        logger.error(f"Failed to get disk space for {path}: {e}")
        return 0


def calculate_confidence_score(confidences: List[float]) -> float:
    """
    Calculate average confidence score.

    Args:
        confidences: List of confidence values

    Returns:
        Average confidence score
    """
    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries recursively.

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested keys
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions on failure.

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s..."
                        )
                        import time

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")

            raise last_exception

        return wrapper

    return decorator


def rate_limit(calls: int, period: float):
    """
    Decorator for rate limiting function calls.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds

    Returns:
        Decorator function
    """

    def decorator(func):
        call_times = []

        def wrapper(*args, **kwargs):
            import time

            now = time.time()

            # Remove old calls outside the period
            call_times[:] = [t for t in call_times if now - t < period]

            # Check if we're at the limit
            if len(call_times) >= calls:
                sleep_time = period - (now - call_times[0])
                if sleep_time > 0:
                    logger.warning(
                        f"Rate limit reached. Sleeping for {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)

            call_times.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging.

    Args:
        data: Data to mask
        mask_char: Character to use for masking
        visible_chars: Number of characters to keep visible at start and end

    Returns:
        Masked data string
    """
    if len(data) <= visible_chars * 2:
        return mask_char * len(data)

    start = data[:visible_chars]
    end = data[-visible_chars:]
    middle = mask_char * (len(data) - visible_chars * 2)

    return f"{start}{middle}{end}"


def parse_time_string(time_str: str) -> Optional[datetime]:
    """
    Parse time string in various formats.

    Args:
        time_str: Time string to parse

    Returns:
        datetime object or None
    """
    time_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%H:%M:%S",
        "%H:%M",
    ]

    for fmt in time_formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    return None


def is_json_serializable(obj: Any) -> bool:
    """
    Check if object is JSON serializable.

    Args:
        obj: Object to check

    Returns:
        True if serializable, False otherwise
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text with HTML tags

    Returns:
        Clean text without HTML tags
    """
    import re

    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.

    Args:
        text: Text to search for URLs

    Returns:
        List of URLs found
    """
    url_pattern = r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?"
    return re.findall(url_pattern, text)


def validate_json_schema(data: Dict, schema: Dict) -> tuple[bool, List[str]]:
    """
    Validate JSON data against a schema.

    Args:
        data: Data to validate
        schema: Schema to validate against

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    def validate_field(
        field_name: str, field_data: Any, field_schema: Dict, path: str = ""
    ):
        current_path = f"{path}.{field_name}" if path else field_name

        # Check required fields
        if field_schema.get("required", False) and field_data is None:
            errors.append(f"Required field '{current_path}' is missing")
            return

        if field_data is None:
            return

        # Check type
        expected_type = field_schema.get("type")
        if expected_type:
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }

            expected_python_type = type_map.get(expected_type)
            if expected_python_type and not isinstance(
                field_data, expected_python_type
            ):
                errors.append(
                    f"Field '{current_path}' should be of type {expected_type}"
                )

        # Check nested objects
        if expected_type == "object" and isinstance(field_data, dict):
            nested_schema = field_schema.get("properties", {})
            for nested_field, nested_field_schema in nested_schema.items():
                validate_field(
                    nested_field,
                    field_data.get(nested_field),
                    nested_field_schema,
                    current_path,
                )

    # Validate all fields in schema
    for field_name, field_schema in schema.get("properties", {}).items():
        validate_field(field_name, data.get(field_name), field_schema)

    return len(errors) == 0, errors


# Import secrets for token generation
import secrets
