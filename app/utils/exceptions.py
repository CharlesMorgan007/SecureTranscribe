"""
Custom exceptions for SecureTranscribe application.
Defines specific exception types for different error scenarios.
"""


class SecureTranscribeError(Exception):
    """Base exception for SecureTranscribe application."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert exception to dictionary format."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class AudioProcessingError(SecureTranscribeError):
    """Exception raised for audio processing errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUDIO_PROCESSING_ERROR", details)


class TranscriptionError(SecureTranscribeError):
    """Exception raised for transcription errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "TRANSCRIPTION_ERROR", details)


class DiarizationError(SecureTranscribeError):
    """Exception raised for speaker diarization errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DIARIZATION_ERROR", details)


class SpeakerError(SecureTranscribeError):
    """Exception raised for speaker management errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "SPEAKER_ERROR", details)


class ExportError(SecureTranscribeError):
    """Exception raised for export generation errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "EXPORT_ERROR", details)


class QueueError(SecureTranscribeError):
    """Exception raised for queue management errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "QUEUE_ERROR", details)


class ValidationError(SecureTranscribeError):
    """Exception raised for data validation errors."""

    def __init__(self, message: str, field: str = None, details: dict = None):
        if details is None:
            details = {}
        if field:
            details["field"] = field
        super().__init__(message, "VALIDATION_ERROR", details)


class ConfigurationError(SecureTranscribeError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class DatabaseError(SecureTranscribeError):
    """Exception raised for database operation errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DATABASE_ERROR", details)


class AuthenticationError(SecureTranscribeError):
    """Exception raised for authentication errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(SecureTranscribeError):
    """Exception raised for authorization errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class FileUploadError(SecureTranscribeError):
    """Exception raised for file upload errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "FILE_UPLOAD_ERROR", details)


class ModelLoadError(SecureTranscribeError):
    """Exception raised when AI models fail to load."""

    def __init__(self, message: str, model_name: str = None, details: dict = None):
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, "MODEL_LOAD_ERROR", details)


class GPUNotAvailableError(SecureTranscribeError):
    """Exception raised when GPU is requested but not available."""

    def __init__(self, message: str = "GPU not available", details: dict = None):
        super().__init__(message, "GPU_NOT_AVAILABLE", details)


class ResourceExhaustedError(SecureTranscribeError):
    """Exception raised when system resources are exhausted."""

    def __init__(self, message: str, resource_type: str = None, details: dict = None):
        if details is None:
            details = {}
        if resource_type:
            details["resource_type"] = resource_type
        super().__init__(message, "RESOURCE_EXHAUSTED", details)


class TimeoutError(SecureTranscribeError):
    """Exception raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: int = None, details: dict = None):
        if details is None:
            details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, "TIMEOUT_ERROR", details)


class ExternalServiceError(SecureTranscribeError):
    """Exception raised when external services are unavailable."""

    def __init__(self, message: str, service_name: str = None, details: dict = None):
        if details is None:
            details = {}
        if service_name:
            details["service_name"] = service_name
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", details)
