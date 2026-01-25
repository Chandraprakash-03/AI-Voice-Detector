"""
Custom exception classes for the AI-Generated Voice Detection API.

This module defines specific exception types for different error categories
to enable precise error handling and consistent error responses.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class APIError(Exception):
    """
    Base exception class for all API-related errors.
    
    Provides common functionality for error handling including
    HTTP status codes, error messages, and logging.
    """
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize API error.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code for the error
            error_code: Internal error code for categorization
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        
        # Log the error for debugging
        logger.error(
            f"{self.error_code}: {message} (status: {status_code})",
            extra={"error_details": self.details}
        )


class AuthenticationError(APIError):
    """
    Exception for authentication-related errors.
    
    Raised when API key validation fails or authentication is missing.
    """
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class ValidationError(APIError):
    """
    Exception for request validation errors.
    
    Raised when request data fails validation checks.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
            
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class AudioProcessingError(APIError):
    """
    Exception for audio processing errors.
    
    Raised when audio decoding, validation, or feature extraction fails.
    """
    
    def __init__(self, message: str, processing_stage: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if processing_stage:
            error_details["processing_stage"] = processing_stage
            
        super().__init__(
            message=message,
            status_code=422,
            error_code="AUDIO_PROCESSING_ERROR",
            details=error_details
        )


class DetectionEngineError(APIError):
    """
    Exception for ML detection engine errors.
    
    Raised when model loading, inference, or classification fails.
    """
    
    def __init__(self, message: str, language: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if language:
            error_details["language"] = language
            
        super().__init__(
            message=message,
            status_code=500,
            error_code="DETECTION_ENGINE_ERROR",
            details=error_details
        )


class ModelLoadingError(DetectionEngineError):
    """
    Exception for model loading failures.
    
    Raised when ML models cannot be loaded or initialized.
    """
    
    def __init__(self, message: str, language: Optional[str] = None, model_path: Optional[str] = None):
        details = {}
        if model_path:
            details["model_path"] = model_path
            
        super().__init__(
            message=message,
            language=language,
            details=details
        )
        self.error_code = "MODEL_LOADING_ERROR"


class UnsupportedLanguageError(ValidationError):
    """
    Exception for unsupported language requests.
    
    Raised when a request specifies a language not in the supported list.
    """
    
    def __init__(self, language: str, supported_languages: list):
        message = (
            f"Unsupported language: '{language}'. "
            f"Supported languages are: {', '.join(supported_languages)}"
        )
        super().__init__(
            message=message,
            field="language",
            details={
                "provided_language": language,
                "supported_languages": supported_languages
            }
        )
        self.error_code = "UNSUPPORTED_LANGUAGE_ERROR"


class InvalidAudioFormatError(ValidationError):
    """
    Exception for invalid audio format requests.
    
    Raised when audio format is not supported.
    """
    
    def __init__(self, format_provided: str, supported_formats: list = None):
        supported_formats = supported_formats or ["mp3"]
        message = (
            f"Invalid audio format: '{format_provided}'. "
            f"Supported formats are: {', '.join(supported_formats)}"
        )
        super().__init__(
            message=message,
            field="audioFormat",
            details={
                "provided_format": format_provided,
                "supported_formats": supported_formats
            }
        )
        self.error_code = "INVALID_AUDIO_FORMAT_ERROR"


class Base64DecodingError(AudioProcessingError):
    """
    Exception for base64 decoding failures.
    
    Raised when base64 audio data cannot be decoded.
    """
    
    def __init__(self, message: str = "Invalid base64 encoding in audioBase64 field"):
        super().__init__(
            message=message,
            processing_stage="base64_decoding"
        )
        self.error_code = "BASE64_DECODING_ERROR"


class MP3ValidationError(AudioProcessingError):
    """
    Exception for MP3 format validation failures.
    
    Raised when decoded audio data is not valid MP3 format.
    """
    
    def __init__(self, message: str = "Audio file is corrupted or not a valid MP3 format"):
        super().__init__(
            message=message,
            processing_stage="mp3_validation"
        )
        self.error_code = "MP3_VALIDATION_ERROR"


class FeatureExtractionError(AudioProcessingError):
    """
    Exception for audio feature extraction failures.
    
    Raised when audio feature extraction fails.
    """
    
    def __init__(self, message: str, feature_type: Optional[str] = None):
        details = {}
        if feature_type:
            details["feature_type"] = feature_type
            
        super().__init__(
            message=message,
            processing_stage="feature_extraction",
            details=details
        )
        self.error_code = "FEATURE_EXTRACTION_ERROR"


class InferenceError(DetectionEngineError):
    """
    Exception for ML model inference failures.
    
    Raised when model prediction fails.
    """
    
    def __init__(self, message: str, language: Optional[str] = None, model_version: Optional[str] = None):
        details = {}
        if model_version:
            details["model_version"] = model_version
            
        super().__init__(
            message=message,
            language=language,
            details=details
        )
        self.error_code = "INFERENCE_ERROR"


class ConfigurationError(APIError):
    """
    Exception for system configuration errors.
    
    Raised when system configuration is invalid or missing.
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
            
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class RateLimitError(APIError):
    """
    Exception for rate limiting errors.
    
    Raised when API rate limits are exceeded.
    """
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
            
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_ERROR",
            details=details
        )


class ServiceUnavailableError(APIError):
    """
    Exception for service unavailability.
    
    Raised when the service is temporarily unavailable.
    """
    
    def __init__(self, message: str = "Service temporarily unavailable", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
            
        super().__init__(
            message=message,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE_ERROR",
            details=details
        )