"""
Exception handlers for the AI-Generated Voice Detection API.

This module provides FastAPI exception handlers for consistent error responses
across all error conditions in the API.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.exceptions import (
    APIError,
    AuthenticationError,
    ValidationError,
    AudioProcessingError,
    DetectionEngineError,
    ModelLoadingError,
    UnsupportedLanguageError,
    InvalidAudioFormatError,
    Base64DecodingError,
    MP3ValidationError,
    FeatureExtractionError,
    InferenceError,
    ConfigurationError,
    RateLimitError,
    ServiceUnavailableError
)
from src.models.schemas import ErrorResponse

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handling for the API.
    
    Provides methods to format error responses consistently and securely.
    """
    
    @staticmethod
    def create_error_response(
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """
        Create a standardized error response.
        
        Args:
            message: Error message for the user
            status_code: HTTP status code
            error_code: Internal error code for debugging
            request_id: Request ID for tracing
            
        Returns:
            JSONResponse with error details
        """
        # Create error response content
        error_content = ErrorResponse(message=message).model_dump()
        
        # Add optional fields for debugging (not exposed to client)
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if error_code:
            headers["X-Error-Code"] = error_code
            
        return JSONResponse(
            status_code=status_code,
            content=error_content,
            headers=headers
        )
    
    @staticmethod
    def sanitize_error_message(message: str, is_production: bool = True) -> str:
        """
        Sanitize error messages to prevent information disclosure.
        
        Args:
            message: Original error message
            is_production: Whether running in production mode
            
        Returns:
            Sanitized error message
        """
        if not is_production:
            return message
            
        # In production, sanitize sensitive information
        sensitive_patterns = [
            "file not found",
            "permission denied",
            "connection refused",
            "timeout",
            "internal server error"
        ]
        
        message_lower = message.lower()
        for pattern in sensitive_patterns:
            if pattern in message_lower:
                return "An internal error occurred. Please try again later."
                
        return message
    
    @staticmethod
    def log_error(
        error: Exception,
        request: Request,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log error details for debugging and monitoring.
        
        Args:
            error: The exception that occurred
            request: The HTTP request that caused the error
            additional_context: Additional context for logging
        """
        context = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        if additional_context:
            context.update(additional_context)
            
        # Remove sensitive information from headers
        if "x-api-key" in context["headers"]:
            context["headers"]["x-api-key"] = "***REDACTED***"
            
        logger.error(
            f"API Error: {type(error).__name__}: {str(error)}",
            extra={"context": context},
            exc_info=True
        )


# Exception handler functions
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """
    Handle custom API errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The API error exception
        
    Returns:
        JSONResponse with error details
    """
    ErrorHandler.log_error(exc, request, {"error_code": exc.error_code})
    
    # Sanitize message for production
    sanitized_message = ErrorHandler.sanitize_error_message(exc.message)
    
    return ErrorHandler.create_error_response(
        message=sanitized_message,
        status_code=exc.status_code,
        error_code=exc.error_code
    )


async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    """
    Handle authentication errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The authentication error exception
        
    Returns:
        JSONResponse with authentication error
    """
    ErrorHandler.log_error(exc, request)
    
    return ErrorHandler.create_error_response(
        message=exc.message,
        status_code=401,
        error_code="AUTHENTICATION_ERROR"
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle validation errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The validation error exception
        
    Returns:
        JSONResponse with validation error
    """
    ErrorHandler.log_error(exc, request)
    
    return ErrorHandler.create_error_response(
        message=exc.message,
        status_code=400,
        error_code="VALIDATION_ERROR"
    )


async def audio_processing_error_handler(request: Request, exc: AudioProcessingError) -> JSONResponse:
    """
    Handle audio processing errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The audio processing error exception
        
    Returns:
        JSONResponse with audio processing error
    """
    ErrorHandler.log_error(exc, request, {"processing_stage": exc.details.get("processing_stage")})
    
    return ErrorHandler.create_error_response(
        message=exc.message,
        status_code=422,
        error_code="AUDIO_PROCESSING_ERROR"
    )


async def detection_engine_error_handler(request: Request, exc: DetectionEngineError) -> JSONResponse:
    """
    Handle detection engine errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The detection engine error exception
        
    Returns:
        JSONResponse with detection engine error
    """
    ErrorHandler.log_error(exc, request, {"language": exc.details.get("language")})
    
    # Sanitize message to avoid exposing internal details
    sanitized_message = "Voice detection analysis failed. Please try again."
    
    return ErrorHandler.create_error_response(
        message=sanitized_message,
        status_code=500,
        error_code="DETECTION_ENGINE_ERROR"
    )


async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic request validation errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The request validation error exception
        
    Returns:
        JSONResponse with validation error details
    """
    ErrorHandler.log_error(exc, request)
    
    # Format validation errors into user-friendly messages
    error_messages = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_type = error["type"]
        
        # Create user-friendly error messages
        if error_type == "missing":
            error_messages.append(f"Missing required field: {field}")
        elif error_type == "value_error":
            error_messages.append(f"Invalid value for {field}: {message}")
        elif error_type == "type_error":
            error_messages.append(f"Invalid type for {field}: {message}")
        else:
            error_messages.append(f"{field}: {message}")
    
    combined_message = "Validation error: " + "; ".join(error_messages)
    
    return ErrorHandler.create_error_response(
        message=combined_message,
        status_code=400,
        error_code="REQUEST_VALIDATION_ERROR"
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTP exceptions.
    
    Args:
        request: The HTTP request that caused the error
        exc: The HTTP exception
        
    Returns:
        JSONResponse with HTTP error
    """
    ErrorHandler.log_error(exc, request)
    
    return ErrorHandler.create_error_response(
        message=exc.detail,
        status_code=exc.status_code,
        error_code="HTTP_ERROR"
    )


async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle Starlette HTTP exceptions.
    
    Args:
        request: The HTTP request that caused the error
        exc: The Starlette HTTP exception
        
    Returns:
        JSONResponse with HTTP error
    """
    ErrorHandler.log_error(exc, request)
    
    return ErrorHandler.create_error_response(
        message=exc.detail,
        status_code=exc.status_code,
        error_code="HTTP_ERROR"
    )


async def rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """
    Handle rate limit errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The rate limit error exception
        
    Returns:
        JSONResponse with rate limit error
    """
    ErrorHandler.log_error(exc, request)
    
    headers = {}
    retry_after = exc.details.get("retry_after")
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    
    response = ErrorHandler.create_error_response(
        message=exc.message,
        status_code=429,
        error_code="RATE_LIMIT_ERROR"
    )
    
    # Add retry-after header
    for key, value in headers.items():
        response.headers[key] = value
        
    return response


async def service_unavailable_error_handler(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
    """
    Handle service unavailable errors.
    
    Args:
        request: The HTTP request that caused the error
        exc: The service unavailable error exception
        
    Returns:
        JSONResponse with service unavailable error
    """
    ErrorHandler.log_error(exc, request)
    
    headers = {}
    retry_after = exc.details.get("retry_after")
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    
    response = ErrorHandler.create_error_response(
        message=exc.message,
        status_code=503,
        error_code="SERVICE_UNAVAILABLE_ERROR"
    )
    
    # Add retry-after header
    for key, value in headers.items():
        response.headers[key] = value
        
    return response


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Args:
        request: The HTTP request that caused the error
        exc: The unexpected exception
        
    Returns:
        JSONResponse with generic error message
    """
    ErrorHandler.log_error(exc, request)
    
    # Return generic error message to avoid information disclosure
    return ErrorHandler.create_error_response(
        message="An internal server error occurred. Please try again later.",
        status_code=500,
        error_code="INTERNAL_SERVER_ERROR"
    )


# Dictionary mapping exception types to their handlers
EXCEPTION_HANDLERS = {
    APIError: api_error_handler,
    AuthenticationError: authentication_error_handler,
    ValidationError: validation_error_handler,
    AudioProcessingError: audio_processing_error_handler,
    DetectionEngineError: detection_engine_error_handler,
    ModelLoadingError: detection_engine_error_handler,  # Use detection engine handler
    UnsupportedLanguageError: validation_error_handler,  # Use validation handler
    InvalidAudioFormatError: validation_error_handler,  # Use validation handler
    Base64DecodingError: audio_processing_error_handler,  # Use audio processing handler
    MP3ValidationError: audio_processing_error_handler,  # Use audio processing handler
    FeatureExtractionError: audio_processing_error_handler,  # Use audio processing handler
    InferenceError: detection_engine_error_handler,  # Use detection engine handler
    ConfigurationError: api_error_handler,  # Use generic API error handler
    RateLimitError: rate_limit_error_handler,
    ServiceUnavailableError: service_unavailable_error_handler,
    RequestValidationError: request_validation_error_handler,
    HTTPException: http_exception_handler,
    StarletteHTTPException: starlette_http_exception_handler,
    Exception: general_exception_handler,  # Catch-all for unexpected errors
}