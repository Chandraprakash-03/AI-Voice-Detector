"""
Unit tests for error handling system.

Tests each exception type, handler, error message content, and HTTP status code correctness.
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from fastapi import Request
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError

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
from src.error_handlers import (
    ErrorHandler,
    api_error_handler,
    authentication_error_handler,
    validation_error_handler,
    audio_processing_error_handler,
    detection_engine_error_handler,
    request_validation_error_handler,
    http_exception_handler,
    rate_limit_error_handler,
    service_unavailable_error_handler,
    general_exception_handler
)
from src.main import app

client = TestClient(app)


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_api_error_initialization(self):
        """Test APIError initialization with all parameters."""
        error = APIError(
            message="Test error",
            status_code=400,
            error_code="TEST_ERROR",
            details={"field": "value"}
        )
        
        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"field": "value"}
        assert str(error) == "Test error"
    
    def test_api_error_default_values(self):
        """Test APIError with default values."""
        error = APIError("Test error")
        
        assert error.message == "Test error"
        assert error.status_code == 500
        assert error.error_code == "APIError"
        assert error.details == {}
    
    def test_authentication_error(self):
        """Test AuthenticationError initialization."""
        error = AuthenticationError("Invalid credentials")
        
        assert error.message == "Invalid credentials"
        assert error.status_code == 401
        assert error.error_code == "AUTHENTICATION_ERROR"
    
    def test_authentication_error_default_message(self):
        """Test AuthenticationError with default message."""
        error = AuthenticationError()
        
        assert error.message == "Authentication failed"
        assert error.status_code == 401
    
    def test_validation_error(self):
        """Test ValidationError initialization."""
        error = ValidationError("Invalid field", field="username")
        
        assert error.message == "Invalid field"
        assert error.status_code == 400
        assert error.error_code == "VALIDATION_ERROR"
        assert error.details["field"] == "username"
    
    def test_audio_processing_error(self):
        """Test AudioProcessingError initialization."""
        error = AudioProcessingError("Processing failed", processing_stage="decoding")
        
        assert error.message == "Processing failed"
        assert error.status_code == 422
        assert error.error_code == "AUDIO_PROCESSING_ERROR"
        assert error.details["processing_stage"] == "decoding"
    
    def test_detection_engine_error(self):
        """Test DetectionEngineError initialization."""
        error = DetectionEngineError("Model failed", language="English")
        
        assert error.message == "Model failed"
        assert error.status_code == 500
        assert error.error_code == "DETECTION_ENGINE_ERROR"
        assert error.details["language"] == "English"
    
    def test_model_loading_error(self):
        """Test ModelLoadingError initialization."""
        error = ModelLoadingError("Model not found", language="Tamil", model_path="/path/to/model")
        
        assert error.message == "Model not found"
        assert error.status_code == 500
        assert error.error_code == "MODEL_LOADING_ERROR"
        assert error.details["language"] == "Tamil"
        assert error.details["model_path"] == "/path/to/model"
    
    def test_unsupported_language_error(self):
        """Test UnsupportedLanguageError initialization."""
        supported = ["English", "Tamil"]
        error = UnsupportedLanguageError("French", supported)
        
        assert "French" in error.message
        assert "English, Tamil" in error.message
        assert error.status_code == 400
        assert error.error_code == "UNSUPPORTED_LANGUAGE_ERROR"
        assert error.details["provided_language"] == "French"
        assert error.details["supported_languages"] == supported
    
    def test_invalid_audio_format_error(self):
        """Test InvalidAudioFormatError initialization."""
        error = InvalidAudioFormatError("wav", ["mp3"])
        
        assert "wav" in error.message
        assert "mp3" in error.message
        assert error.status_code == 400
        assert error.error_code == "INVALID_AUDIO_FORMAT_ERROR"
        assert error.details["provided_format"] == "wav"
        assert error.details["supported_formats"] == ["mp3"]
    
    def test_base64_decoding_error(self):
        """Test Base64DecodingError initialization."""
        error = Base64DecodingError("Invalid base64")
        
        assert error.message == "Invalid base64"
        assert error.status_code == 422
        assert error.error_code == "BASE64_DECODING_ERROR"
        assert error.details["processing_stage"] == "base64_decoding"
    
    def test_mp3_validation_error(self):
        """Test MP3ValidationError initialization."""
        error = MP3ValidationError("Not MP3 format")
        
        assert error.message == "Not MP3 format"
        assert error.status_code == 422
        assert error.error_code == "MP3_VALIDATION_ERROR"
        assert error.details["processing_stage"] == "mp3_validation"
    
    def test_feature_extraction_error(self):
        """Test FeatureExtractionError initialization."""
        error = FeatureExtractionError("MFCC failed", feature_type="mfcc")
        
        assert error.message == "MFCC failed"
        assert error.status_code == 422
        assert error.error_code == "FEATURE_EXTRACTION_ERROR"
        assert error.details["processing_stage"] == "feature_extraction"
        assert error.details["feature_type"] == "mfcc"
    
    def test_inference_error(self):
        """Test InferenceError initialization."""
        error = InferenceError("Prediction failed", language="Hindi", model_version="1.0")
        
        assert error.message == "Prediction failed"
        assert error.status_code == 500
        assert error.error_code == "INFERENCE_ERROR"
        assert error.details["language"] == "Hindi"
        assert error.details["model_version"] == "1.0"
    
    def test_configuration_error(self):
        """Test ConfigurationError initialization."""
        error = ConfigurationError("Missing config", config_key="api_key")
        
        assert error.message == "Missing config"
        assert error.status_code == 500
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.details["config_key"] == "api_key"
    
    def test_rate_limit_error(self):
        """Test RateLimitError initialization."""
        error = RateLimitError("Too many requests", retry_after=60)
        
        assert error.message == "Too many requests"
        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.details["retry_after"] == 60
    
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError initialization."""
        error = ServiceUnavailableError("Service down", retry_after=120)
        
        assert error.message == "Service down"
        assert error.status_code == 503
        assert error.error_code == "SERVICE_UNAVAILABLE_ERROR"
        assert error.details["retry_after"] == 120


class TestErrorHandler:
    """Test ErrorHandler utility class."""
    
    def test_create_error_response(self):
        """Test creating standardized error response."""
        response = ErrorHandler.create_error_response(
            message="Test error",
            status_code=400,
            error_code="TEST_ERROR",
            request_id="req-123"
        )
        
        assert response.status_code == 400
        assert response.body == b'{"status":"error","message":"Test error"}'
        assert response.headers["X-Request-ID"] == "req-123"
        assert response.headers["X-Error-Code"] == "TEST_ERROR"
    
    def test_create_error_response_minimal(self):
        """Test creating error response with minimal parameters."""
        response = ErrorHandler.create_error_response("Simple error")
        
        assert response.status_code == 500
        assert response.body == b'{"status":"error","message":"Simple error"}'
    
    def test_sanitize_error_message_production(self):
        """Test error message sanitization in production mode."""
        message = "File not found: /secret/path/file.txt"
        sanitized = ErrorHandler.sanitize_error_message(message, is_production=True)
        
        assert sanitized == "An internal error occurred. Please try again later."
    
    def test_sanitize_error_message_development(self):
        """Test error message sanitization in development mode."""
        message = "File not found: /secret/path/file.txt"
        sanitized = ErrorHandler.sanitize_error_message(message, is_production=False)
        
        assert sanitized == message  # Should not be sanitized
    
    def test_sanitize_error_message_safe_message(self):
        """Test that safe messages are not sanitized."""
        message = "Invalid input format"
        sanitized = ErrorHandler.sanitize_error_message(message, is_production=True)
        
        assert sanitized == message
    
    @patch('src.error_handlers.logger')
    def test_log_error(self, mock_logger):
        """Test error logging functionality."""
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://test.com/api/test"
        mock_request.headers = {"x-api-key": "secret-key", "content-type": "application/json"}
        
        error = ValueError("Test error")
        ErrorHandler.log_error(error, mock_request, {"extra": "context"})
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        
        # Check that sensitive information is redacted
        context = call_args[1]["extra"]["context"]
        assert context["headers"]["x-api-key"] == "***REDACTED***"
        assert context["method"] == "POST"
        assert context["error_type"] == "ValueError"
        assert context["extra"] == "context"


class TestExceptionHandlers:
    """Test exception handler functions."""
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = "http://test.com/api/test"
        request.headers = {"content-type": "application/json"}
        return request
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_api_error_handler(self, mock_log, mock_request):
        """Test API error handler."""
        error = APIError("Test API error", status_code=400, error_code="TEST_ERROR")
        response = await api_error_handler(mock_request, error)
        
        assert response.status_code == 400
        assert b"Test API error" in response.body
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_authentication_error_handler(self, mock_log, mock_request):
        """Test authentication error handler."""
        error = AuthenticationError("Invalid API key")
        response = await authentication_error_handler(mock_request, error)
        
        assert response.status_code == 401
        assert b"Invalid API key" in response.body
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_validation_error_handler(self, mock_log, mock_request):
        """Test validation error handler."""
        error = ValidationError("Invalid field", field="username")
        response = await validation_error_handler(mock_request, error)
        
        assert response.status_code == 400
        assert b"Invalid field" in response.body
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_audio_processing_error_handler(self, mock_log, mock_request):
        """Test audio processing error handler."""
        error = AudioProcessingError("Processing failed", processing_stage="decoding")
        response = await audio_processing_error_handler(mock_request, error)
        
        assert response.status_code == 422
        assert b"Processing failed" in response.body
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_detection_engine_error_handler(self, mock_log, mock_request):
        """Test detection engine error handler."""
        error = DetectionEngineError("Model failed", language="English")
        response = await detection_engine_error_handler(mock_request, error)
        
        assert response.status_code == 500
        # Should be sanitized message, not the original
        assert b"Voice detection analysis failed" in response.body
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_request_validation_error_handler(self, mock_log, mock_request):
        """Test request validation error handler."""
        # Create a mock RequestValidationError
        mock_error = Mock(spec=RequestValidationError)
        mock_error.errors.return_value = [
            {"loc": ("field1",), "msg": "field required", "type": "missing"},
            {"loc": ("field2",), "msg": "invalid value", "type": "value_error"}
        ]
        
        response = await request_validation_error_handler(mock_request, mock_error)
        
        assert response.status_code == 400
        assert b"Missing required field: field1" in response.body
        assert b"Invalid value for field2" in response.body
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_rate_limit_error_handler(self, mock_log, mock_request):
        """Test rate limit error handler."""
        error = RateLimitError("Too many requests", retry_after=60)
        response = await rate_limit_error_handler(mock_request, error)
        
        assert response.status_code == 429
        assert b"Too many requests" in response.body
        assert response.headers["Retry-After"] == "60"
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_service_unavailable_error_handler(self, mock_log, mock_request):
        """Test service unavailable error handler."""
        error = ServiceUnavailableError("Service down", retry_after=120)
        response = await service_unavailable_error_handler(mock_request, error)
        
        assert response.status_code == 503
        assert b"Service down" in response.body
        assert response.headers["Retry-After"] == "120"
        mock_log.assert_called_once()
    
    @patch('src.error_handlers.ErrorHandler.log_error')
    async def test_general_exception_handler(self, mock_log, mock_request):
        """Test general exception handler."""
        error = ValueError("Unexpected error")
        response = await general_exception_handler(mock_request, error)
        
        assert response.status_code == 500
        assert b"An internal server error occurred" in response.body
        mock_log.assert_called_once()


class TestErrorHandlingIntegration:
    """Test error handling integration with FastAPI."""
    
    def test_authentication_error_integration(self):
        """Test authentication error through API."""
        response = client.post("/api/voice-detection", json={})
        
        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"
        assert "API key" in data["message"]
    
    def test_validation_error_integration(self):
        """Test validation error through API."""
        headers = {"x-api-key": "dev-api-key-12345"}
        # Send invalid JSON structure
        response = client.post("/api/voice-detection", headers=headers, json={})
        
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "Validation error" in data["message"]
    
    def test_invalid_audio_format_integration(self):
        """Test invalid audio format error through API."""
        headers = {"x-api-key": "dev-api-key-12345"}
        payload = {
            "language": "English",
            "audioFormat": "wav",  # Invalid format
            "audioBase64": "dGVzdA=="  # Valid base64
        }
        
        response = client.post("/api/voice-detection", headers=headers, json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "audioFormat" in data["message"]
    
    def test_invalid_language_integration(self):
        """Test invalid language error through API."""
        headers = {"x-api-key": "dev-api-key-12345"}
        payload = {
            "language": "French",  # Unsupported language
            "audioFormat": "mp3",
            "audioBase64": "dGVzdA=="
        }
        
        response = client.post("/api/voice-detection", headers=headers, json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "language" in data["message"]
    
    def test_invalid_base64_integration(self):
        """Test invalid base64 error through API."""
        headers = {"x-api-key": "dev-api-key-12345"}
        payload = {
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "invalid-base64!"  # Invalid base64
        }
        
        response = client.post("/api/voice-detection", headers=headers, json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "base64" in data["message"]


class TestErrorMessageContent:
    """Test error message content and format."""
    
    def test_authentication_error_message_format(self):
        """Test authentication error message format."""
        error = AuthenticationError("Invalid API key")
        assert "Invalid API key" in str(error)
        assert error.status_code == 401
    
    def test_validation_error_message_format(self):
        """Test validation error message format."""
        error = ValidationError("Field is required", field="username")
        assert "Field is required" in str(error)
        assert error.status_code == 400
    
    def test_audio_processing_error_message_format(self):
        """Test audio processing error message format."""
        error = AudioProcessingError("Invalid MP3 format")
        assert "Invalid MP3 format" in str(error)
        assert error.status_code == 422
    
    def test_detection_engine_error_message_format(self):
        """Test detection engine error message format."""
        error = DetectionEngineError("Model inference failed")
        assert "Model inference failed" in str(error)
        assert error.status_code == 500
    
    def test_unsupported_language_error_message_content(self):
        """Test unsupported language error message content."""
        error = UnsupportedLanguageError("French", ["English", "Tamil"])
        message = str(error)
        assert "French" in message
        assert "English" in message
        assert "Tamil" in message
        assert "Supported languages" in message
    
    def test_invalid_audio_format_error_message_content(self):
        """Test invalid audio format error message content."""
        error = InvalidAudioFormatError("wav", ["mp3"])
        message = str(error)
        assert "wav" in message
        assert "mp3" in message
        assert "Invalid audio format" in message


class TestHTTPStatusCodes:
    """Test HTTP status code correctness."""
    
    def test_authentication_error_status_code(self):
        """Test authentication error returns 401."""
        error = AuthenticationError("Invalid credentials")
        assert error.status_code == 401
    
    def test_validation_error_status_code(self):
        """Test validation error returns 400."""
        error = ValidationError("Invalid input")
        assert error.status_code == 400
    
    def test_audio_processing_error_status_code(self):
        """Test audio processing error returns 422."""
        error = AudioProcessingError("Processing failed")
        assert error.status_code == 422
    
    def test_detection_engine_error_status_code(self):
        """Test detection engine error returns 500."""
        error = DetectionEngineError("Model failed")
        assert error.status_code == 500
    
    def test_rate_limit_error_status_code(self):
        """Test rate limit error returns 429."""
        error = RateLimitError("Too many requests")
        assert error.status_code == 429
    
    def test_service_unavailable_error_status_code(self):
        """Test service unavailable error returns 503."""
        error = ServiceUnavailableError("Service down")
        assert error.status_code == 503
    
    def test_configuration_error_status_code(self):
        """Test configuration error returns 500."""
        error = ConfigurationError("Missing config")
        assert error.status_code == 500


class TestErrorLogging:
    """Test error logging functionality."""
    
    @patch('src.exceptions.logger')
    def test_api_error_logging(self, mock_logger):
        """Test that API errors are logged."""
        error = APIError("Test error", status_code=400, error_code="TEST_ERROR")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        assert "TEST_ERROR" in call_args[0]
        assert "Test error" in call_args[0]
    
    @patch('src.exceptions.logger')
    def test_authentication_error_logging(self, mock_logger):
        """Test that authentication errors are logged."""
        error = AuthenticationError("Invalid API key")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        assert "AUTHENTICATION_ERROR" in call_args[0]
        assert "Invalid API key" in call_args[0]
    
    @patch('src.exceptions.logger')
    def test_detection_engine_error_logging(self, mock_logger):
        """Test that detection engine errors are logged."""
        error = DetectionEngineError("Model failed", language="English")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        assert "DETECTION_ENGINE_ERROR" in call_args[0]
        assert "Model failed" in call_args[0]