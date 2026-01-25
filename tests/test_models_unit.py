"""
Unit tests for Pydantic models.

This module contains unit tests that validate specific examples and edge cases
for the AI-Generated Voice Detection API data models.
"""

import pytest
from pydantic import ValidationError
import base64
import json

from src.models.schemas import VoiceDetectionRequest, SuccessResponse, ErrorResponse


class TestVoiceDetectionRequest:
    """Unit tests for VoiceDetectionRequest model."""
    
    def test_valid_request_creation(self):
        """
        Test valid request model creation with all supported languages.
        **Validates: Requirements 1.2, 1.4, 4.1**
        """
        valid_base64 = base64.b64encode(b"test audio data").decode('ascii')
        
        # Test each supported language
        supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        
        for language in supported_languages:
            request_data = {
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": valid_base64
            }
            
            request = VoiceDetectionRequest(**request_data)
            
            assert request.language == language
            assert request.audioFormat == "mp3"
            assert request.audioBase64 == valid_base64
    
    def test_missing_language_field(self):
        """
        Test validation error for missing language field.
        **Validates: Requirements 1.2, 4.2**
        """
        request_data = {
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        errors = exc_info.value.errors()
        language_errors = [e for e in errors if e['loc'] == ('language',)]
        assert len(language_errors) > 0
        assert language_errors[0]['type'] == 'missing'
    
    def test_invalid_language_values(self):
        """
        Test validation errors for invalid language values.
        **Validates: Requirements 4.1, 4.2**
        """
        invalid_languages = ["French", "German", "Spanish", "chinese", "ENGLISH", ""]
        valid_base64 = base64.b64encode(b"test audio data").decode('ascii')
        
        for invalid_language in invalid_languages:
            request_data = {
                "language": invalid_language,
                "audioFormat": "mp3",
                "audioBase64": valid_base64
            }
            
            with pytest.raises(ValidationError) as exc_info:
                VoiceDetectionRequest(**request_data)
            
            errors = exc_info.value.errors()
            language_errors = [e for e in errors if e['loc'] == ('language',)]
            assert len(language_errors) > 0
    
    def test_missing_audio_format_field(self):
        """
        Test validation error for missing audioFormat field.
        **Validates: Requirements 1.2**
        """
        request_data = {
            "language": "English",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        errors = exc_info.value.errors()
        format_errors = [e for e in errors if e['loc'] == ('audioFormat',)]
        assert len(format_errors) > 0
        assert format_errors[0]['type'] == 'missing'
    
    def test_invalid_audio_format_values(self):
        """
        Test validation errors for invalid audioFormat values.
        **Validates: Requirements 1.4**
        """
        invalid_formats = ["wav", "flac", "ogg", "m4a", "MP3", ""]
        valid_base64 = base64.b64encode(b"test audio data").decode('ascii')
        
        for invalid_format in invalid_formats:
            request_data = {
                "language": "English",
                "audioFormat": invalid_format,
                "audioBase64": valid_base64
            }
            
            with pytest.raises(ValidationError) as exc_info:
                VoiceDetectionRequest(**request_data)
            
            errors = exc_info.value.errors()
            format_errors = [e for e in errors if e['loc'] == ('audioFormat',)]
            assert len(format_errors) > 0
    
    def test_missing_audio_base64_field(self):
        """
        Test validation error for missing audioBase64 field.
        **Validates: Requirements 1.4**
        """
        request_data = {
            "language": "English",
            "audioFormat": "mp3"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        errors = exc_info.value.errors()
        base64_errors = [e for e in errors if e['loc'] == ('audioBase64',)]
        assert len(base64_errors) > 0
        assert base64_errors[0]['type'] == 'missing'
    
    def test_invalid_base64_values(self):
        """
        Test validation errors for invalid base64 values.
        **Validates: Requirements 1.4**
        """
        invalid_base64_values = [
            "",  # Empty string
            "   ",  # Whitespace only
            "invalid!@#$%",  # Invalid characters
            "abc",  # Too short/invalid padding
            "not_base64_at_all",  # Invalid format
        ]
        
        for invalid_base64 in invalid_base64_values:
            request_data = {
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": invalid_base64
            }
            
            with pytest.raises(ValidationError) as exc_info:
                VoiceDetectionRequest(**request_data)
            
            errors = exc_info.value.errors()
            base64_errors = [e for e in errors if e['loc'] == ('audioBase64',)]
            assert len(base64_errors) > 0
    
    def test_valid_base64_accepted(self):
        """
        Test that valid base64 strings are accepted.
        **Validates: Requirements 1.4**
        """
        # Test various valid base64 strings
        valid_base64_strings = [
            base64.b64encode(b"test").decode('ascii'),
            base64.b64encode(b"longer test data with more content").decode('ascii'),
            base64.b64encode(b"binary data \x00\x01\x02\x03").decode('ascii'),
        ]
        
        for valid_base64 in valid_base64_strings:
            request_data = {
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": valid_base64
            }
            
            # Should not raise ValidationError
            request = VoiceDetectionRequest(**request_data)
            assert request.audioBase64 == valid_base64


class TestSuccessResponse:
    """Unit tests for SuccessResponse model."""
    
    def test_valid_success_response_creation(self):
        """
        Test valid success response model creation.
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
        """
        response_data = {
            "language": "English",
            "classification": "HUMAN",
            "confidenceScore": 0.87,
            "explanation": "The audio exhibits natural speech patterns."
        }
        
        response = SuccessResponse(**response_data)
        
        assert response.status == "success"  # Default value
        assert response.language == "English"
        assert response.classification == "HUMAN"
        assert response.confidenceScore == 0.87
        assert response.explanation == "The audio exhibits natural speech patterns."
    
    def test_valid_classification_values(self):
        """
        Test both valid classification values.
        **Validates: Requirements 6.3**
        """
        classifications = ["AI_GENERATED", "HUMAN"]
        
        for classification in classifications:
            response_data = {
                "language": "Tamil",
                "classification": classification,
                "confidenceScore": 0.75,
                "explanation": "Test explanation"
            }
            
            response = SuccessResponse(**response_data)
            assert response.classification == classification
    
    def test_invalid_classification_values(self):
        """
        Test validation errors for invalid classification values.
        **Validates: Requirements 6.3**
        """
        invalid_classifications = ["ai_generated", "human", "UNKNOWN", "MAYBE", ""]
        
        for invalid_classification in invalid_classifications:
            response_data = {
                "language": "English",
                "classification": invalid_classification,
                "confidenceScore": 0.5,
                "explanation": "Test explanation"
            }
            
            with pytest.raises(ValidationError) as exc_info:
                SuccessResponse(**response_data)
            
            errors = exc_info.value.errors()
            classification_errors = [e for e in errors if e['loc'] == ('classification',)]
            assert len(classification_errors) > 0
    
    def test_confidence_score_boundaries(self):
        """
        Test confidence score boundary validation.
        **Validates: Requirements 6.4**
        """
        # Valid boundary values
        valid_scores = [0.0, 0.5, 1.0, 0.001, 0.999]
        
        for score in valid_scores:
            response_data = {
                "language": "Hindi",
                "classification": "HUMAN",
                "confidenceScore": score,
                "explanation": "Test explanation"
            }
            
            response = SuccessResponse(**response_data)
            assert response.confidenceScore == score
        
        # Invalid boundary values
        invalid_scores = [-0.1, 1.1, -1.0, 2.0, 100.0]
        
        for score in invalid_scores:
            response_data = {
                "language": "Hindi",
                "classification": "HUMAN",
                "confidenceScore": score,
                "explanation": "Test explanation"
            }
            
            with pytest.raises(ValidationError) as exc_info:
                SuccessResponse(**response_data)
            
            errors = exc_info.value.errors()
            score_errors = [e for e in errors if e['loc'] == ('confidenceScore',)]
            assert len(score_errors) > 0
    
    def test_empty_explanation_rejected(self):
        """
        Test that empty explanations are rejected.
        **Validates: Requirements 6.5**
        """
        response_data = {
            "language": "Malayalam",
            "classification": "AI_GENERATED",
            "confidenceScore": 0.9,
            "explanation": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SuccessResponse(**response_data)
        
        errors = exc_info.value.errors()
        explanation_errors = [e for e in errors if e['loc'] == ('explanation',)]
        assert len(explanation_errors) > 0
    
    def test_response_serialization(self):
        """
        Test response model serialization to JSON.
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
        """
        response_data = {
            "language": "Telugu",
            "classification": "AI_GENERATED",
            "confidenceScore": 0.92,
            "explanation": "The audio shows characteristics typical of AI-generated speech."
        }
        
        response = SuccessResponse(**response_data)
        
        # Test JSON serialization
        json_data = response.model_dump()
        
        assert json_data["status"] == "success"
        assert json_data["language"] == "Telugu"
        assert json_data["classification"] == "AI_GENERATED"
        assert json_data["confidenceScore"] == 0.92
        assert json_data["explanation"] == "The audio shows characteristics typical of AI-generated speech."


class TestErrorResponse:
    """Unit tests for ErrorResponse model."""
    
    def test_valid_error_response_creation(self):
        """
        Test valid error response model creation.
        **Validates: Requirements 6.6**
        """
        response_data = {
            "message": "Invalid language specified"
        }
        
        response = ErrorResponse(**response_data)
        
        assert response.status == "error"  # Default value
        assert response.message == "Invalid language specified"
    
    def test_empty_message_rejected(self):
        """
        Test that empty error messages are rejected.
        **Validates: Requirements 6.6**
        """
        response_data = {
            "message": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(**response_data)
        
        errors = exc_info.value.errors()
        message_errors = [e for e in errors if e['loc'] == ('message',)]
        assert len(message_errors) > 0
    
    def test_missing_message_field(self):
        """
        Test validation error for missing message field.
        **Validates: Requirements 6.6**
        """
        response_data = {}
        
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(**response_data)
        
        errors = exc_info.value.errors()
        message_errors = [e for e in errors if e['loc'] == ('message',)]
        assert len(message_errors) > 0
        assert message_errors[0]['type'] == 'missing'
    
    def test_error_response_serialization(self):
        """
        Test error response model serialization to JSON.
        **Validates: Requirements 6.6**
        """
        response_data = {
            "message": "Missing required field: audioBase64"
        }
        
        response = ErrorResponse(**response_data)
        
        # Test JSON serialization
        json_data = response.model_dump()
        
        assert json_data["status"] == "error"
        assert json_data["message"] == "Missing required field: audioBase64"
    
    def test_various_error_messages(self):
        """
        Test various types of error messages.
        **Validates: Requirements 6.6**
        """
        error_messages = [
            "Invalid language specified",
            "Missing required field: language",
            "Invalid base64 encoding in audioBase64 field",
            "Audio format must be 'mp3'",
            "Internal server error occurred"
        ]
        
        for message in error_messages:
            response_data = {"message": message}
            response = ErrorResponse(**response_data)
            
            assert response.status == "error"
            assert response.message == message