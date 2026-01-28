"""
Property-based tests for error response consistency validation.

**Property 7: Error Response Consistency**
**Validates: Requirements 6.6, 7.2, 8.1, 8.4, 8.5**

Tests that for any error condition (malformed JSON, missing fields, invalid values, 
processing failures), the system returns a consistent JSON error response format 
with status "error" and a descriptive message.
"""

import json
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient
import pytest

from src.main import app


# Test client
client = TestClient(app)

# Valid and invalid API keys
VALID_API_KEY = "dev-api-key-12345"
INVALID_API_KEY = "invalid-key"

# Supported and unsupported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
UNSUPPORTED_LANGUAGES = ["French", "German", "Spanish", "Chinese", "Japanese"]

# Valid and invalid audio formats
VALID_AUDIO_FORMAT = "mp3"
INVALID_AUDIO_FORMATS = ["wav", "flac", "ogg", "m4a", "aac"]

# Valid and invalid base64 strings
VALID_BASE64 = "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
INVALID_BASE64_STRINGS = ["not-base64!", "invalid@base64", "123-invalid", ""]


def validate_error_response_structure(response_data):
    """
    Validate that a response follows the error response structure.
    
    Args:
        response_data: The JSON response data to validate
    """
    # Must be a dictionary
    assert isinstance(response_data, dict), f"Error response must be a dictionary, got {type(response_data)}"
    
    # Must have exactly two fields: status and message
    expected_fields = {"status", "message"}
    actual_fields = set(response_data.keys())
    assert actual_fields == expected_fields, f"Error response must have exactly {expected_fields}, got {actual_fields}"
    
    # Status must be "error"
    assert response_data["status"] == "error", f"Error response status must be 'error', got '{response_data.get('status')}'"
    
    # Message must be a non-empty string
    message = response_data["message"]
    assert isinstance(message, str), f"Error message must be string, got {type(message)}"
    assert len(message.strip()) > 0, "Error message cannot be empty"


@given(
    api_key=st.one_of(
        st.just(None),  # Missing API key
        st.just(INVALID_API_KEY),  # Invalid API key
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50).filter(lambda x: x != VALID_API_KEY)  # ASCII printable invalid keys
    )
)
@settings(max_examples=5)
def test_authentication_error_response_consistency(api_key):
    """
    **Feature: ai-voice-detection-api, Property 7**: Error Response Consistency (Authentication Errors)
    
    For any authentication error (missing or invalid API key), the system should return 
    a consistent JSON error response format with status "error" and descriptive message.
    
    **Validates: Requirements 6.6, 8.1, 8.4, 8.5**
    """
    request_data = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": VALID_BASE64
    }
    
    # Prepare headers
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    
    response = client.post("/api/voice-detection", json=request_data, headers=headers)
    
    # Should get 401 Unauthorized for authentication errors
    assert response.status_code == 401, f"Expected 401 for auth error, got {response.status_code}"
    
    # Response should be valid JSON
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        pytest.fail("Authentication error response is not valid JSON")
    
    # Validate error response structure
    validate_error_response_structure(response_data)


@given(
    language=st.one_of(
        st.sampled_from(UNSUPPORTED_LANGUAGES),  # Unsupported languages
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=20).filter(lambda x: x not in SUPPORTED_LANGUAGES)  # ASCII printable invalid languages
    ),
    audio_format=st.just(VALID_AUDIO_FORMAT),
    audio_base64=st.just(VALID_BASE64)
)
@settings(max_examples=5)
def test_validation_error_response_consistency_language(language, audio_format, audio_base64):
    """
    **Feature: ai-voice-detection-api, Property 7**: Error Response Consistency (Language Validation Errors)
    
    For any language validation error, the system should return a consistent JSON error 
    response format with status "error" and descriptive message.
    
    **Validates: Requirements 6.6, 8.1, 8.2, 8.4, 8.5**
    """
    request_data = {
        "language": language,
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Should get 400 Bad Request for validation errors
    assert response.status_code == 400, f"Expected 400 for validation error, got {response.status_code}"
    
    # Response should be valid JSON
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        pytest.fail("Validation error response is not valid JSON")
    
    # Validate error response structure
    validate_error_response_structure(response_data)


@given(
    language=st.sampled_from(SUPPORTED_LANGUAGES),
    audio_format=st.sampled_from(INVALID_AUDIO_FORMATS),
    audio_base64=st.just(VALID_BASE64)
)
@settings(max_examples=5)
def test_validation_error_response_consistency_audio_format(language, audio_format, audio_base64):
    """
    **Feature: ai-voice-detection-api, Property 7**: Error Response Consistency (Audio Format Validation Errors)
    
    For any audio format validation error, the system should return a consistent JSON error 
    response format with status "error" and descriptive message.
    
    **Validates: Requirements 6.6, 8.1, 8.2, 8.4, 8.5**
    """
    request_data = {
        "language": language,
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Should get 400 Bad Request for validation errors
    assert response.status_code == 400, f"Expected 400 for validation error, got {response.status_code}"
    
    # Response should be valid JSON
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        pytest.fail("Validation error response is not valid JSON")
    
    # Validate error response structure
    validate_error_response_structure(response_data)


@given(
    language=st.sampled_from(SUPPORTED_LANGUAGES),
    audio_format=st.just(VALID_AUDIO_FORMAT),
    audio_base64=st.sampled_from(INVALID_BASE64_STRINGS)
)
@settings(max_examples=5)
def test_validation_error_response_consistency_base64(language, audio_format, audio_base64):
    """
    **Feature: ai-voice-detection-api, Property 7**: Error Response Consistency (Base64 Validation Errors)
    
    For any base64 validation error, the system should return a consistent JSON error 
    response format with status "error" and descriptive message.
    
    **Validates: Requirements 6.6, 8.1, 8.3, 8.4, 8.5**
    """
    request_data = {
        "language": language,
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Should get 400 Bad Request for validation errors or 422 for processing errors
    assert response.status_code in [400, 422], f"Expected 400 or 422 for base64 error, got {response.status_code}"
    
    # Response should be valid JSON
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        pytest.fail("Base64 validation error response is not valid JSON")
    
    # Validate error response structure
    validate_error_response_structure(response_data)


def test_missing_fields_error_response_consistency():
    """
    Test error response consistency for missing required fields.
    
    **Validates: Requirements 6.6, 8.1, 8.2, 8.4, 8.5**
    """
    # Test missing language field
    request_data = {
        "audioFormat": "mp3",
        "audioBase64": VALID_BASE64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    assert response.status_code == 400
    response_data = response.json()
    validate_error_response_structure(response_data)
    
    # Test missing audioFormat field
    request_data = {
        "language": "English",
        "audioBase64": VALID_BASE64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    assert response.status_code == 400
    response_data = response.json()
    validate_error_response_structure(response_data)
    
    # Test missing audioBase64 field
    request_data = {
        "language": "English",
        "audioFormat": "mp3"
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    assert response.status_code == 400
    response_data = response.json()
    validate_error_response_structure(response_data)


def test_malformed_json_error_response_consistency():
    """
    Test error response consistency for malformed JSON.
    
    **Validates: Requirements 6.6, 8.1, 8.4, 8.5**
    """
    # Send malformed JSON
    response = client.post(
        "/api/voice-detection",
        data='{"language": "English", "audioFormat": "mp3", "audioBase64": "invalid-json"',  # Missing closing brace
        headers={
            "x-api-key": VALID_API_KEY,
            "content-type": "application/json"
        }
    )
    
    # Should get 400 or 422 for malformed JSON (depends on FastAPI version)
    assert response.status_code in [400, 422], f"Expected 400 or 422 for malformed JSON, got {response.status_code}"
    
    # Response should be valid JSON
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        pytest.fail("Malformed JSON error response is not valid JSON")
    
    # Validate error response structure
    validate_error_response_structure(response_data)


def test_empty_request_error_response_consistency():
    """
    Test error response consistency for empty request body.
    
    **Validates: Requirements 6.6, 8.1, 8.4, 8.5**
    """
    response = client.post(
        "/api/voice-detection",
        json={},
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Should get 400 Bad Request for missing fields
    assert response.status_code == 400
    
    response_data = response.json()
    validate_error_response_structure(response_data)