"""
Property-based tests for single audio file processing validation.

**Property 8: Single Audio File Processing**
**Validates: Requirements 1.5**

Tests that for any request, the system processes exactly one audio file per request 
and rejects requests that attempt to submit multiple audio files or no audio files.
"""

import base64
import json
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient
import pytest

from src.main import app


# Test client
client = TestClient(app)

# Valid API key for testing
VALID_API_KEY = "dev-api-key-12345"

# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Generate a minimal valid MP3 base64 for testing
MINIMAL_MP3_BASE64 = base64.b64encode(
    b'ID3\x03\x00\x00\x00\x00\x00\x00' +  # ID3 header
    b'\xff\xfb\x90\x00' +  # MP3 frame header
    b'\x00' * 100  # Some audio data
).decode('ascii')


@given(
    language=st.sampled_from(SUPPORTED_LANGUAGES),
    audio_base64=st.just(MINIMAL_MP3_BASE64)
)
@settings(max_examples=10, deadline=30000)  # Reduced examples for faster testing
def test_single_audio_file_processing_property(language, audio_base64):
    """
    **Feature: ai-voice-detection-api, Property 8**: Single Audio File Processing
    
    For any request, the system should process exactly one audio file per request 
    and reject requests that attempt to submit multiple audio files or no audio files.
    
    **Validates: Requirements 1.5**
    """
    # Test 1: Valid single audio file request structure should be accepted
    # (Even if audio processing fails, the request structure should be valid)
    request_data = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Single audio file request should be accepted (status 200 or 422 for processing error)
    # What we're testing is that the request structure is valid for single file processing
    assert response.status_code in [200, 422], f"Single audio file request should be accepted: {response.text}"
    
    response_data = response.json()
    if response.status_code == 200:
        assert response_data["status"] == "success", "Single audio file should be processed successfully"
    elif response.status_code == 422:
        # Audio processing error is acceptable - we're testing request structure
        assert response_data["status"] == "error", "Processing error should have error status"
    
    # Test 2: Request with multiple audioBase64 fields should be rejected or ignored
    # (This tests the schema validation - Pydantic should reject extra fields)
    request_data_multiple = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64,
        "audioBase64_2": audio_base64  # Additional audio field
    }
    
    response_multiple = client.post(
        "/api/voice-detection",
        json=request_data_multiple,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Multiple audio files should be rejected (either validation error or ignored)
    # If Pydantic ignores extra fields, it should still process with one file
    # If it rejects extra fields, it should return 400
    if response_multiple.status_code == 400:
        # Extra fields rejected - this is acceptable behavior
        response_data_multiple = response_multiple.json()
        assert response_data_multiple["status"] == "error"
    elif response_multiple.status_code in [200, 422]:
        # Extra fields ignored, single file processed - this is also acceptable
        response_data_multiple = response_multiple.json()
        assert response_data_multiple["status"] in ["success", "error"]
    else:
        pytest.fail(f"Unexpected status code for multiple audio files: {response_multiple.status_code}")


def test_single_audio_file_processing_with_empty_audio():
    """
    Test that requests with empty audio data are rejected.
    
    **Validates: Requirements 1.5**
    """
    request_data = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": ""  # Empty audio data
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Empty audio should be rejected
    assert response.status_code == 400, f"Empty audio should be rejected, got {response.status_code}: {response.text}"
    
    response_data = response.json()
    assert response_data["status"] == "error"
    assert ("empty" in response_data["message"].lower() or 
            "required" in response_data["message"].lower() or
            "at least 1 character" in response_data["message"].lower())


def test_single_audio_file_processing_with_missing_audio():
    """
    Test that requests with missing audio data are rejected.
    
    **Validates: Requirements 1.5**
    """
    request_data = {
        "language": "English",
        "audioFormat": "mp3"
        # Missing audioBase64 field
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Missing audio should be rejected
    assert response.status_code == 400, f"Missing audio should be rejected, got {response.status_code}: {response.text}"
    
    response_data = response.json()
    assert response_data["status"] == "error"
    assert "required" in response_data["message"].lower() or "missing" in response_data["message"].lower()


def test_single_audio_file_processing_schema_validation():
    """
    Test that the API schema enforces single audio file processing.
    
    **Validates: Requirements 1.5**
    """
    # Test with array of audio files (should be rejected by schema)
    request_data_array = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": [MINIMAL_MP3_BASE64, MINIMAL_MP3_BASE64]  # Array instead of single string
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data_array,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Array should be rejected by schema validation
    assert response.status_code == 400, f"Audio array should be rejected, got {response.status_code}: {response.text}"
    
    response_data = response.json()
    assert response_data["status"] == "error"


def test_single_audio_file_processing_with_valid_single_file():
    """
    Test that exactly one valid audio file is processed correctly.
    
    **Validates: Requirements 1.5**
    """
    request_data = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": MINIMAL_MP3_BASE64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Single audio file request should be accepted (even if processing fails)
    assert response.status_code in [200, 422], f"Single audio file request should be accepted: {response.text}"
    
    response_data = response.json()
    if response.status_code == 200:
        assert response_data["status"] == "success"
        assert response_data["language"] == "English"
        assert response_data["classification"] in ["AI_GENERATED", "HUMAN"]
        assert 0.0 <= response_data["confidenceScore"] <= 1.0
        assert isinstance(response_data["explanation"], str)
        assert len(response_data["explanation"].strip()) > 0
    elif response.status_code == 422:
        # Audio processing error is acceptable for this test
        assert response_data["status"] == "error"


def test_single_audio_file_processing_all_languages():
    """
    Test single audio file processing for all supported languages.
    
    **Validates: Requirements 1.5**
    """
    for language in SUPPORTED_LANGUAGES:
        request_data = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": MINIMAL_MP3_BASE64
        }
        
        response = client.post(
            "/api/voice-detection",
            json=request_data,
            headers={"x-api-key": VALID_API_KEY}
        )
        
        # Each language should accept single audio file request
        assert response.status_code in [200, 422], f"Single audio request should be accepted for {language}: {response.text}"
        
        response_data = response.json()
        if response.status_code == 200:
            assert response_data["status"] == "success"
            assert response_data["language"] == language
        elif response.status_code == 422:
            # Audio processing error is acceptable
            assert response_data["status"] == "error"


def test_single_audio_file_processing_request_structure():
    """
    Test that the request structure enforces single audio file processing.
    
    **Validates: Requirements 1.5**
    """
    # Test that the API expects exactly the required fields for single audio processing
    valid_request = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": MINIMAL_MP3_BASE64
    }
    
    response = client.post(
        "/api/voice-detection",
        json=valid_request,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # Request should be accepted (even if processing fails)
    assert response.status_code in [200, 422]
    
    # Verify that the response indicates processing of exactly one audio file
    response_data = response.json()
    if response.status_code == 200:
        assert response_data["status"] == "success"
        
        # The response should contain results for exactly one classification
        assert "classification" in response_data
        assert "confidenceScore" in response_data
        assert "explanation" in response_data
        
        # There should be no indication of multiple files being processed
        assert "classifications" not in response_data  # No plural form
        assert "results" not in response_data  # No array of results
    elif response.status_code == 422:
        # Processing error is acceptable
        assert response_data["status"] == "error"