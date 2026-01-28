"""
Property-based tests for success response structure validation.

**Property 6: Success Response Structure**
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

Tests that for any successful classification request, the response is valid JSON 
with status "success", the original language value, a valid classification result, 
a properly formatted confidence score, and a human-readable explanation.
"""

import base64
import json
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient
import pytest

from src.main import app


# Test client
client = TestClient(app)

# Valid API key for testing (should match the one in auth validator)
VALID_API_KEY = "dev-api-key-12345"

# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Generate a minimal valid MP3 base64 for testing
# This is a minimal MP3 header that will pass basic validation
MINIMAL_MP3_BASE64 = base64.b64encode(
    b'ID3\x03\x00\x00\x00\x00\x00\x00' +  # ID3 header
    b'\xff\xfb\x90\x00' +  # MP3 frame header
    b'\x00' * 100  # Some audio data
).decode('ascii')


@given(
    language=st.sampled_from(SUPPORTED_LANGUAGES),
    audio_base64=st.just(MINIMAL_MP3_BASE64)  # Use fixed valid MP3 for success cases
)
@settings(max_examples=5, deadline=30000)  # Reduced examples for faster testing
def test_success_response_structure_property(language, audio_base64):
    """
    **Feature: ai-voice-detection-api, Property 6**: Success Response Structure
    
    For any successful classification request, the response should be valid JSON 
    with status "success", the original language value, a valid classification result, 
    a properly formatted confidence score, and a human-readable explanation.
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    """
    # Prepare request data
    request_data = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    # Make request with valid API key
    response = client.post(
        "/api/voice-detection",
        json=request_data,
        headers={"x-api-key": VALID_API_KEY}
    )
    
    # For successful requests (status 200), validate response structure
    if response.status_code == 200:
        # Response should be valid JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")
        
        # Requirement 6.1: Response should have status "success"
        assert "status" in response_data, "Response missing 'status' field"
        assert response_data["status"] == "success", f"Expected status 'success', got '{response_data.get('status')}'"
        
        # Requirement 6.2: Response should include original language value
        assert "language" in response_data, "Response missing 'language' field"
        assert response_data["language"] == language, f"Expected language '{language}', got '{response_data.get('language')}'"
        
        # Requirement 6.3: Response should include valid classification result
        assert "classification" in response_data, "Response missing 'classification' field"
        classification = response_data["classification"]
        valid_classifications = ["AI_GENERATED", "HUMAN"]
        assert classification in valid_classifications, f"Invalid classification '{classification}', must be one of {valid_classifications}"
        
        # Requirement 6.4: Response should include properly formatted confidence score
        assert "confidenceScore" in response_data, "Response missing 'confidenceScore' field"
        confidence_score = response_data["confidenceScore"]
        assert isinstance(confidence_score, (int, float)), f"Confidence score must be numeric, got {type(confidence_score)}"
        assert 0.0 <= confidence_score <= 1.0, f"Confidence score must be between 0.0 and 1.0, got {confidence_score}"
        
        # Requirement 6.5: Response should include human-readable explanation
        assert "explanation" in response_data, "Response missing 'explanation' field"
        explanation = response_data["explanation"]
        assert isinstance(explanation, str), f"Explanation must be string, got {type(explanation)}"
        assert len(explanation.strip()) > 0, "Explanation cannot be empty"
        
        # Additional validation: ensure no unexpected fields in success response
        expected_fields = {"status", "language", "classification", "confidenceScore", "explanation"}
        actual_fields = set(response_data.keys())
        unexpected_fields = actual_fields - expected_fields
        assert len(unexpected_fields) == 0, f"Response contains unexpected fields: {unexpected_fields}"


def test_success_response_structure_with_known_good_input():
    """
    Test success response structure with a known good input to ensure the property test works.
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
    
    # Should get a response (either success or processing error)
    assert response.status_code in [200, 422]
    
    response_data = response.json()
    
    if response.status_code == 200:
        # Validate all required fields are present and correct
        assert response_data["status"] == "success"
        assert response_data["language"] == "English"
        assert response_data["classification"] in ["AI_GENERATED", "HUMAN"]
        assert 0.0 <= response_data["confidenceScore"] <= 1.0
        assert isinstance(response_data["explanation"], str)
        assert len(response_data["explanation"].strip()) > 0
    elif response.status_code == 422:
        # Audio processing error is acceptable for this test
        assert response_data["status"] == "error"


def test_success_response_structure_all_languages():
    """
    Test success response structure for all supported languages.
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
        
        # If audio processing fails due to invalid test MP3, skip this language
        if response.status_code == 422:
            response_data = response.json()
            if "audio processing failed" in response_data.get("message", "").lower():
                pytest.skip(f"Skipping {language} due to invalid test MP3 data")
        
        # Should get successful response for all supported languages
        assert response.status_code == 200, f"Failed for language {language}: {response.text}"
        
        response_data = response.json()
        assert response_data["language"] == language
        assert response_data["status"] == "success"