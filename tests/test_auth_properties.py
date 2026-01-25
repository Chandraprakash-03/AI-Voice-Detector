"""
Property-based tests for authentication enforcement.

**Feature: ai-voice-detection-api, Property 2**: Authentication Enforcement
**Validates: Requirements 2.1, 2.3, 2.4**
"""

import pytest
from hypothesis import given, strategies as st
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


@given(
    api_key=st.one_of(
        st.none(),  # Missing API key
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=100),  # ASCII-only API key strings
        st.just("dev-api-key-12345"),  # Valid API key
        st.just(""),  # Empty API key
    ),
    endpoint=st.just("/api/voice-detection"),
    method=st.sampled_from(["POST"])  # Only test POST since endpoint doesn't exist yet
)
def test_authentication_enforcement_property(api_key, endpoint, method):
    """
    Property 2: Authentication Enforcement
    
    For any request to the API, authentication should be enforced such that 
    requests with valid API keys proceed to processing while requests with 
    missing or invalid API keys are rejected with appropriate error responses.
    
    **Validates: Requirements 2.1, 2.3, 2.4**
    """
    # Prepare headers
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    
    # Make request based on method
    if method == "POST":
        response = client.post(endpoint, headers=headers, json={})
    else:
        # Fallback for other methods (shouldn't happen with current strategy)
        response = client.post(endpoint, headers=headers, json={})
    
    # Check authentication enforcement
    if api_key == "dev-api-key-12345":
        # Valid API key should not result in 401 (may get other errors for invalid request format)
        assert response.status_code != 401, f"Valid API key should not return 401, got {response.status_code}"
        if response.status_code >= 400:
            # If there's an error, it should be a proper JSON error response
            assert response.headers.get("content-type", "").startswith("application/json")
            error_data = response.json()
            # Handle both FastAPI default error format and our custom format
            if "status" in error_data:
                assert "message" in error_data
            elif "detail" in error_data:
                # FastAPI default error format is acceptable for non-auth errors
                assert isinstance(error_data["detail"], str)
    else:
        # Invalid, missing, or empty API key should return 401
        assert response.status_code == 401, f"Invalid/missing API key should return 401, got {response.status_code}"
        
        # Response should be JSON with proper error structure
        assert response.headers.get("content-type", "").startswith("application/json")
        error_data = response.json()
        assert error_data["status"] == "error"
        assert "message" in error_data
        assert "API key" in error_data["message"]


@given(
    valid_key=st.just("dev-api-key-12345"),
    public_endpoint=st.sampled_from(["/", "/health", "/docs", "/redoc", "/openapi.json"])
)
def test_public_endpoints_no_auth_required(valid_key, public_endpoint):
    """
    Property: Public endpoints should not require authentication.
    
    Validates that public endpoints are accessible without API keys.
    """
    # Test without API key
    response = client.get(public_endpoint)
    assert response.status_code != 401, f"Public endpoint {public_endpoint} should not require authentication"
    
    # Test with API key (should also work)
    response_with_key = client.get(public_endpoint, headers={"x-api-key": valid_key})
    assert response_with_key.status_code != 401, f"Public endpoint {public_endpoint} should work with API key too"