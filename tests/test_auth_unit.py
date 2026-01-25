"""
Unit tests for API key validation.

Tests specific examples and edge cases for the authentication system.
Requirements: 2.1, 2.2, 2.3
"""

import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.auth.validator import APIKeyValidator
from src.main import app

client = TestClient(app)


class TestAPIKeyValidator:
    """Unit tests for the APIKeyValidator class."""
    
    def test_validate_key_with_valid_key(self):
        """Test that valid API key is accepted."""
        validator = APIKeyValidator()
        assert validator.validate_key("dev-api-key-12345") is True
    
    def test_validate_key_with_invalid_key(self):
        """Test that invalid API key is rejected."""
        validator = APIKeyValidator()
        assert validator.validate_key("invalid-key") is False
    
    def test_validate_key_with_empty_key(self):
        """Test that empty API key is rejected."""
        validator = APIKeyValidator()
        assert validator.validate_key("") is False
    
    def test_validate_key_with_none(self):
        """Test that None API key is rejected."""
        validator = APIKeyValidator()
        assert validator.validate_key(None) is False
    
    def test_extract_api_key_present(self):
        """Test extracting API key when x-api-key header is present."""
        validator = APIKeyValidator()
        headers = {"x-api-key": "test-key-123"}
        assert validator.extract_api_key(headers) == "test-key-123"
    
    def test_extract_api_key_case_insensitive(self):
        """Test extracting API key with different case headers."""
        validator = APIKeyValidator()
        headers = {"X-API-KEY": "test-key-123"}
        assert validator.extract_api_key(headers) == "test-key-123"
    
    def test_extract_api_key_missing(self):
        """Test extracting API key when header is missing."""
        validator = APIKeyValidator()
        headers = {"other-header": "value"}
        assert validator.extract_api_key(headers) is None
    
    def test_extract_api_key_empty_headers(self):
        """Test extracting API key from empty headers."""
        validator = APIKeyValidator()
        headers = {}
        assert validator.extract_api_key(headers) is None
    
    @patch.dict(os.environ, {"API_KEYS": "key1,key2,key3"})
    def test_load_valid_keys_from_env(self):
        """Test loading API keys from environment variable."""
        validator = APIKeyValidator()
        assert "key1" in validator._valid_keys
        assert "key2" in validator._valid_keys
        assert "key3" in validator._valid_keys
    
    @patch.dict(os.environ, {"API_KEYS": "key1, key2 , key3 "})
    def test_load_valid_keys_with_spaces(self):
        """Test loading API keys with spaces from environment."""
        validator = APIKeyValidator()
        assert "key1" in validator._valid_keys
        assert "key2" in validator._valid_keys
        assert "key3" in validator._valid_keys
        assert " key2 " not in validator._valid_keys  # Spaces should be stripped
    
    @patch.dict(os.environ, {}, clear=True)
    def test_load_valid_keys_default(self):
        """Test default API key when no environment variable is set."""
        validator = APIKeyValidator()
        assert "dev-api-key-12345" in validator._valid_keys
    
    def test_generate_auth_error(self):
        """Test that authentication error is properly formatted."""
        validator = APIKeyValidator()
        error = validator.generate_auth_error()
        assert error.status_code == 401
        assert "API key" in error.detail


class TestAuthenticationMiddleware:
    """Unit tests for authentication middleware integration."""
    
    def test_valid_api_key_acceptance(self):
        """Test that valid API key allows access to protected endpoint."""
        headers = {"x-api-key": "dev-api-key-12345"}
        # Since /api/voice-detection doesn't exist yet, we expect 404, not 401
        response = client.post("/api/voice-detection", headers=headers, json={})
        assert response.status_code != 401  # Should not be authentication error
    
    def test_invalid_api_key_rejection(self):
        """Test that invalid API key is rejected."""
        headers = {"x-api-key": "invalid-key"}
        response = client.post("/api/voice-detection", headers=headers, json={})
        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"
        assert "API key" in data["message"]
    
    def test_missing_api_key_handling(self):
        """Test that missing API key is rejected."""
        response = client.post("/api/voice-detection", json={})
        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"
        assert "Missing API key" in data["message"]
    
    def test_public_endpoints_no_auth_required(self):
        """Test that public endpoints don't require authentication."""
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
    
    def test_options_request_no_auth_required(self):
        """Test that OPTIONS requests (CORS preflight) don't require auth."""
        response = client.options("/api/voice-detection")
        assert response.status_code != 401
    
    def test_case_insensitive_header_extraction(self):
        """Test that API key header extraction is case-insensitive."""
        headers = {"X-Api-Key": "dev-api-key-12345"}
        response = client.post("/api/voice-detection", headers=headers, json={})
        assert response.status_code != 401  # Should not be authentication error