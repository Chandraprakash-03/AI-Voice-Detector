"""
API Key Validation Module

Provides authentication services for the AI-Generated Voice Detection API.
"""

import os
from typing import Optional, Set
from fastapi import HTTPException, status

from src.config import get_settings


class APIKeyValidator:
    """
    Handles API key validation and management.
    
    This class provides methods to validate API keys against a configured
    set of valid keys, supporting both environment variable and configuration
    file-based key management.
    """
    
    def __init__(self, settings=None):
        """Initialize the API key validator with configured valid keys."""
        self.settings = settings or get_settings()
        self._valid_keys: Set[str] = set(self.settings.auth.api_keys)
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key against the configured valid keys.
        
        Args:
            api_key (str): The API key to validate
            
        Returns:
            bool: True if the key is valid, False otherwise
        """
        if not api_key:
            return False
        return api_key in self._valid_keys
    
    def extract_api_key(self, headers: dict) -> Optional[str]:
        """
        Extract API key from request headers.
        
        Args:
            headers (dict): Request headers dictionary
            
        Returns:
            Optional[str]: The API key if present, None otherwise
        """
        # Check for x-api-key header (case-insensitive)
        for key, value in headers.items():
            if key.lower() == "x-api-key":
                return value
        return None
    
    def generate_auth_error(self) -> HTTPException:
        """
        Generate a standardized authentication error response.
        
        Returns:
            HTTPException: 401 Unauthorized error with descriptive message
        """
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Please provide a valid x-api-key header."
        )


# Global validator instance
api_key_validator = APIKeyValidator()