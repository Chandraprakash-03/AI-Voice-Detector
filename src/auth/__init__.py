"""
Authentication module for AI-Generated Voice Detection API.

This module provides API key validation and authentication middleware
for securing the voice detection endpoints.
"""

from .validator import APIKeyValidator, api_key_validator
from .middleware import AuthenticationMiddleware

__all__ = ["APIKeyValidator", "api_key_validator", "AuthenticationMiddleware"]
