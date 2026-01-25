"""
Authentication Middleware

FastAPI middleware for API key validation and request authentication.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .validator import api_key_validator
from src.exceptions import AuthenticationError


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authenticating API requests using API keys.
    
    This middleware intercepts all requests to protected endpoints and
    validates the presence and validity of API keys in the x-api-key header.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize the authentication middleware.
        
        Args:
            app (ASGIApp): The ASGI application to wrap
        """
        super().__init__(app)
        self.protected_paths = {"/api/voice-detection"}
        self.public_paths = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and validate authentication for protected endpoints.
        
        Args:
            request (Request): The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: The HTTP response
        """
        # Skip authentication for public endpoints
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check if this is a protected endpoint
        if request.url.path in self.protected_paths:
            try:
                # Extract API key from headers
                api_key = api_key_validator.extract_api_key(dict(request.headers))
                
                if not api_key:
                    raise AuthenticationError("Missing API key. Please provide a valid x-api-key header.")
                
                # Validate the API key
                if not api_key_validator.validate_key(api_key):
                    raise AuthenticationError("Invalid API key. Please provide a valid x-api-key header.")
            
            except AuthenticationError as e:
                # Handle authentication errors in middleware by returning JSON response
                from src.error_handlers import authentication_error_handler
                return await authentication_error_handler(request, e)
        
        # Continue to the next middleware or endpoint
        return await call_next(request)