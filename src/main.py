"""
AI-Generated Voice Detection API

Main FastAPI application entry point.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings, setup_logging
from src.auth import AuthenticationMiddleware
from src.models.schemas import VoiceDetectionRequest, SuccessResponse, ErrorResponse
from src.audio.processor import AudioProcessor
from src.detection.engine import DetectionEngine
from src.error_handlers import EXCEPTION_HANDLERS
from src.exceptions import (
    APIError,
    AuthenticationError,
    ValidationError,
    AudioProcessingError,
    DetectionEngineError,
    InvalidAudioFormatError,
    Base64DecodingError,
    MP3ValidationError,
    FeatureExtractionError,
    InferenceError,
    UnsupportedLanguageError
)

# Load configuration
settings = get_settings()

# Setup logging
setup_logging(settings)
logger = logging.getLogger(__name__)

# Create FastAPI app with configuration
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "AI Voice Detection Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": f"http://{settings.api.host}:{settings.api.port}",
            "description": f"{settings.environment.value.title()} server"
        }
    ],
    tags_metadata=[
        {
            "name": "voice-detection",
            "description": "Voice detection and classification operations",
        },
        {
            "name": "health",
            "description": "Health check and system status operations",
        },
        {
            "name": "system",
            "description": "System information and metadata operations",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=settings.api.cors_methods,
    allow_headers=settings.api.cors_headers,
)

# Add authentication middleware
app.add_middleware(AuthenticationMiddleware)

# Register exception handlers
for exception_type, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exception_type, handler)

# Initialize components with configuration
audio_processor = AudioProcessor(settings)
detection_engine = DetectionEngine(settings)


@app.post(
    "/api/voice-detection", 
    response_model=SuccessResponse,
    tags=["voice-detection"],
    summary="Detect AI-generated voice",
    description="Analyze voice recording to determine if it's AI-generated or human-spoken",
    responses={
        200: {
            "description": "Successful voice detection analysis",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "language": "English",
                        "classification": "AI_GENERATED",
                        "confidenceScore": 0.87,
                        "explanation": "The audio exhibits characteristics typical of AI-generated speech, including consistent pitch patterns and synthetic vocal tract modeling."
                    }
                }
            }
        },
        400: {
            "description": "Bad Request - Invalid input data",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Invalid audioFormat: 'wav'. Only 'mp3' format is supported"
                    }
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Invalid or missing API key. Please provide a valid x-api-key header."
                    }
                }
            }
        },
        422: {
            "description": "Unprocessable Entity - Invalid audio data",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Audio file is corrupted or not a valid MP3 format"
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error - System error",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Internal server error"
                    }
                }
            }
        }
    }
)
async def detect_voice(request: VoiceDetectionRequest):
    """
    Analyze voice recording to determine if it's AI-generated or human.
    
    This endpoint accepts MP3 audio files encoded in base64 format and returns
    a classification result with confidence score and explanation.
    
    Args:
        request: Voice detection request containing language, audio format, and base64 data
        
    Returns:
        SuccessResponse: Classification result with confidence score and explanation
        
    Raises:
        Various API exceptions for different error conditions
    """
    try:
        logger.info(f"Processing voice detection request for language: {request.language}")
        
        # Process audio data
        logger.info("Extracting audio features...")
        audio_features = audio_processor.process_base64_audio(
            request.audioBase64, 
            request.audioFormat
        )
        
        # Perform detection
        logger.info("Running voice detection analysis...")
        detection_result = detection_engine.detect_voice_type(
            audio_features, 
            request.language
        )
        
        # Generate explanation
        explanation = detection_engine.generate_explanation(
            detection_result, 
            audio_features, 
            request.language
        )
        
        # Create success response
        response = SuccessResponse(
            language=request.language,
            classification=detection_result.classification,
            confidenceScore=detection_result.confidence_score,
            explanation=explanation
        )
        
        logger.info(
            f"Voice detection completed: {detection_result.classification} "
            f"(confidence: {detection_result.confidence_score:.3f})"
        )
        
        return response
        
    except (
        InvalidAudioFormatError,
        Base64DecodingError,
        MP3ValidationError,
        FeatureExtractionError,
        UnsupportedLanguageError,
        InferenceError,
        AudioProcessingError,
        DetectionEngineError
    ):
        # Re-raise specific exceptions to be handled by their respective handlers
        raise
    except Exception as e:
        # Log unexpected errors and re-raise for general handler
        logger.error(f"Unexpected error in voice detection: {str(e)}", exc_info=True)
        raise


@app.get(
    "/health",
    tags=["health"],
    summary="Basic health check",
    description="Check if the API is running and responsive",
    responses={
        200: {
            "description": "API is healthy and operational",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "ai-voice-detection-api",
                        "version": "1.0.0",
                        "environment": "production",
                        "debug": False
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Health check endpoint to verify API availability.

    Returns:
        dict: Status information indicating the API is operational
    """
    return {
        "status": "healthy",
        "service": "ai-voice-detection-api",
        "version": settings.api.version,
        "environment": settings.environment.value,
        "debug": settings.debug
    }


@app.get(
    "/health/detailed",
    tags=["health"],
    summary="Detailed health check",
    description="Get detailed system health information including configuration and status",
    responses={
        200: {
            "description": "Detailed system health information",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "ai-voice-detection-api",
                        "version": "1.0.0",
                        "environment": "production",
                        "debug": False,
                        "configuration": {
                            "models_path": "/app/models",
                            "models_available": True,
                            "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
                            "confidence_threshold": 0.5,
                            "max_audio_duration": 300,
                            "min_audio_duration": 0.5
                        },
                        "system": {
                            "cors_enabled": True,
                            "ssl_enabled": True,
                            "security_headers_enabled": True
                        }
                    }
                }
            }
        }
    }
)
async def detailed_health_check():
    """
    Detailed health check endpoint with system information.

    Returns:
        dict: Detailed status information including configuration and system state
    """
    try:
        # Check if models directory exists
        import os
        models_available = os.path.exists(settings.ml.models_base_path)
        
        # Check supported languages
        supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        
        return {
            "status": "healthy",
            "service": "ai-voice-detection-api",
            "version": settings.api.version,
            "environment": settings.environment.value,
            "debug": settings.debug,
            "configuration": {
                "models_path": settings.ml.models_base_path,
                "models_available": models_available,
                "supported_languages": supported_languages,
                "confidence_threshold": settings.ml.confidence_threshold,
                "max_audio_duration": settings.ml.max_audio_duration,
                "min_audio_duration": settings.ml.min_audio_duration,
            },
            "system": {
                "cors_enabled": len(settings.api.cors_origins) > 0,
                "ssl_enabled": settings.security.ssl_enabled,
                "security_headers_enabled": settings.security.security_headers_enabled,
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "ai-voice-detection-api",
            "error": str(e)
        }


@app.get(
    "/",
    tags=["system"],
    summary="API information",
    description="Get basic API information and available endpoints",
    responses={
        200: {
            "description": "API information and endpoints",
            "content": {
                "application/json": {
                    "example": {
                        "message": "AI-Generated Voice Detection API",
                        "version": "1.0.0",
                        "environment": "production",
                        "endpoints": {
                            "health": "/health",
                            "detailed_health": "/health/detailed",
                            "voice_detection": "/api/voice-detection",
                            "documentation": "/docs",
                            "openapi": "/openapi.json"
                        }
                    }
                }
            }
        }
    }
)
async def root():
    """
    Root endpoint providing basic API information.

    Returns:
        dict: Basic API information and available endpoints
    """
    return {
        "message": settings.api.title,
        "version": settings.api.version,
        "environment": settings.environment.value,
        "endpoints": {
            "health": "/health",
            "detailed_health": "/health/detailed",
            "voice_detection": "/api/voice-detection",
            "documentation": "/docs",
            "openapi": "/openapi.json"
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.log_level.value.lower(),
        access_log=True
    )
