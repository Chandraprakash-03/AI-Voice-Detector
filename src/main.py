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
detection_engine = DetectionEngine(settings, foundation_model=settings.ml.foundation_model)


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
        
        # Perform detection with optional ground truth for monitoring
        logger.info("Running voice detection analysis...")
        ground_truth = getattr(request, 'groundTruth', None)  # Optional field for monitoring
        
        detection_result = detection_engine.detect_voice_type(
            audio_features, 
            request.language,
            ground_truth=ground_truth
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
            f"(confidence: {detection_result.confidence_score:.3f}, "
            f"processing_time: {detection_result.processing_time:.3f}s, "
            f"threshold: {getattr(detection_result, 'threshold_used', 'N/A')})"
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
    Detailed health check endpoint with comprehensive system information.

    Returns:
        dict: Detailed status information including configuration, validation, monitoring, and system state
    """
    try:
        # Get comprehensive system health status from detection engine
        system_health = detection_engine.get_system_health_status()
        
        # Check if models directory exists
        import os
        models_available = os.path.exists(settings.ml.models_base_path)
        
        # Check supported languages
        supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        
        # Determine overall status based on health score
        health_score = system_health.get("overall_health", {}).get("score", 0.0)
        overall_status = "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.6 else "unhealthy"
        
        return {
            "status": overall_status,
            "service": "ai-voice-detection-api",
            "version": settings.api.version,
            "environment": settings.environment.value,
            "debug": settings.debug,
            "health_score": health_score,
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
            },
            "validation": system_health.get("classifiers", {}),
            "foundation_model": system_health.get("foundation_model", {}),
            "monitoring": system_health.get("monitoring", {}),
            "registry": system_health.get("registry", {}),
            "fallback": system_health.get("fallback", {}),
            "thresholds": system_health.get("thresholds", {})
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "ai-voice-detection-api",
            "error": str(e),
            "health_score": 0.0
        }


@app.get(
    "/health/monitoring",
    tags=["health"],
    summary="Monitoring health check",
    description="Get monitoring and performance information",
    responses={
        200: {
            "description": "Monitoring information",
            "content": {
                "application/json": {
                    "example": {
                        "monitoring_active": True,
                        "total_predictions": 1500,
                        "accuracy_by_language": {
                            "English": 0.92,
                            "Tamil": 0.89
                        },
                        "recent_alerts": 0,
                        "system_health_score": 0.85
                    }
                }
            }
        }
    }
)
async def monitoring_health_check():
    """
    Monitoring-focused health check endpoint.

    Returns:
        dict: Monitoring status and performance metrics
    """
    try:
        # Get monitoring report for last 24 hours
        monitoring_report = detection_engine.get_monitoring_report(hours=24)
        
        # Get system health status
        system_health = detection_engine.get_system_health_status()
        health_score = system_health.get("overall_health", {}).get("score", 0.0)
        
        # Extract key monitoring metrics
        monitoring_summary = system_health.get("monitoring", {})
        
        return {
            "monitoring_active": True,
            "total_predictions": monitoring_summary.get("total_predictions", 0),
            "monitoring_duration": monitoring_summary.get("monitoring_duration", "0:00:00"),
            "monitored_languages": monitoring_summary.get("monitored_languages", []),
            "current_accuracies": monitoring_summary.get("current_accuracies", {}),
            "baseline_accuracies": monitoring_summary.get("baseline_accuracies", {}),
            "recent_alerts": monitoring_summary.get("recent_alerts_count", 0),
            "system_health_score": health_score,
            "detailed_report": monitoring_report if "error" not in monitoring_report else None,
            "error": monitoring_report.get("error") if "error" in monitoring_report else None
        }
        
    except Exception as e:
        logger.error(f"Monitoring health check failed: {str(e)}")
        return {
            "monitoring_active": False,
            "error": str(e),
            "system_health_score": 0.0
        }


@app.get(
    "/health/models",
    tags=["health"],
    summary="Model validation health check",
    description="Get model validation status and registry information",
    responses={
        200: {
            "description": "Model validation information",
            "content": {
                "application/json": {
                    "example": {
                        "foundation_model_valid": True,
                        "valid_classifiers": 4,
                        "total_classifiers": 5,
                        "registry_models": 5,
                        "validation_details": {
                            "English": {"valid": True, "confidence": 0.85},
                            "Tamil": {"valid": True, "confidence": 0.82}
                        }
                    }
                }
            }
        }
    }
)
async def models_health_check():
    """
    Model-focused health check endpoint.

    Returns:
        dict: Model validation status and registry information
    """
    try:
        # Get system health status
        system_health = detection_engine.get_system_health_status()
        
        # Extract model information
        foundation_model = system_health.get("foundation_model", {})
        classifiers = system_health.get("classifiers", {})
        registry = system_health.get("registry", {})
        
        # Count valid classifiers
        valid_classifiers = sum(1 for status in classifiers.values() 
                               if status.get("classifier_valid", False))
        total_classifiers = len(classifiers)
        
        # Get registry information
        registry_info = detection_engine.get_model_registry_info()
        
        return {
            "foundation_model": {
                "name": foundation_model.get("name", "unknown"),
                "valid": foundation_model.get("valid", False),
                "config": foundation_model.get("config", {})
            },
            "classifiers": {
                "valid_count": valid_classifiers,
                "total_count": total_classifiers,
                "validation_details": classifiers
            },
            "registry": {
                "total_models": registry.get("total_models", 0),
                "models_by_type": registry.get("models_by_type", {}),
                "models_by_status": registry.get("models_by_status", {}),
                "models_info": registry_info.get("models", {}) if "error" not in registry_info else None
            },
            "thresholds": system_health.get("thresholds", {}),
            "overall_model_health": "healthy" if valid_classifiers == total_classifiers and foundation_model.get("valid", False) else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Models health check failed: {str(e)}")
        return {
            "foundation_model": {"valid": False},
            "classifiers": {"valid_count": 0, "total_count": 0},
            "registry": {"total_models": 0},
            "overall_model_health": "unhealthy",
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
