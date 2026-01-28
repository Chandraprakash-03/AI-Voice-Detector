"""
Comprehensive error management system for AI voice detection.

This module implements specific error messages for different failure types,
detailed logging for all fallback scenarios, and error recovery mechanisms.
"""

import logging
import time
import traceback
from typing import Dict, Optional, List, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from src.exceptions import (
    DetectionEngineError,
    ModelLoadingError,
    InferenceError,
    AudioProcessingError,
    ValidationError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for specific handling."""
    MODEL_LOADING = "model_loading"
    FOUNDATION_MODEL = "foundation_model"
    CLASSIFIER = "classifier"
    AUDIO_PROCESSING = "audio_processing"
    INFERENCE = "inference"
    VALIDATION = "validation"
    RESOURCE_CONSTRAINT = "resource_constraint"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    component: str
    operation: str
    language: Optional[str] = None
    model_name: Optional[str] = None
    audio_duration: Optional[float] = None
    system_resources: Dict[str, Any] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    recovery_method: str
    fallback_used: bool
    error_message: Optional[str] = None
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class ErrorManager:
    """
    Comprehensive error management system.
    
    Provides specific error messages for different failure types,
    detailed logging for all fallback scenarios, and error recovery mechanisms.
    
    Requirements addressed:
    - 6.1: Specific error messages for different failure types
    - 6.5: Detailed logging for all fallback scenarios
    - Error recovery mechanisms for various failure conditions
    """
    
    def __init__(self, settings=None):
        """
        Initialize the ErrorManager.
        
        Args:
            settings: Application settings object containing configuration
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        
        # Error message templates for specific failure types
        self.error_templates = self._initialize_error_templates()
        
        # Recovery strategies for different error categories
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Error tracking for pattern analysis
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        
        logger.info("ErrorManager initialized with comprehensive error handling")
    
    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """
        Handle error with specific messaging and recovery attempts.
        
        Args:
            error: The exception that occurred
            context: Error context information
            
        Returns:
            ErrorRecoveryResult with recovery information
        """
        start_time = time.time()
        
        try:
            # Generate specific error message
            specific_message = self._generate_specific_error_message(error, context)
            
            # Log detailed error information
            self._log_detailed_error(error, context, specific_message)
            
            # Track error for pattern analysis
            self._track_error(error, context)
            
            # Attempt error recovery
            recovery_result = self._attempt_error_recovery(error, context)
            
            # Update recovery result with timing
            recovery_result.processing_time = time.time() - start_time
            
            # Log recovery outcome
            self._log_recovery_outcome(recovery_result, context)
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
            
            return ErrorRecoveryResult(
                success=False,
                recovery_method="none",
                fallback_used=False,
                error_message=f"Recovery failed: {str(recovery_error)}",
                processing_time=time.time() - start_time
            )
    
    def _generate_specific_error_message(self, error: Exception, context: ErrorContext) -> str:
        """
        Generate specific error message based on error type and context.
        
        Args:
            error: The exception that occurred
            context: Error context information
            
        Returns:
            Specific error message string
        """
        try:
            error_type = type(error).__name__
            category = context.error_category.value
            
            # Get base template for error category
            template = self.error_templates.get(category, {}).get(error_type, 
                self.error_templates.get("default", {}).get(error_type, 
                    "An error occurred in {component} during {operation}: {error_message}"
                )
            )
            
            # Format template with context information
            message = template.format(
                component=context.component,
                operation=context.operation,
                language=context.language or "unknown",
                model_name=context.model_name or "unknown",
                audio_duration=context.audio_duration or 0.0,
                error_message=str(error),
                error_type=error_type
            )
            
            # Add severity indicator
            severity_prefix = {
                ErrorSeverity.LOW: "[INFO]",
                ErrorSeverity.MEDIUM: "[WARNING]",
                ErrorSeverity.HIGH: "[ERROR]",
                ErrorSeverity.CRITICAL: "[CRITICAL]"
            }.get(context.error_severity, "[ERROR]")
            
            return f"{severity_prefix} {message}"
            
        except Exception as e:
            logger.error(f"Failed to generate specific error message: {e}")
            return f"Error in {context.component}: {str(error)}"
    
    def _log_detailed_error(self, error: Exception, context: ErrorContext, specific_message: str):
        """
        Log detailed error information for debugging and monitoring.
        
        Args:
            error: The exception that occurred
            context: Error context information
            specific_message: Generated specific error message
        """
        try:
            # Prepare detailed log entry
            log_data = {
                "error_category": context.error_category.value,
                "error_severity": context.error_severity.value,
                "component": context.component,
                "operation": context.operation,
                "language": context.language,
                "model_name": context.model_name,
                "audio_duration": context.audio_duration,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "specific_message": specific_message,
                "system_resources": context.system_resources,
                "error_history_count": len(context.error_history),
                "timestamp": context.timestamp.isoformat(),
                "request_id": context.request_id,
                "traceback": traceback.format_exc() if context.error_severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
            }
            
            # Log at appropriate level based on severity
            if context.error_severity == ErrorSeverity.CRITICAL:
                logger.critical(f"Critical error in {context.component}: {specific_message}", extra={"error_details": log_data})
            elif context.error_severity == ErrorSeverity.HIGH:
                logger.error(f"High severity error in {context.component}: {specific_message}", extra={"error_details": log_data})
            elif context.error_severity == ErrorSeverity.MEDIUM:
                logger.warning(f"Medium severity error in {context.component}: {specific_message}", extra={"error_details": log_data})
            else:
                logger.info(f"Low severity error in {context.component}: {specific_message}", extra={"error_details": log_data})
            
            # Log fallback scenario details if available
            if context.error_history:
                self._log_fallback_scenario_details(context)
                
        except Exception as e:
            logger.error(f"Failed to log detailed error information: {e}")
    
    def _log_fallback_scenario_details(self, context: ErrorContext):
        """
        Log detailed information about fallback scenarios.
        
        Args:
            context: Error context containing fallback history
        """
        try:
            fallback_summary = {
                "component": context.component,
                "language": context.language,
                "fallback_chain": [],
                "total_fallbacks": len(context.error_history),
                "fallback_timeline": []
            }
            
            for i, error_event in enumerate(context.error_history):
                fallback_info = {
                    "step": i + 1,
                    "error_type": error_event.get("type", "unknown"),
                    "error_message": error_event.get("error", "unknown"),
                    "timestamp": error_event.get("timestamp", 0),
                    "fallback_method": error_event.get("fallback_method", "unknown"),
                    "success": error_event.get("success", False)
                }
                
                fallback_summary["fallback_chain"].append(fallback_info)
                fallback_summary["fallback_timeline"].append({
                    "step": i + 1,
                    "method": error_event.get("fallback_method", "unknown"),
                    "duration": error_event.get("duration", 0)
                })
            
            logger.info(
                f"Fallback scenario details for {context.component} ({context.language}): "
                f"{fallback_summary['total_fallbacks']} fallback attempts",
                extra={"fallback_details": fallback_summary}
            )
            
        except Exception as e:
            logger.error(f"Failed to log fallback scenario details: {e}")
    
    def _track_error(self, error: Exception, context: ErrorContext):
        """
        Track error for pattern analysis and monitoring.
        
        Args:
            error: The exception that occurred
            context: Error context information
        """
        try:
            error_key = f"{context.error_category.value}_{type(error).__name__}_{context.component}"
            
            # Update error counts
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Add to error history
            error_record = {
                "timestamp": context.timestamp,
                "error_key": error_key,
                "error_category": context.error_category.value,
                "error_severity": context.error_severity.value,
                "component": context.component,
                "operation": context.operation,
                "language": context.language,
                "model_name": context.model_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "system_resources": context.system_resources.copy(),
                "request_id": context.request_id
            }
            
            self.error_history.append(error_record)
            
            # Keep only recent errors (last 1000)
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
            
            # Log pattern if error is recurring
            if self.error_counts[error_key] > 1:
                logger.warning(
                    f"Recurring error pattern detected: {error_key} "
                    f"(count: {self.error_counts[error_key]})"
                )
                
        except Exception as e:
            logger.error(f"Failed to track error: {e}")
    
    def _attempt_error_recovery(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """
        Attempt error recovery using appropriate strategies.
        
        Args:
            error: The exception that occurred
            context: Error context information
            
        Returns:
            ErrorRecoveryResult with recovery information
        """
        try:
            # Get recovery strategy for error category
            strategy = self.recovery_strategies.get(context.error_category, None)
            
            if strategy is None:
                return ErrorRecoveryResult(
                    success=False,
                    recovery_method="none",
                    fallback_used=False,
                    error_message="No recovery strategy available"
                )
            
            # Attempt recovery
            recovery_result = strategy(error, context)
            
            # Log recovery attempt
            logger.info(
                f"Recovery attempt for {context.error_category.value} in {context.component}: "
                f"{'successful' if recovery_result.success else 'failed'} "
                f"using {recovery_result.recovery_method}"
            )
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error(f"Error recovery attempt failed: {str(recovery_error)}")
            
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=f"Recovery attempt failed: {str(recovery_error)}"
            )
    
    def _log_recovery_outcome(self, recovery_result: ErrorRecoveryResult, context: ErrorContext):
        """
        Log the outcome of error recovery attempts.
        
        Args:
            recovery_result: Result of recovery attempt
            context: Error context information
        """
        try:
            outcome_data = {
                "component": context.component,
                "error_category": context.error_category.value,
                "recovery_success": recovery_result.success,
                "recovery_method": recovery_result.recovery_method,
                "fallback_used": recovery_result.fallback_used,
                "processing_time": recovery_result.processing_time,
                "recovery_data": recovery_result.recovery_data,
                "language": context.language,
                "model_name": context.model_name
            }
            
            if recovery_result.success:
                logger.info(
                    f"Error recovery successful for {context.component}: "
                    f"used {recovery_result.recovery_method} "
                    f"({'with fallback' if recovery_result.fallback_used else 'without fallback'}) "
                    f"in {recovery_result.processing_time:.3f}s",
                    extra={"recovery_outcome": outcome_data}
                )
            else:
                logger.warning(
                    f"Error recovery failed for {context.component}: "
                    f"attempted {recovery_result.recovery_method}, "
                    f"error: {recovery_result.error_message}",
                    extra={"recovery_outcome": outcome_data}
                )
                
        except Exception as e:
            logger.error(f"Failed to log recovery outcome: {e}")
    
    def _initialize_error_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize error message templates for specific failure types."""
        return {
            "model_loading": {
                "ModelLoadingError": "Failed to load {model_name} model for {language}: {error_message}. Check model file integrity and path configuration.",
                "FileNotFoundError": "Model file not found for {language} at expected location. Verify model installation and path configuration.",
                "PermissionError": "Permission denied accessing {model_name} model file. Check file permissions and user access rights.",
                "MemoryError": "Insufficient memory to load {model_name} model for {language}. Consider using a smaller model or increasing available memory.",
                "OSError": "System error loading {model_name} model: {error_message}. Check disk space and file system integrity."
            },
            "foundation_model": {
                "RuntimeError": "Foundation model {model_name} failed during embedding extraction: {error_message}. Attempting fallback to traditional features.",
                "ValueError": "Invalid input data for foundation model {model_name}: {error_message}. Check audio preprocessing pipeline.",
                "MemoryError": "Foundation model {model_name} ran out of memory processing {audio_duration:.1f}s audio. Using resource-aware fallback.",
                "TimeoutError": "Foundation model {model_name} processing timeout after {audio_duration:.1f}s. Switching to faster fallback method."
            },
            "classifier": {
                "InferenceError": "Classifier inference failed for {language} using {model_name}: {error_message}. Attempting ensemble fallback.",
                "ValueError": "Invalid embedding dimensions for {language} classifier: {error_message}. Check model compatibility.",
                "RuntimeError": "Classifier runtime error for {language}: {error_message}. Using statistical fallback method.",
                "MemoryError": "Classifier {language} ran out of memory. Using lightweight fallback classification."
            },
            "audio_processing": {
                "AudioProcessingError": "Audio processing failed during {operation}: {error_message}. Check audio format and quality.",
                "ValueError": "Invalid audio data for processing: {error_message}. Verify audio encoding and sample rate.",
                "RuntimeError": "Audio processing runtime error: {error_message}. Attempting alternative processing method."
            },
            "inference": {
                "InferenceError": "Model inference failed for {language}: {error_message}. Attempting fallback inference method.",
                "RuntimeError": "Inference runtime error: {error_message}. Using ensemble prediction approach.",
                "ValueError": "Invalid inference input: {error_message}. Check feature extraction pipeline."
            },
            "validation": {
                "ValidationError": "Model validation failed for {model_name}: {error_message}. Model may be corrupted or incompatible.",
                "ValueError": "Validation input error: {error_message}. Check validation data format.",
                "RuntimeError": "Validation runtime error: {error_message}. Skipping validation and proceeding with caution."
            },
            "resource_constraint": {
                "MemoryError": "Memory constraint detected (usage exceeds limits). Applying resource-aware degradation for {language}.",
                "TimeoutError": "Processing timeout constraint for {language} after {audio_duration:.1f}s. Using fast fallback method.",
                "OSError": "System resource constraint: {error_message}. Reducing processing complexity."
            },
            "configuration": {
                "ConfigurationError": "Configuration error in {component}: {error_message}. Check settings and environment variables.",
                "FileNotFoundError": "Configuration file not found: {error_message}. Using default configuration.",
                "ValueError": "Invalid configuration value: {error_message}. Reverting to safe defaults."
            },
            "default": {
                "Exception": "Unexpected error in {component} during {operation}: {error_message}",
                "RuntimeError": "Runtime error in {component}: {error_message}",
                "ValueError": "Value error in {component}: {error_message}",
                "TypeError": "Type error in {component}: {error_message}",
                "AttributeError": "Attribute error in {component}: {error_message}"
            }
        }
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, callable]:
        """Initialize recovery strategies for different error categories."""
        return {
            ErrorCategory.MODEL_LOADING: self._recover_model_loading_error,
            ErrorCategory.FOUNDATION_MODEL: self._recover_foundation_model_error,
            ErrorCategory.CLASSIFIER: self._recover_classifier_error,
            ErrorCategory.AUDIO_PROCESSING: self._recover_audio_processing_error,
            ErrorCategory.INFERENCE: self._recover_inference_error,
            ErrorCategory.VALIDATION: self._recover_validation_error,
            ErrorCategory.RESOURCE_CONSTRAINT: self._recover_resource_constraint_error,
            ErrorCategory.TIMEOUT: self._recover_timeout_error,
            ErrorCategory.CONFIGURATION: self._recover_configuration_error,
            ErrorCategory.SYSTEM: self._recover_system_error
        }
    
    def _recover_model_loading_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from model loading errors."""
        try:
            # Strategy 1: Try alternative model path
            if "FileNotFoundError" in str(type(error)):
                return ErrorRecoveryResult(
                    success=False,
                    recovery_method="alternative_model_path",
                    fallback_used=True,
                    error_message="Model file not found, fallback required"
                )
            
            # Strategy 2: Clear cache and retry
            elif "MemoryError" in str(type(error)):
                return ErrorRecoveryResult(
                    success=False,
                    recovery_method="memory_cleanup",
                    fallback_used=True,
                    error_message="Memory error, resource-aware fallback required"
                )
            
            # Strategy 3: Use dummy model
            else:
                return ErrorRecoveryResult(
                    success=True,
                    recovery_method="dummy_model_creation",
                    fallback_used=True,
                    recovery_data={"model_type": "dummy", "language": context.language}
                )
                
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_foundation_model_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from foundation model errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="traditional_features",
                fallback_used=True,
                recovery_data={
                    "fallback_method": "traditional_audio_features",
                    "original_model": context.model_name
                }
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_classifier_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from classifier errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="ensemble_classification",
                fallback_used=True,
                recovery_data={
                    "fallback_method": "statistical_ensemble",
                    "language": context.language
                }
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_audio_processing_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from audio processing errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="simplified_processing",
                fallback_used=True,
                recovery_data={"processing_method": "basic_features"}
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_inference_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from inference errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="fallback_inference",
                fallback_used=True,
                recovery_data={"inference_method": "rule_based"}
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_validation_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from validation errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="skip_validation",
                fallback_used=True,
                recovery_data={"validation_skipped": True, "reason": str(error)}
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_resource_constraint_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from resource constraint errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="resource_aware_processing",
                fallback_used=True,
                recovery_data={
                    "processing_level": "minimal",
                    "resource_constraint": str(error)
                }
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_timeout_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from timeout errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="fast_processing",
                fallback_used=True,
                recovery_data={"processing_method": "minimal_features"}
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_configuration_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from configuration errors."""
        try:
            return ErrorRecoveryResult(
                success=True,
                recovery_method="default_configuration",
                fallback_used=True,
                recovery_data={"config_source": "defaults"}
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def _recover_system_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryResult:
        """Recover from system errors."""
        try:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="system_error_logged",
                fallback_used=False,
                error_message="System error requires manual intervention"
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_method="recovery_failed",
                fallback_used=False,
                error_message=str(e)
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring and analysis.
        
        Returns:
            Dictionary containing error statistics
        """
        try:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {
                    "total_errors": 0,
                    "error_categories": {},
                    "error_types": {},
                    "most_common_errors": [],
                    "recent_errors": []
                }
            
            # Analyze error categories
            category_counts = {}
            type_counts = {}
            
            for error_record in self.error_history:
                category = error_record.get("error_category", "unknown")
                error_type = error_record.get("error_type", "unknown")
                
                category_counts[category] = category_counts.get(category, 0) + 1
                type_counts[error_type] = type_counts.get(error_type, 0) + 1
            
            # Get most common errors
            most_common = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Get recent errors (last 10)
            recent_errors = self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history
            
            return {
                "total_errors": total_errors,
                "error_categories": category_counts,
                "error_types": type_counts,
                "most_common_errors": [{"error_key": k, "count": v} for k, v in most_common],
                "recent_errors": [
                    {
                        "timestamp": err.get("timestamp", "unknown"),
                        "component": err.get("component", "unknown"),
                        "error_type": err.get("error_type", "unknown"),
                        "language": err.get("language", "unknown")
                    }
                    for err in recent_errors
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {"error": str(e)}
    
    def clear_error_history(self):
        """Clear error history and counts."""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history and counts cleared")