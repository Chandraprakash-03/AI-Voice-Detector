"""
Error recovery strategies for AI voice detection system.

This module implements error recovery mechanisms for various failure
conditions with systematic recovery approaches and fallback coordination.
"""

import logging
import time
from typing import Dict, Optional, List, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RecoveryLevel(Enum):
    """Recovery levels from least to most intrusive."""
    RETRY = "retry"
    ALTERNATIVE_METHOD = "alternative_method"
    DEGRADED_SERVICE = "degraded_service"
    FALLBACK_MODE = "fallback_mode"
    EMERGENCY_MODE = "emergency_mode"


class RecoveryOutcome(Enum):
    """Possible outcomes of recovery attempts."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ESCALATION_NEEDED = "escalation_needed"


@dataclass
class RecoveryContext:
    """Context information for recovery operations."""
    component: str
    operation: str
    error_type: str
    error_message: str
    language: Optional[str] = None
    model_name: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    resource_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_constraints is None:
            self.resource_constraints = {}


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    outcome: RecoveryOutcome
    recovery_level: RecoveryLevel
    recovery_method: str
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    recovery_data: Dict[str, Any] = None
    next_action: Optional[str] = None
    
    def __post_init__(self):
        if self.recovery_data is None:
            self.recovery_data = {}


class RecoveryStrategy(ABC):
    """
    Abstract base class for recovery strategies.
    
    Defines the interface for implementing specific recovery strategies
    for different types of errors and failure conditions.
    """
    
    @abstractmethod
    def can_handle(self, error_type: str, context: RecoveryContext) -> bool:
        """
        Check if this strategy can handle the given error type.
        
        Args:
            error_type: Type of error to handle
            context: Recovery context information
            
        Returns:
            True if this strategy can handle the error
        """
        pass
    
    @abstractmethod
    def attempt_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """
        Attempt recovery using this strategy.
        
        Args:
            context: Recovery context information
            
        Returns:
            RecoveryResult with outcome information
        """
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        Get priority of this strategy (lower numbers = higher priority).
        
        Returns:
            Priority value
        """
        pass


class RetryStrategy(RecoveryStrategy):
    """
    Simple retry strategy with exponential backoff.
    
    Attempts to retry the failed operation with increasing delays
    between attempts to handle transient failures.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def can_handle(self, error_type: str, context: RecoveryContext) -> bool:
        """Check if retry is appropriate for this error type."""
        # Retry is appropriate for transient errors
        transient_errors = [
            "TimeoutError",
            "ConnectionError",
            "TemporaryFailure",
            "ResourceBusy",
            "MemoryError"  # Sometimes transient
        ]
        
        return (error_type in transient_errors and 
                context.attempt_count < self.max_retries)
    
    def attempt_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery by retrying the operation."""
        start_time = time.time()
        
        try:
            # Calculate delay with exponential backoff
            delay = self.base_delay * (2 ** context.attempt_count)
            
            logger.info(
                f"Retrying {context.operation} in {context.component} "
                f"(attempt {context.attempt_count + 1}/{self.max_retries}) "
                f"after {delay:.1f}s delay"
            )
            
            # Apply delay
            time.sleep(delay)
            
            # Note: Actual retry logic would be implemented by the calling component
            # This strategy just provides the framework and timing
            
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.SUCCESS,
                recovery_level=RecoveryLevel.RETRY,
                recovery_method="exponential_backoff_retry",
                success=True,
                processing_time=processing_time,
                recovery_data={
                    "attempt_number": context.attempt_count + 1,
                    "delay_applied": delay,
                    "max_retries": self.max_retries
                },
                next_action="retry_operation"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.RETRY,
                recovery_method="exponential_backoff_retry",
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def get_priority(self) -> int:
        """Retry has high priority as it's least intrusive."""
        return 1


class AlternativeMethodStrategy(RecoveryStrategy):
    """
    Strategy that attempts alternative methods for the same operation.
    
    When the primary method fails, this strategy attempts to use
    alternative approaches to achieve the same result.
    """
    
    def __init__(self):
        """Initialize alternative method strategy."""
        # Define alternative methods for different operations
        self.alternative_methods = {
            "foundation_model_embedding": [
                "alternative_foundation_model",
                "traditional_audio_features",
                "cached_embeddings"
            ],
            "classifier_inference": [
                "ensemble_classification",
                "statistical_classification",
                "rule_based_classification"
            ],
            "audio_processing": [
                "simplified_processing",
                "basic_feature_extraction",
                "minimal_preprocessing"
            ],
            "model_loading": [
                "cached_model",
                "dummy_model",
                "default_model"
            ]
        }
    
    def can_handle(self, error_type: str, context: RecoveryContext) -> bool:
        """Check if alternative methods are available."""
        return context.operation in self.alternative_methods
    
    def attempt_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery using alternative methods."""
        start_time = time.time()
        
        try:
            available_methods = self.alternative_methods.get(context.operation, [])
            
            if not available_methods:
                return RecoveryResult(
                    outcome=RecoveryOutcome.FAILURE,
                    recovery_level=RecoveryLevel.ALTERNATIVE_METHOD,
                    recovery_method="no_alternatives_available",
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="No alternative methods available"
                )
            
            # Select best alternative based on context
            selected_method = self._select_best_alternative(available_methods, context)
            
            logger.info(
                f"Attempting alternative method '{selected_method}' for {context.operation} "
                f"in {context.component}"
            )
            
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.SUCCESS,
                recovery_level=RecoveryLevel.ALTERNATIVE_METHOD,
                recovery_method=selected_method,
                success=True,
                processing_time=processing_time,
                recovery_data={
                    "original_operation": context.operation,
                    "alternative_method": selected_method,
                    "available_alternatives": available_methods
                },
                next_action=f"use_alternative_{selected_method}"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.ALTERNATIVE_METHOD,
                recovery_method="alternative_method_failed",
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _select_best_alternative(self, available_methods: List[str], 
                               context: RecoveryContext) -> str:
        """Select the best alternative method based on context."""
        # Simple selection logic - could be enhanced with more sophisticated criteria
        
        # Prefer methods based on resource constraints
        if context.resource_constraints.get("memory_limited", False):
            memory_efficient = ["traditional_audio_features", "basic_feature_extraction", "minimal_preprocessing"]
            for method in memory_efficient:
                if method in available_methods:
                    return method
        
        if context.resource_constraints.get("time_limited", False):
            fast_methods = ["cached_embeddings", "cached_model", "rule_based_classification"]
            for method in fast_methods:
                if method in available_methods:
                    return method
        
        # Default to first available method
        return available_methods[0]
    
    def get_priority(self) -> int:
        """Alternative methods have medium priority."""
        return 2


class DegradedServiceStrategy(RecoveryStrategy):
    """
    Strategy that provides degraded but functional service.
    
    When normal operation fails, this strategy reduces functionality
    or quality to maintain basic service availability.
    """
    
    def can_handle(self, error_type: str, context: RecoveryContext) -> bool:
        """Check if degraded service is appropriate."""
        # Degraded service is appropriate for resource constraints and quality issues
        degradable_errors = [
            "MemoryError",
            "TimeoutError",
            "PerformanceError",
            "QualityError",
            "ResourceConstraint"
        ]
        
        return error_type in degradable_errors
    
    def attempt_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery with degraded service."""
        start_time = time.time()
        
        try:
            # Determine degradation strategy based on error type and context
            degradation_strategy = self._determine_degradation_strategy(context)
            
            logger.info(
                f"Applying degraded service strategy '{degradation_strategy}' "
                f"for {context.operation} in {context.component}"
            )
            
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.PARTIAL_SUCCESS,
                recovery_level=RecoveryLevel.DEGRADED_SERVICE,
                recovery_method=degradation_strategy,
                success=True,
                processing_time=processing_time,
                recovery_data={
                    "degradation_strategy": degradation_strategy,
                    "service_level": "degraded",
                    "quality_impact": "reduced"
                },
                next_action=f"apply_degradation_{degradation_strategy}"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.DEGRADED_SERVICE,
                recovery_method="degradation_failed",
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _determine_degradation_strategy(self, context: RecoveryContext) -> str:
        """Determine appropriate degradation strategy."""
        if context.error_type == "MemoryError":
            return "reduce_model_complexity"
        elif context.error_type == "TimeoutError":
            return "reduce_processing_time"
        elif context.error_type == "PerformanceError":
            return "reduce_accuracy_for_speed"
        elif context.error_type == "QualityError":
            return "accept_lower_quality"
        else:
            return "general_degradation"
    
    def get_priority(self) -> int:
        """Degraded service has lower priority than alternatives."""
        return 3


class FallbackModeStrategy(RecoveryStrategy):
    """
    Strategy that switches to fallback mode with basic functionality.
    
    When primary systems fail, this strategy activates fallback systems
    that provide basic functionality with reduced capabilities.
    """
    
    def can_handle(self, error_type: str, context: RecoveryContext) -> bool:
        """Check if fallback mode is appropriate."""
        # Fallback mode is appropriate for system failures
        fallback_appropriate = [
            "ModelLoadingError",
            "InferenceError",
            "SystemError",
            "ConfigurationError",
            "CriticalError"
        ]
        
        return error_type in fallback_appropriate
    
    def attempt_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery using fallback mode."""
        start_time = time.time()
        
        try:
            # Determine fallback mode based on component and operation
            fallback_mode = self._determine_fallback_mode(context)
            
            logger.warning(
                f"Activating fallback mode '{fallback_mode}' for {context.operation} "
                f"in {context.component}"
            )
            
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.PARTIAL_SUCCESS,
                recovery_level=RecoveryLevel.FALLBACK_MODE,
                recovery_method=fallback_mode,
                success=True,
                processing_time=processing_time,
                recovery_data={
                    "fallback_mode": fallback_mode,
                    "service_level": "basic",
                    "functionality": "limited"
                },
                next_action=f"activate_fallback_{fallback_mode}"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.FALLBACK_MODE,
                recovery_method="fallback_activation_failed",
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _determine_fallback_mode(self, context: RecoveryContext) -> str:
        """Determine appropriate fallback mode."""
        component_fallbacks = {
            "detection_engine": "statistical_detection",
            "foundation_model": "traditional_features",
            "classifier": "rule_based_classification",
            "audio_processor": "basic_processing",
            "model_validator": "skip_validation"
        }
        
        return component_fallbacks.get(context.component, "emergency_mode")
    
    def get_priority(self) -> int:
        """Fallback mode has low priority."""
        return 4


class EmergencyModeStrategy(RecoveryStrategy):
    """
    Emergency strategy that provides minimal functionality.
    
    This is the last resort strategy that provides the absolute minimum
    functionality to prevent complete system failure.
    """
    
    def can_handle(self, error_type: str, context: RecoveryContext) -> bool:
        """Emergency mode can handle any error as last resort."""
        return True
    
    def attempt_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery using emergency mode."""
        start_time = time.time()
        
        try:
            logger.critical(
                f"Activating emergency mode for {context.operation} in {context.component} "
                f"due to {context.error_type}: {context.error_message}"
            )
            
            # Emergency mode provides conservative defaults
            emergency_response = self._get_emergency_response(context)
            
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.PARTIAL_SUCCESS,
                recovery_level=RecoveryLevel.EMERGENCY_MODE,
                recovery_method="emergency_response",
                success=True,
                processing_time=processing_time,
                recovery_data={
                    "emergency_response": emergency_response,
                    "service_level": "minimal",
                    "reliability": "conservative"
                },
                next_action="use_emergency_response"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.EMERGENCY_MODE,
                recovery_method="emergency_mode_failed",
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _get_emergency_response(self, context: RecoveryContext) -> Dict[str, Any]:
        """Get emergency response for the given context."""
        # Conservative defaults for different operations
        emergency_responses = {
            "voice_detection": {
                "classification": "HUMAN",  # Conservative default
                "confidence": 0.1,
                "method": "emergency_default"
            },
            "model_loading": {
                "use_dummy": True,
                "model_type": "minimal"
            },
            "audio_processing": {
                "skip_complex_processing": True,
                "use_basic_features": True
            },
            "inference": {
                "use_rule_based": True,
                "confidence": 0.1
            }
        }
        
        return emergency_responses.get(context.operation, {"status": "emergency_mode_active"})
    
    def get_priority(self) -> int:
        """Emergency mode has lowest priority (highest number)."""
        return 10


class RecoveryCoordinator:
    """
    Coordinates recovery strategies and manages recovery attempts.
    
    This class manages multiple recovery strategies and coordinates
    their application based on error types and context.
    """
    
    def __init__(self):
        """Initialize recovery coordinator with default strategies."""
        self.strategies: List[RecoveryStrategy] = [
            RetryStrategy(),
            AlternativeMethodStrategy(),
            DegradedServiceStrategy(),
            FallbackModeStrategy(),
            EmergencyModeStrategy()
        ]
        
        # Sort strategies by priority
        self.strategies.sort(key=lambda s: s.get_priority())
        
        logger.info(f"RecoveryCoordinator initialized with {len(self.strategies)} strategies")
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """
        Add a custom recovery strategy.
        
        Args:
            strategy: Recovery strategy to add
        """
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.get_priority())
        logger.info(f"Added recovery strategy: {type(strategy).__name__}")
    
    def attempt_recovery(self, error_type: str, context: RecoveryContext) -> RecoveryResult:
        """
        Attempt recovery using appropriate strategies.
        
        Args:
            error_type: Type of error to recover from
            context: Recovery context information
            
        Returns:
            RecoveryResult with outcome information
        """
        start_time = time.time()
        
        try:
            # Find applicable strategies
            applicable_strategies = [
                strategy for strategy in self.strategies
                if strategy.can_handle(error_type, context)
            ]
            
            if not applicable_strategies:
                logger.error(f"No recovery strategies available for {error_type}")
                return RecoveryResult(
                    outcome=RecoveryOutcome.FAILURE,
                    recovery_level=RecoveryLevel.EMERGENCY_MODE,
                    recovery_method="no_strategies_available",
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="No applicable recovery strategies found"
                )
            
            # Try strategies in priority order
            for strategy in applicable_strategies:
                logger.info(
                    f"Attempting recovery with {type(strategy).__name__} "
                    f"for {error_type} in {context.component}"
                )
                
                try:
                    result = strategy.attempt_recovery(context)
                    
                    if result.success or result.outcome == RecoveryOutcome.PARTIAL_SUCCESS:
                        logger.info(
                            f"Recovery successful using {type(strategy).__name__}: "
                            f"{result.recovery_method}"
                        )
                        return result
                    else:
                        logger.warning(
                            f"Recovery failed with {type(strategy).__name__}: "
                            f"{result.error_message}"
                        )
                        
                except Exception as strategy_error:
                    logger.error(
                        f"Recovery strategy {type(strategy).__name__} failed: "
                        f"{str(strategy_error)}"
                    )
                    continue
            
            # All strategies failed
            logger.error(f"All recovery strategies failed for {error_type}")
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.EMERGENCY_MODE,
                recovery_method="all_strategies_failed",
                success=False,
                processing_time=time.time() - start_time,
                error_message="All recovery strategies failed"
            )
            
        except Exception as e:
            logger.error(f"Recovery coordination failed: {str(e)}")
            return RecoveryResult(
                outcome=RecoveryOutcome.FAILURE,
                recovery_level=RecoveryLevel.EMERGENCY_MODE,
                recovery_method="coordination_failed",
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recovery strategies.
        
        Returns:
            Dictionary containing strategy statistics
        """
        return {
            "total_strategies": len(self.strategies),
            "strategies": [
                {
                    "name": type(strategy).__name__,
                    "priority": strategy.get_priority(),
                    "type": type(strategy).__name__
                }
                for strategy in self.strategies
            ]
        }