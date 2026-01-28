"""
Model Lifecycle Management for AI Voice Detection system.

This module provides advanced lifecycle management features including
automated A/B testing, rollback capabilities, and backward compatibility validation.
It works in conjunction with the ModelRegistry to provide comprehensive
model management throughout the entire model lifecycle.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import statistics
import numpy as np

from .model_registry import ModelRegistry, ModelRecord, ABTestRecord, ABTestStatus, ModelStatus
from src.models.schemas import PerformanceMetrics


logger = logging.getLogger(__name__)


class LifecycleStage(str, Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "DEVELOPMENT"
    TESTING = "TESTING"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


class RollbackReason(str, Enum):
    """Reasons for model rollback."""
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    COMPATIBILITY_ISSUE = "COMPATIBILITY_ISSUE"
    STABILITY_ISSUE = "STABILITY_ISSUE"
    MANUAL_REQUEST = "MANUAL_REQUEST"
    FAILED_AB_TEST = "FAILED_AB_TEST"


@dataclass
class LifecyclePolicy:
    """
    Policy configuration for model lifecycle management.
    
    Defines rules and thresholds for automated lifecycle decisions.
    """
    min_accuracy_threshold: float = 0.85
    min_f1_threshold: float = 0.80
    min_sample_count_for_promotion: int = 1000
    ab_test_duration_days: int = 7
    ab_test_min_samples: int = 500
    ab_test_confidence_threshold: float = 0.95
    auto_rollback_enabled: bool = True
    rollback_accuracy_threshold: float = 0.80
    backup_retention_days: int = 30
    staging_duration_days: int = 3


@dataclass
class LifecycleEvent:
    """
    Record of lifecycle events for audit trail.
    """
    event_id: str
    model_id: str
    event_type: str
    timestamp: datetime
    old_stage: Optional[LifecycleStage]
    new_stage: LifecycleStage
    reason: str
    metadata: Dict[str, Any]
    triggered_by: str  # "system" or user identifier


class ModelLifecycleManager:
    """
    Advanced model lifecycle management system.
    
    Provides automated A/B testing, rollback capabilities, and backward
    compatibility validation. Manages the complete lifecycle of models
    from development through production deployment and eventual deprecation.
    """
    
    def __init__(self, 
                 registry: ModelRegistry,
                 policy: Optional[LifecyclePolicy] = None):
        """
        Initialize ModelLifecycleManager.
        
        Args:
            registry: ModelRegistry instance
            policy: Lifecycle policy configuration
        """
        self.registry = registry
        self.policy = policy or LifecyclePolicy()
        self._lifecycle_events: List[LifecycleEvent] = []
        self._active_monitors: Dict[str, Callable] = {}
        
        logger.info("ModelLifecycleManager initialized")
    
    # A/B Testing Support
    
    def start_ab_test(self,
                     control_model_id: str,
                     test_model_id: str,
                     language: Optional[str] = None,
                     traffic_split: float = 0.1,
                     duration_days: Optional[int] = None,
                     success_criteria: Optional[Dict[str, float]] = None) -> str:
        """
        Start automated A/B test between two models.
        
        Args:
            control_model_id: Current production model
            test_model_id: New model to test
            language: Language to test (None for all)
            traffic_split: Percentage of traffic to test model (0.0-1.0)
            duration_days: Test duration in days
            success_criteria: Custom success criteria thresholds
            
        Returns:
            A/B test ID
            
        Raises:
            ValueError: If models are incompatible or invalid parameters
        """
        # Validate model compatibility
        is_compatible, issues = self.registry.validate_model_compatibility(
            control_model_id, test_model_id
        )
        if not is_compatible:
            raise ValueError(f"Models incompatible for A/B test: {', '.join(issues)}")
        
        # Validate models are in appropriate stages
        control_model = self.registry.get_model(control_model_id)
        test_model = self.registry.get_model(test_model_id)
        
        if control_model.status != ModelStatus.ACTIVE:
            raise ValueError(f"Control model {control_model_id} must be ACTIVE")
        
        if test_model.status not in [ModelStatus.TESTING, ModelStatus.ACTIVE]:
            raise ValueError(f"Test model {test_model_id} must be TESTING or ACTIVE")
        
        # Generate test ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_id = f"ab_test_{control_model_id}_{test_model_id}_{timestamp}"
        
        # Set up test configuration
        test_config = {
            "duration_days": duration_days or self.policy.ab_test_duration_days,
            "success_criteria": success_criteria or {
                "min_accuracy": self.policy.min_accuracy_threshold,
                "min_f1_score": self.policy.min_f1_threshold,
                "min_samples": self.policy.ab_test_min_samples
            },
            "auto_promote": True,
            "auto_rollback": self.policy.auto_rollback_enabled
        }
        
        # Create A/B test
        ab_test = self.registry.create_ab_test(
            test_id=test_id,
            model_a_id=control_model_id,
            model_b_id=test_model_id,
            traffic_split=traffic_split,
            language=language,
            test_config=test_config
        )
        
        # Record lifecycle event
        self._record_event(
            model_id=test_model_id,
            event_type="AB_TEST_STARTED",
            old_stage=None,
            new_stage=LifecycleStage.TESTING,
            reason=f"A/B test started against {control_model_id}",
            metadata={"ab_test_id": test_id, "traffic_split": traffic_split},
            triggered_by="system"
        )
        
        logger.info(f"Started A/B test {test_id}: {control_model_id} vs {test_model_id}")
        return test_id
    
    def evaluate_ab_test(self, test_id: str) -> Dict[str, Any]:
        """
        Evaluate A/B test results and determine winner.
        
        Args:
            test_id: A/B test identifier
            
        Returns:
            Dictionary with evaluation results and recommendations
        """
        ab_test = self.registry.get_ab_test(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")
        
        if ab_test.status != ABTestStatus.RUNNING:
            raise ValueError(f"A/B test {test_id} is not running")
        
        # Check if test has sufficient data
        min_samples = ab_test.test_config.get("min_samples", self.policy.ab_test_min_samples)
        if ab_test.sample_count_a < min_samples or ab_test.sample_count_b < min_samples:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {min_samples} samples per model",
                "samples_a": ab_test.sample_count_a,
                "samples_b": ab_test.sample_count_b,
                "recommendation": "continue_test"
            }
        
        # Compare metrics
        metrics_a = ab_test.metrics_a
        metrics_b = ab_test.metrics_b
        
        if not metrics_a or not metrics_b:
            return {
                "status": "missing_metrics",
                "message": "Metrics not available for both models",
                "recommendation": "continue_test"
            }
        
        # Statistical significance test (simplified)
        accuracy_diff = metrics_b.accuracy - metrics_a.accuracy
        f1_diff = metrics_b.f1_score - metrics_a.f1_score
        
        # Determine winner based on success criteria
        success_criteria = ab_test.test_config.get("success_criteria", {})
        min_accuracy = success_criteria.get("min_accuracy", self.policy.min_accuracy_threshold)
        min_f1 = success_criteria.get("min_f1_score", self.policy.min_f1_threshold)
        
        model_b_meets_criteria = (
            metrics_b.accuracy >= min_accuracy and
            metrics_b.f1_score >= min_f1
        )
        
        # Determine statistical significance (simplified)
        is_significant = abs(accuracy_diff) > 0.02 or abs(f1_diff) > 0.02
        confidence_level = 0.95 if is_significant else 0.80  # Simplified
        
        # Make recommendation
        if model_b_meets_criteria and accuracy_diff > 0 and f1_diff > 0:
            winner = ab_test.model_b_id
            recommendation = "promote_model_b"
        elif accuracy_diff < -0.05 or f1_diff < -0.05:  # Significant degradation
            winner = ab_test.model_a_id
            recommendation = "rollback_to_model_a"
        else:
            winner = None
            recommendation = "no_clear_winner"
        
        return {
            "status": "evaluation_complete",
            "winner": winner,
            "confidence_level": confidence_level,
            "accuracy_diff": accuracy_diff,
            "f1_diff": f1_diff,
            "model_a_metrics": metrics_a.model_dump(),
            "model_b_metrics": metrics_b.model_dump(),
            "recommendation": recommendation,
            "is_significant": is_significant
        }
    
    def complete_ab_test(self, 
                        test_id: str, 
                        auto_promote: bool = True) -> Dict[str, Any]:
        """
        Complete A/B test and optionally promote winner.
        
        Args:
            test_id: A/B test identifier
            auto_promote: Whether to automatically promote winner
            
        Returns:
            Dictionary with completion results
        """
        evaluation = self.evaluate_ab_test(test_id)
        
        if evaluation["status"] != "evaluation_complete":
            return evaluation
        
        ab_test = self.registry.get_ab_test(test_id)
        winner = evaluation["winner"]
        
        # Complete the A/B test
        self.registry.complete_ab_test(
            test_id=test_id,
            winner_model_id=winner,
            confidence_level=evaluation["confidence_level"]
        )
        
        # Auto-promote if enabled and there's a clear winner
        if auto_promote and winner and evaluation["recommendation"] == "promote_model_b":
            self.promote_model_to_production(winner, f"A/B test winner: {test_id}")
            
            # Demote the old model
            self.demote_model_from_production(
                ab_test.model_a_id, 
                f"Replaced by A/B test winner: {test_id}"
            )
        
        # Record lifecycle event
        self._record_event(
            model_id=ab_test.model_b_id,
            event_type="AB_TEST_COMPLETED",
            old_stage=LifecycleStage.TESTING,
            new_stage=LifecycleStage.PRODUCTION if winner == ab_test.model_b_id else LifecycleStage.TESTING,
            reason=f"A/B test completed, winner: {winner}",
            metadata={"ab_test_id": test_id, "evaluation": evaluation},
            triggered_by="system"
        )
        
        logger.info(f"Completed A/B test {test_id}, winner: {winner}")
        return {**evaluation, "test_completed": True, "auto_promoted": auto_promote and winner}
    
    # Model Promotion and Demotion
    
    def promote_model_to_production(self, 
                                   model_id: str, 
                                   reason: str = "Manual promotion") -> bool:
        """
        Promote model to production status.
        
        Args:
            model_id: Model identifier
            reason: Reason for promotion
            
        Returns:
            True if promotion successful
        """
        model = self.registry.get_model(model_id)
        if not model:
            logger.error(f"Model {model_id} not found for promotion")
            return False
        
        # Validate model meets promotion criteria
        if not self._validate_promotion_criteria(model):
            logger.error(f"Model {model_id} does not meet promotion criteria")
            return False
        
        # Create backup before promotion
        try:
            backup_path = self.registry.create_model_backup(model_id)
            logger.info(f"Created backup before promotion: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup for {model_id}: {e}")
            return False
        
        # Update model status
        old_status = model.status
        success = self.registry.update_model_status(model_id, ModelStatus.ACTIVE)
        
        if success:
            self._record_event(
                model_id=model_id,
                event_type="MODEL_PROMOTED",
                old_stage=self._status_to_stage(old_status),
                new_stage=LifecycleStage.PRODUCTION,
                reason=reason,
                metadata={"backup_path": backup_path},
                triggered_by="system"
            )
            
            logger.info(f"Promoted model {model_id} to production")
        
        return success
    
    def demote_model_from_production(self, 
                                    model_id: str, 
                                    reason: str = "Manual demotion") -> bool:
        """
        Demote model from production status.
        
        Args:
            model_id: Model identifier
            reason: Reason for demotion
            
        Returns:
            True if demotion successful
        """
        model = self.registry.get_model(model_id)
        if not model:
            logger.error(f"Model {model_id} not found for demotion")
            return False
        
        old_status = model.status
        success = self.registry.update_model_status(model_id, ModelStatus.DEPRECATED)
        
        if success:
            self._record_event(
                model_id=model_id,
                event_type="MODEL_DEMOTED",
                old_stage=self._status_to_stage(old_status),
                new_stage=LifecycleStage.DEPRECATED,
                reason=reason,
                metadata={},
                triggered_by="system"
            )
            
            logger.info(f"Demoted model {model_id} from production")
        
        return success
    
    # Rollback Capabilities
    
    def rollback_model(self, 
                      model_id: str, 
                      reason: RollbackReason = RollbackReason.MANUAL_REQUEST,
                      backup_path: Optional[str] = None) -> bool:
        """
        Rollback model to previous version.
        
        Args:
            model_id: Model identifier
            reason: Reason for rollback
            backup_path: Specific backup to restore (optional)
            
        Returns:
            True if rollback successful
        """
        model = self.registry.get_model(model_id)
        if not model:
            logger.error(f"Model {model_id} not found for rollback")
            return False
        
        # Find appropriate backup if not specified
        if not backup_path:
            backup_path = self._find_latest_backup(model_id)
            if not backup_path:
                logger.error(f"No backup found for model {model_id}")
                return False
        
        # Perform rollback
        success = self.registry.rollback_model(model_id, backup_path)
        
        if success:
            self._record_event(
                model_id=model_id,
                event_type="MODEL_ROLLBACK",
                old_stage=self._status_to_stage(model.status),
                new_stage=LifecycleStage.PRODUCTION,  # Assume rollback restores to production
                reason=f"Rollback: {reason.value}",
                metadata={"backup_path": backup_path, "rollback_reason": reason.value},
                triggered_by="system"
            )
            
            logger.info(f"Rolled back model {model_id} due to {reason.value}")
        
        return success
    
    def auto_rollback_on_degradation(self, 
                                    model_id: str, 
                                    current_metrics: PerformanceMetrics) -> bool:
        """
        Automatically rollback model if performance degrades below threshold.
        
        Args:
            model_id: Model identifier
            current_metrics: Current performance metrics
            
        Returns:
            True if rollback was triggered
        """
        if not self.policy.auto_rollback_enabled:
            return False
        
        model = self.registry.get_model(model_id)
        if not model or model.status != ModelStatus.ACTIVE:
            return False
        
        # Check if performance is below rollback threshold
        baseline_accuracy = model.validation_metrics.accuracy
        current_accuracy = current_metrics.accuracy
        
        accuracy_degradation = baseline_accuracy - current_accuracy
        
        if (current_accuracy < self.policy.rollback_accuracy_threshold or 
            accuracy_degradation > 0.10):  # More than 10% degradation
            
            logger.warning(f"Performance degradation detected for {model_id}: "
                         f"{baseline_accuracy:.3f} -> {current_accuracy:.3f}")
            
            return self.rollback_model(
                model_id, 
                RollbackReason.PERFORMANCE_DEGRADATION
            )
        
        return False
    
    # Backward Compatibility Validation
    
    def validate_backward_compatibility(self, 
                                       old_model_id: str, 
                                       new_model_id: str) -> Tuple[bool, List[str]]:
        """
        Validate backward compatibility between model versions.
        
        Args:
            old_model_id: Previous model version
            new_model_id: New model version
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        # Use registry's compatibility validation as base
        is_compatible, issues = self.registry.validate_model_compatibility(
            old_model_id, new_model_id
        )
        
        if not is_compatible:
            return is_compatible, issues
        
        # Additional lifecycle-specific compatibility checks
        old_model = self.registry.get_model(old_model_id)
        new_model = self.registry.get_model(new_model_id)
        
        # Check performance regression
        if (new_model.validation_metrics.accuracy < 
            old_model.validation_metrics.accuracy - 0.05):
            issues.append("Significant accuracy regression detected")
        
        # Check model size increase (potential deployment issue)
        size_increase = new_model.model_size_bytes / old_model.model_size_bytes
        if size_increase > 2.0:  # More than 2x size increase
            issues.append(f"Model size increased by {size_increase:.1f}x")
        
        # Check version compatibility
        old_version_parts = old_model.version.split('.')
        new_version_parts = new_model.version.split('.')
        
        if len(old_version_parts) >= 2 and len(new_version_parts) >= 2:
            old_major = int(old_version_parts[0])
            new_major = int(new_version_parts[0])
            
            if new_major > old_major:
                issues.append("Major version change may break compatibility")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues
    
    def create_compatibility_report(self, 
                                   model_id: str) -> Dict[str, Any]:
        """
        Create comprehensive compatibility report for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with compatibility analysis
        """
        model = self.registry.get_model(model_id)
        if not model:
            return {"error": f"Model {model_id} not found"}
        
        # Get model lineage
        ancestors = self.registry.get_model_ancestry(model_id)
        descendants = self.registry.get_model_lineage(model_id)
        
        # Check compatibility with ancestors
        ancestor_compatibility = {}
        for ancestor_id in ancestors[-3:]:  # Check last 3 ancestors
            is_compatible, issues = self.validate_backward_compatibility(
                ancestor_id, model_id
            )
            ancestor_compatibility[ancestor_id] = {
                "compatible": is_compatible,
                "issues": issues
            }
        
        # Check compatibility with active models of same type/language
        active_models = self.registry.list_models(
            model_type=model.model_type,
            language=model.language,
            status=ModelStatus.ACTIVE
        )
        
        active_compatibility = {}
        for active_model in active_models:
            if active_model.model_id != model_id:
                is_compatible, issues = self.registry.validate_model_compatibility(
                    active_model.model_id, model_id
                )
                active_compatibility[active_model.model_id] = {
                    "compatible": is_compatible,
                    "issues": issues
                }
        
        return {
            "model_id": model_id,
            "model_type": model.model_type,
            "language": model.language,
            "version": model.version,
            "ancestors": ancestors,
            "descendants": descendants,
            "ancestor_compatibility": ancestor_compatibility,
            "active_model_compatibility": active_compatibility,
            "overall_compatibility_score": self._calculate_compatibility_score(
                ancestor_compatibility, active_compatibility
            )
        }
    
    # Utility Methods
    
    def _validate_promotion_criteria(self, model: ModelRecord) -> bool:
        """Validate if model meets promotion criteria."""
        metrics = model.validation_metrics
        
        return (
            metrics.accuracy >= self.policy.min_accuracy_threshold and
            metrics.f1_score >= self.policy.min_f1_threshold and
            metrics.sample_count >= self.policy.min_sample_count_for_promotion
        )
    
    def _status_to_stage(self, status: ModelStatus) -> LifecycleStage:
        """Convert ModelStatus to LifecycleStage."""
        mapping = {
            ModelStatus.TESTING: LifecycleStage.TESTING,
            ModelStatus.ACTIVE: LifecycleStage.PRODUCTION,
            ModelStatus.DEPRECATED: LifecycleStage.DEPRECATED,
            ModelStatus.ARCHIVED: LifecycleStage.ARCHIVED
        }
        return mapping.get(status, LifecycleStage.DEVELOPMENT)
    
    def _find_latest_backup(self, model_id: str) -> Optional[str]:
        """Find the latest backup file for a model."""
        backup_dir = self.registry.registry_path / "backups"
        if not backup_dir.exists():
            return None
        
        # Find backup files for this model
        backup_files = list(backup_dir.glob(f"{model_id}_*.backup"))
        if not backup_files:
            return None
        
        # Return the most recent backup
        latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
        return str(latest_backup)
    
    def _calculate_compatibility_score(self, 
                                     ancestor_compat: Dict[str, Any],
                                     active_compat: Dict[str, Any]) -> float:
        """Calculate overall compatibility score (0.0-1.0)."""
        total_checks = len(ancestor_compat) + len(active_compat)
        if total_checks == 0:
            return 1.0
        
        compatible_count = 0
        for compat_info in ancestor_compat.values():
            if compat_info["compatible"]:
                compatible_count += 1
        
        for compat_info in active_compat.values():
            if compat_info["compatible"]:
                compatible_count += 1
        
        return compatible_count / total_checks
    
    def _record_event(self, 
                     model_id: str,
                     event_type: str,
                     old_stage: Optional[LifecycleStage],
                     new_stage: LifecycleStage,
                     reason: str,
                     metadata: Dict[str, Any],
                     triggered_by: str):
        """Record lifecycle event for audit trail."""
        event = LifecycleEvent(
            event_id=f"{event_type}_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_id=model_id,
            event_type=event_type,
            timestamp=datetime.now(),
            old_stage=old_stage,
            new_stage=new_stage,
            reason=reason,
            metadata=metadata,
            triggered_by=triggered_by
        )
        
        self._lifecycle_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self._lifecycle_events) > 1000:
            self._lifecycle_events = self._lifecycle_events[-1000:]
    
    def get_lifecycle_events(self, 
                           model_id: Optional[str] = None,
                           event_type: Optional[str] = None,
                           limit: int = 100) -> List[LifecycleEvent]:
        """
        Get lifecycle events with optional filtering.
        
        Args:
            model_id: Filter by model ID
            event_type: Filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of lifecycle events
        """
        events = self._lifecycle_events
        
        if model_id:
            events = [e for e in events if e.model_id == model_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Sort by timestamp (most recent first) and limit
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit]