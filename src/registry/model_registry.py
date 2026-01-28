"""
Model Registry implementation for AI Voice Detection system.

This module provides comprehensive model metadata management, version tracking,
and lifecycle management capabilities. It supports model record storage and retrieval,
A/B testing, rollback capabilities, and backward compatibility validation.
"""

import json
import os
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from pydantic import BaseModel, Field
from src.models.schemas import ModelMetadata, PerformanceMetrics


logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model status enumeration."""
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    TESTING = "TESTING"
    ARCHIVED = "ARCHIVED"


class ABTestStatus(str, Enum):
    """A/B test status enumeration."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class ModelRecord:
    """
    Data class for model registry records.
    
    Contains comprehensive model information including metadata,
    performance metrics, and lifecycle status.
    """
    model_id: str
    model_type: str  # "foundation" or "classifier"
    language: Optional[str]
    version: str
    file_path: str
    training_date: datetime
    validation_metrics: PerformanceMetrics
    status: ModelStatus
    created_timestamp: datetime
    last_updated: datetime
    model_size_bytes: int
    training_data_hash: str
    parent_model_id: Optional[str] = None  # For model lineage
    deployment_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    additional_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metadata is None:
            self.additional_metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif key == 'validation_metrics' and value:
                # Convert PerformanceMetrics to dict
                data[key] = value.model_dump() if hasattr(value, 'model_dump') else value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRecord':
        """Create ModelRecord from dictionary."""
        # Convert ISO strings back to datetime objects
        datetime_fields = ['training_date', 'created_timestamp', 'last_updated', 
                          'deployment_date', 'deprecation_date']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert validation_metrics back to PerformanceMetrics
        if data.get('validation_metrics'):
            if isinstance(data['validation_metrics'], dict):
                data['validation_metrics'] = PerformanceMetrics(**data['validation_metrics'])
        
        # Convert status to enum
        if isinstance(data.get('status'), str):
            data['status'] = ModelStatus(data['status'])
        
        return cls(**data)


@dataclass
class ThresholdRecord:
    """
    Data class for threshold registry records.
    
    Contains language-specific threshold information and optimization history.
    """
    language: str
    model_version: str
    threshold: float
    optimization_date: datetime
    validation_f1_score: float
    precision: float
    recall: float
    sample_count: int
    optimizer_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.optimizer_config is None:
            self.optimizer_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        if data.get('optimization_date'):
            data['optimization_date'] = data['optimization_date'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdRecord':
        """Create ThresholdRecord from dictionary."""
        if data.get('optimization_date'):
            data['optimization_date'] = datetime.fromisoformat(data['optimization_date'])
        return cls(**data)


@dataclass
class ABTestRecord:
    """
    Data class for A/B test records.
    
    Contains information about model comparison tests and their results.
    """
    test_id: str
    model_a_id: str
    model_b_id: str
    language: Optional[str]
    start_date: datetime
    end_date: Optional[datetime]
    status: ABTestStatus
    traffic_split: float  # Percentage of traffic to model B (0.0-1.0)
    sample_count_a: int = 0
    sample_count_b: int = 0
    metrics_a: Optional[PerformanceMetrics] = None
    metrics_b: Optional[PerformanceMetrics] = None
    winner_model_id: Optional[str] = None
    confidence_level: Optional[float] = None
    test_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.test_config is None:
            self.test_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        datetime_fields = ['start_date', 'end_date']
        for field in datetime_fields:
            if data.get(field):
                data[field] = data[field].isoformat()
        
        # Convert metrics to dict
        for metrics_field in ['metrics_a', 'metrics_b']:
            if data.get(metrics_field):
                data[metrics_field] = data[metrics_field].model_dump() if hasattr(data[metrics_field], 'model_dump') else data[metrics_field]
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestRecord':
        """Create ABTestRecord from dictionary."""
        # Convert ISO strings back to datetime objects
        datetime_fields = ['start_date', 'end_date']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert metrics back to PerformanceMetrics
        for metrics_field in ['metrics_a', 'metrics_b']:
            if data.get(metrics_field) and isinstance(data[metrics_field], dict):
                data[metrics_field] = PerformanceMetrics(**data[metrics_field])
        
        # Convert status to enum
        if isinstance(data.get('status'), str):
            data['status'] = ABTestStatus(data['status'])
        
        return cls(**data)


class ModelRegistry:
    """
    Model Registry for managing model metadata, versions, and lifecycle.
    
    Provides comprehensive model management including:
    - Model record storage and retrieval
    - Version tracking and model lineage
    - Model status tracking (active, deprecated, testing)
    - A/B testing support for model comparisons
    - Rollback capabilities for model updates
    - Backward compatibility validation
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize ModelRegistry.
        
        Args:
            registry_path: Path to store registry data files
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry data files
        self.models_file = self.registry_path / "models.json"
        self.thresholds_file = self.registry_path / "thresholds.json"
        self.ab_tests_file = self.registry_path / "ab_tests.json"
        self.lineage_file = self.registry_path / "lineage.json"
        
        # In-memory caches
        self._models_cache: Dict[str, ModelRecord] = {}
        self._thresholds_cache: Dict[str, ThresholdRecord] = {}
        self._ab_tests_cache: Dict[str, ABTestRecord] = {}
        self._lineage_cache: Dict[str, List[str]] = {}
        
        # Load existing data
        self._load_registry_data()
        
        logger.info(f"ModelRegistry initialized with {len(self._models_cache)} models")
    
    def _load_registry_data(self):
        """Load registry data from files."""
        try:
            # Load models
            if self.models_file.exists():
                with open(self.models_file, 'r') as f:
                    models_data = json.load(f)
                    self._models_cache = {
                        model_id: ModelRecord.from_dict(data)
                        for model_id, data in models_data.items()
                    }
            
            # Load thresholds
            if self.thresholds_file.exists():
                with open(self.thresholds_file, 'r') as f:
                    thresholds_data = json.load(f)
                    self._thresholds_cache = {
                        key: ThresholdRecord.from_dict(data)
                        for key, data in thresholds_data.items()
                    }
            
            # Load A/B tests
            if self.ab_tests_file.exists():
                with open(self.ab_tests_file, 'r') as f:
                    ab_tests_data = json.load(f)
                    self._ab_tests_cache = {
                        test_id: ABTestRecord.from_dict(data)
                        for test_id, data in ab_tests_data.items()
                    }
            
            # Load lineage
            if self.lineage_file.exists():
                with open(self.lineage_file, 'r') as f:
                    self._lineage_cache = json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading registry data: {e}")
            # Initialize empty caches on error
            self._models_cache = {}
            self._thresholds_cache = {}
            self._ab_tests_cache = {}
            self._lineage_cache = {}
    
    def _save_registry_data(self):
        """Save registry data to files."""
        try:
            # Ensure registry directory exists
            self.registry_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            models_data = {
                model_id: record.to_dict()
                for model_id, record in self._models_cache.items()
            }
            with open(self.models_file, 'w') as f:
                json.dump(models_data, f, indent=2, default=str)
            
            # Save thresholds
            thresholds_data = {
                key: record.to_dict()
                for key, record in self._thresholds_cache.items()
            }
            with open(self.thresholds_file, 'w') as f:
                json.dump(thresholds_data, f, indent=2, default=str)
            
            # Save A/B tests
            ab_tests_data = {
                test_id: record.to_dict()
                for test_id, record in self._ab_tests_cache.items()
            }
            with open(self.ab_tests_file, 'w') as f:
                json.dump(ab_tests_data, f, indent=2, default=str)
            
            # Save lineage
            with open(self.lineage_file, 'w') as f:
                json.dump(self._lineage_cache, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving registry data: {e}")
            raise
    
    # Model Record Management
    
    def register_model(self, 
                      model_id: str,
                      model_type: str,
                      file_path: str,
                      validation_metrics: PerformanceMetrics,
                      language: Optional[str] = None,
                      version: str = "1.0.0",
                      training_date: Optional[datetime] = None,
                      parent_model_id: Optional[str] = None,
                      additional_metadata: Optional[Dict[str, Any]] = None) -> ModelRecord:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model ("foundation" or "classifier")
            file_path: Path to the model file
            validation_metrics: Performance metrics from validation
            language: Language for classifier models
            version: Model version string
            training_date: When the model was trained
            parent_model_id: Parent model for lineage tracking
            additional_metadata: Additional model-specific information
            
        Returns:
            ModelRecord: The created model record
            
        Raises:
            ValueError: If model_id already exists or invalid parameters
            FileNotFoundError: If model file doesn't exist
        """
        if model_id in self._models_cache:
            raise ValueError(f"Model {model_id} already exists in registry")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Calculate file size and hash
        model_size_bytes = os.path.getsize(file_path)
        training_data_hash = self._calculate_file_hash(file_path)
        
        # Create model record
        now = datetime.now()
        record = ModelRecord(
            model_id=model_id,
            model_type=model_type,
            language=language,
            version=version,
            file_path=file_path,
            training_date=training_date or now,
            validation_metrics=validation_metrics,
            status=ModelStatus.TESTING,  # New models start in testing
            created_timestamp=now,
            last_updated=now,
            model_size_bytes=model_size_bytes,
            training_data_hash=training_data_hash,
            parent_model_id=parent_model_id,
            additional_metadata=additional_metadata or {}
        )
        
        # Add to cache and save
        self._models_cache[model_id] = record
        
        # Update lineage if parent specified
        if parent_model_id:
            self._update_lineage(parent_model_id, model_id)
        
        self._save_registry_data()
        
        logger.info(f"Registered model {model_id} (type: {model_type}, language: {language})")
        return record
    
    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """
        Get model record by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelRecord or None if not found
        """
        return self._models_cache.get(model_id)
    
    def list_models(self, 
                   model_type: Optional[str] = None,
                   language: Optional[str] = None,
                   status: Optional[ModelStatus] = None) -> List[ModelRecord]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            language: Filter by language
            status: Filter by status
            
        Returns:
            List of matching model records
        """
        models = list(self._models_cache.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if language:
            models = [m for m in models if m.language == language]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda m: m.created_timestamp, reverse=True)
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """
        Update model status.
        
        Args:
            model_id: Model identifier
            status: New status
            
        Returns:
            True if updated successfully
        """
        if model_id not in self._models_cache:
            return False
        
        record = self._models_cache[model_id]
        old_status = record.status
        record.status = status
        record.last_updated = datetime.now()
        
        # Set deployment/deprecation dates
        if status == ModelStatus.ACTIVE and old_status != ModelStatus.ACTIVE:
            record.deployment_date = datetime.now()
        elif status == ModelStatus.DEPRECATED and old_status != ModelStatus.DEPRECATED:
            record.deprecation_date = datetime.now()
        
        self._save_registry_data()
        
        logger.info(f"Updated model {model_id} status: {old_status} -> {status}")
        return True
    
    def update_model_metrics(self, model_id: str, metrics: PerformanceMetrics) -> bool:
        """
        Update model performance metrics.
        
        Args:
            model_id: Model identifier
            metrics: New performance metrics
            
        Returns:
            True if updated successfully
        """
        if model_id not in self._models_cache:
            return False
        
        record = self._models_cache[model_id]
        record.validation_metrics = metrics
        record.last_updated = datetime.now()
        
        self._save_registry_data()
        
        logger.info(f"Updated metrics for model {model_id}")
        return True
    
    def delete_model(self, model_id: str, remove_file: bool = False) -> bool:
        """
        Delete model from registry.
        
        Args:
            model_id: Model identifier
            remove_file: Whether to also delete the model file
            
        Returns:
            True if deleted successfully
        """
        if model_id not in self._models_cache:
            return False
        
        record = self._models_cache[model_id]
        
        # Remove file if requested
        if remove_file and os.path.exists(record.file_path):
            try:
                os.remove(record.file_path)
                logger.info(f"Deleted model file: {record.file_path}")
            except Exception as e:
                logger.error(f"Error deleting model file {record.file_path}: {e}")
        
        # Remove from cache
        del self._models_cache[model_id]
        
        # Clean up lineage references
        self._cleanup_lineage(model_id)
        
        self._save_registry_data()
        
        logger.info(f"Deleted model {model_id} from registry")
        return True
    
    # Version and Lineage Management
    
    def get_model_lineage(self, model_id: str) -> List[str]:
        """
        Get model lineage (child models).
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of child model IDs
        """
        return self._lineage_cache.get(model_id, [])
    
    def get_model_ancestry(self, model_id: str) -> List[str]:
        """
        Get model ancestry (parent models).
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of ancestor model IDs (from oldest to newest)
        """
        ancestry = []
        current_id = model_id
        
        while current_id:
            record = self.get_model(current_id)
            if not record or not record.parent_model_id:
                break
            ancestry.append(record.parent_model_id)
            current_id = record.parent_model_id
        
        return list(reversed(ancestry))  # Oldest first
    
    def _update_lineage(self, parent_id: str, child_id: str):
        """Update lineage tracking."""
        if parent_id not in self._lineage_cache:
            self._lineage_cache[parent_id] = []
        
        if child_id not in self._lineage_cache[parent_id]:
            self._lineage_cache[parent_id].append(child_id)
    
    def _cleanup_lineage(self, model_id: str):
        """Clean up lineage references for deleted model."""
        # Remove as parent
        if model_id in self._lineage_cache:
            del self._lineage_cache[model_id]
        
        # Remove as child
        for parent_id, children in self._lineage_cache.items():
            if model_id in children:
                children.remove(model_id)
    
    # Threshold Management
    
    def register_threshold(self, 
                          language: str,
                          model_version: str,
                          threshold: float,
                          validation_f1_score: float,
                          precision: float,
                          recall: float,
                          sample_count: int,
                          optimizer_config: Optional[Dict[str, Any]] = None) -> ThresholdRecord:
        """
        Register optimized threshold for a language/model combination.
        
        Args:
            language: Language identifier
            model_version: Model version
            threshold: Optimized threshold value
            validation_f1_score: F1 score achieved with this threshold
            precision: Precision achieved
            recall: Recall achieved
            sample_count: Number of validation samples used
            optimizer_config: Configuration used for optimization
            
        Returns:
            ThresholdRecord: The created threshold record
        """
        key = f"{language}_{model_version}"
        
        record = ThresholdRecord(
            language=language,
            model_version=model_version,
            threshold=threshold,
            optimization_date=datetime.now(),
            validation_f1_score=validation_f1_score,
            precision=precision,
            recall=recall,
            sample_count=sample_count,
            optimizer_config=optimizer_config or {}
        )
        
        self._thresholds_cache[key] = record
        self._save_registry_data()
        
        logger.info(f"Registered threshold for {language} v{model_version}: {threshold:.4f}")
        return record
    
    def get_threshold(self, language: str, model_version: str) -> Optional[ThresholdRecord]:
        """
        Get threshold record for language/model combination.
        
        Args:
            language: Language identifier
            model_version: Model version
            
        Returns:
            ThresholdRecord or None if not found
        """
        key = f"{language}_{model_version}"
        return self._thresholds_cache.get(key)
    
    def list_thresholds(self, language: Optional[str] = None) -> List[ThresholdRecord]:
        """
        List threshold records with optional language filtering.
        
        Args:
            language: Filter by language
            
        Returns:
            List of threshold records
        """
        thresholds = list(self._thresholds_cache.values())
        
        if language:
            thresholds = [t for t in thresholds if t.language == language]
        
        return sorted(thresholds, key=lambda t: t.optimization_date, reverse=True)
    
    # A/B Testing Support
    
    def create_ab_test(self,
                      test_id: str,
                      model_a_id: str,
                      model_b_id: str,
                      traffic_split: float = 0.5,
                      language: Optional[str] = None,
                      test_config: Optional[Dict[str, Any]] = None) -> ABTestRecord:
        """
        Create A/B test for model comparison.
        
        Args:
            test_id: Unique test identifier
            model_a_id: Control model ID
            model_b_id: Test model ID
            traffic_split: Percentage of traffic to model B (0.0-1.0)
            language: Language to test (None for all)
            test_config: Additional test configuration
            
        Returns:
            ABTestRecord: The created test record
            
        Raises:
            ValueError: If test_id exists or models not found
        """
        if test_id in self._ab_tests_cache:
            raise ValueError(f"A/B test {test_id} already exists")
        
        # Validate models exist
        if model_a_id not in self._models_cache:
            raise ValueError(f"Model A {model_a_id} not found in registry")
        if model_b_id not in self._models_cache:
            raise ValueError(f"Model B {model_b_id} not found in registry")
        
        record = ABTestRecord(
            test_id=test_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            language=language,
            start_date=datetime.now(),
            end_date=None,
            status=ABTestStatus.RUNNING,
            traffic_split=traffic_split,
            test_config=test_config or {}
        )
        
        self._ab_tests_cache[test_id] = record
        self._save_registry_data()
        
        logger.info(f"Created A/B test {test_id}: {model_a_id} vs {model_b_id}")
        return record
    
    def update_ab_test_metrics(self,
                              test_id: str,
                              model_a_metrics: Optional[PerformanceMetrics] = None,
                              model_b_metrics: Optional[PerformanceMetrics] = None,
                              sample_count_a: Optional[int] = None,
                              sample_count_b: Optional[int] = None) -> bool:
        """
        Update A/B test metrics.
        
        Args:
            test_id: Test identifier
            model_a_metrics: Updated metrics for model A
            model_b_metrics: Updated metrics for model B
            sample_count_a: Sample count for model A
            sample_count_b: Sample count for model B
            
        Returns:
            True if updated successfully
        """
        if test_id not in self._ab_tests_cache:
            return False
        
        record = self._ab_tests_cache[test_id]
        
        if model_a_metrics:
            record.metrics_a = model_a_metrics
        if model_b_metrics:
            record.metrics_b = model_b_metrics
        if sample_count_a is not None:
            record.sample_count_a = sample_count_a
        if sample_count_b is not None:
            record.sample_count_b = sample_count_b
        
        self._save_registry_data()
        
        logger.info(f"Updated A/B test {test_id} metrics")
        return True
    
    def complete_ab_test(self,
                        test_id: str,
                        winner_model_id: Optional[str] = None,
                        confidence_level: Optional[float] = None) -> bool:
        """
        Complete A/B test and declare winner.
        
        Args:
            test_id: Test identifier
            winner_model_id: ID of winning model (None for no winner)
            confidence_level: Statistical confidence level
            
        Returns:
            True if completed successfully
        """
        if test_id not in self._ab_tests_cache:
            return False
        
        record = self._ab_tests_cache[test_id]
        record.status = ABTestStatus.COMPLETED
        record.end_date = datetime.now()
        record.winner_model_id = winner_model_id
        record.confidence_level = confidence_level
        
        self._save_registry_data()
        
        logger.info(f"Completed A/B test {test_id}, winner: {winner_model_id}")
        return True
    
    def get_ab_test(self, test_id: str) -> Optional[ABTestRecord]:
        """Get A/B test record by ID."""
        return self._ab_tests_cache.get(test_id)
    
    def list_ab_tests(self, status: Optional[ABTestStatus] = None) -> List[ABTestRecord]:
        """
        List A/B tests with optional status filtering.
        
        Args:
            status: Filter by test status
            
        Returns:
            List of A/B test records
        """
        tests = list(self._ab_tests_cache.values())
        
        if status:
            tests = [t for t in tests if t.status == status]
        
        return sorted(tests, key=lambda t: t.start_date, reverse=True)
    
    # Rollback and Compatibility
    
    def create_model_backup(self, model_id: str, backup_path: Optional[str] = None) -> str:
        """
        Create backup of model file for rollback purposes.
        
        Args:
            model_id: Model identifier
            backup_path: Custom backup path (optional)
            
        Returns:
            Path to backup file
            
        Raises:
            ValueError: If model not found
            FileNotFoundError: If model file doesn't exist
        """
        record = self.get_model(model_id)
        if not record:
            raise ValueError(f"Model {model_id} not found")
        
        if not os.path.exists(record.file_path):
            raise FileNotFoundError(f"Model file not found: {record.file_path}")
        
        # Generate backup path
        if not backup_path:
            backup_dir = self.registry_path / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{model_id}_{timestamp}.backup"
            backup_path = backup_dir / backup_filename
        
        # Copy model file
        shutil.copy2(record.file_path, backup_path)
        
        logger.info(f"Created backup for model {model_id}: {backup_path}")
        return str(backup_path)
    
    def rollback_model(self, model_id: str, backup_path: str) -> bool:
        """
        Rollback model to previous version from backup.
        
        Args:
            model_id: Model identifier
            backup_path: Path to backup file
            
        Returns:
            True if rollback successful
        """
        record = self.get_model(model_id)
        if not record:
            logger.error(f"Model {model_id} not found for rollback")
            return False
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create current backup before rollback
            current_backup = self.create_model_backup(model_id)
            logger.info(f"Created current backup before rollback: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_path, record.file_path)
            
            # Update record
            record.last_updated = datetime.now()
            record.status = ModelStatus.ACTIVE  # Assume rollback makes it active
            
            self._save_registry_data()
            
            logger.info(f"Successfully rolled back model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback of model {model_id}: {e}")
            return False
    
    def validate_model_compatibility(self, 
                                   model_a_id: str, 
                                   model_b_id: str) -> Tuple[bool, List[str]]:
        """
        Validate compatibility between two models.
        
        Args:
            model_a_id: First model ID
            model_b_id: Second model ID
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        record_a = self.get_model(model_a_id)
        record_b = self.get_model(model_b_id)
        
        issues = []
        
        if not record_a:
            issues.append(f"Model {model_a_id} not found")
        if not record_b:
            issues.append(f"Model {model_b_id} not found")
        
        if issues:
            return False, issues
        
        # Check model type compatibility
        if record_a.model_type != record_b.model_type:
            issues.append(f"Model type mismatch: {record_a.model_type} vs {record_b.model_type}")
        
        # Check language compatibility for classifiers
        if record_a.model_type == "classifier":
            if record_a.language != record_b.language:
                issues.append(f"Language mismatch: {record_a.language} vs {record_b.language}")
        
        # Check file existence
        if not os.path.exists(record_a.file_path):
            issues.append(f"Model A file not found: {record_a.file_path}")
        if not os.path.exists(record_b.file_path):
            issues.append(f"Model B file not found: {record_b.file_path}")
        
        # Check additional metadata compatibility
        if record_a.additional_metadata.get('embedding_dim') != record_b.additional_metadata.get('embedding_dim'):
            issues.append("Embedding dimension mismatch")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues
    
    # Utility Methods
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return f"sha256:{hash_sha256.hexdigest()}"
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        models_by_type = {}
        models_by_status = {}
        models_by_language = {}
        
        for record in self._models_cache.values():
            # By type
            models_by_type[record.model_type] = models_by_type.get(record.model_type, 0) + 1
            
            # By status
            status_str = record.status.value
            models_by_status[status_str] = models_by_status.get(status_str, 0) + 1
            
            # By language (for classifiers)
            if record.language:
                models_by_language[record.language] = models_by_language.get(record.language, 0) + 1
        
        return {
            "total_models": len(self._models_cache),
            "models_by_type": models_by_type,
            "models_by_status": models_by_status,
            "models_by_language": models_by_language,
            "total_thresholds": len(self._thresholds_cache),
            "total_ab_tests": len(self._ab_tests_cache),
            "running_ab_tests": len([t for t in self._ab_tests_cache.values() 
                                   if t.status == ABTestStatus.RUNNING])
        }
    
    def cleanup_old_backups(self, days_to_keep: int = 30):
        """
        Clean up old backup files.
        
        Args:
            days_to_keep: Number of days to keep backups
        """
        backup_dir = self.registry_path / "backups"
        if not backup_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        for backup_file in backup_dir.glob("*.backup"):
            try:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    backup_file.unlink()
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting backup {backup_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old backup files")