"""
Configuration Management Module for AI Voice Detection System

This module provides comprehensive configuration management for thresholds,
model paths, and deployment settings with validation and persistence.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from src.config import get_settings

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class DeploymentEnvironment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ThresholdConfiguration:
    """Configuration for language-specific thresholds."""
    language: str
    threshold: float
    precision: float
    recall: float
    f1_score: float
    optimization_date: datetime
    model_version: str
    is_optimized: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['optimization_date'] = self.optimization_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfiguration':
        """Create from dictionary."""
        if isinstance(data['optimization_date'], str):
            data['optimization_date'] = datetime.fromisoformat(data['optimization_date'])
        return cls(**data)


@dataclass
class ModelPathConfiguration:
    """Configuration for model file paths."""
    foundation_models: Dict[str, str]
    classifiers: Dict[str, str]
    explanation_models: Dict[str, str]
    preprocessing_models: Dict[str, str]
    base_path: str
    
    def validate_paths(self) -> List[str]:
        """Validate that all configured paths exist."""
        missing_paths = []
        
        # Check foundation models
        for model_name, path in self.foundation_models.items():
            full_path = Path(self.base_path) / path
            if not full_path.exists():
                missing_paths.append(f"Foundation model {model_name}: {full_path}")
        
        # Check classifiers
        for language, path in self.classifiers.items():
            full_path = Path(self.base_path) / path
            if not full_path.exists():
                missing_paths.append(f"Classifier {language}: {full_path}")
        
        # Check explanation models
        for model_name, path in self.explanation_models.items():
            full_path = Path(self.base_path) / path
            if not full_path.exists():
                missing_paths.append(f"Explanation model {model_name}: {full_path}")
        
        return missing_paths


@dataclass
class DeploymentConfiguration:
    """Configuration for deployment settings."""
    environment: DeploymentEnvironment
    api_host: str
    api_port: int
    workers: int
    log_level: str
    enable_monitoring: bool
    enable_validation: bool
    enable_fallback: bool
    max_request_size: int
    request_timeout: int
    cors_origins: List[str]
    ssl_enabled: bool
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate deployment configuration."""
        errors = []
        
        if self.api_port < 1 or self.api_port > 65535:
            errors.append(f"Invalid port number: {self.api_port}")
        
        if self.workers < 1:
            errors.append(f"Workers must be at least 1: {self.workers}")
        
        if self.ssl_enabled:
            if not self.ssl_cert_path or not self.ssl_key_path:
                errors.append("SSL enabled but certificate or key path not provided")
            elif not Path(self.ssl_cert_path).exists():
                errors.append(f"SSL certificate not found: {self.ssl_cert_path}")
            elif not Path(self.ssl_key_path).exists():
                errors.append(f"SSL key not found: {self.ssl_key_path}")
        
        return errors


class ConfigurationManager:
    """
    Comprehensive configuration manager for the AI voice detection system.
    
    Manages thresholds, model paths, deployment settings with validation
    and persistence capabilities.
    """
    
    def __init__(self, config_dir: str = ".kiro/config"):
        """
        Initialize ConfigurationManager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file paths
        self.thresholds_file = self.config_dir / "thresholds.json"
        self.model_paths_file = self.config_dir / "model_paths.json"
        self.deployment_file = self.config_dir / "deployment.json"
        self.system_config_file = self.config_dir / "system_config.json"
        
        # Load configurations
        self.thresholds: Dict[str, ThresholdConfiguration] = {}
        self.model_paths: Optional[ModelPathConfiguration] = None
        self.deployment: Optional[DeploymentConfiguration] = None
        self.system_config: Dict[str, Any] = {}
        
        self._load_all_configurations()
        
        logger.info(f"ConfigurationManager initialized with config directory: {config_dir}")
    
    def _load_all_configurations(self):
        """Load all configuration files."""
        try:
            self._load_thresholds()
            self._load_model_paths()
            self._load_deployment_config()
            self._load_system_config()
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def _load_thresholds(self):
        """Load threshold configurations."""
        try:
            if self.thresholds_file.exists():
                with open(self.thresholds_file, 'r') as f:
                    data = json.load(f)
                    self.thresholds = {
                        lang: ThresholdConfiguration.from_dict(config)
                        for lang, config in data.items()
                    }
                logger.info(f"Loaded {len(self.thresholds)} threshold configurations")
        except Exception as e:
            logger.error(f"Error loading thresholds: {e}")
            self.thresholds = {}
    
    def _load_model_paths(self):
        """Load model path configurations."""
        try:
            if self.model_paths_file.exists():
                with open(self.model_paths_file, 'r') as f:
                    data = json.load(f)
                    self.model_paths = ModelPathConfiguration(**data)
                logger.info("Loaded model path configurations")
            else:
                # Create default model paths configuration
                self._create_default_model_paths()
        except Exception as e:
            logger.error(f"Error loading model paths: {e}")
            self._create_default_model_paths()
    
    def _load_deployment_config(self):
        """Load deployment configuration."""
        try:
            if self.deployment_file.exists():
                with open(self.deployment_file, 'r') as f:
                    data = json.load(f)
                    # Convert environment string to enum
                    if 'environment' in data:
                        data['environment'] = DeploymentEnvironment(data['environment'])
                    self.deployment = DeploymentConfiguration(**data)
                logger.info("Loaded deployment configuration")
            else:
                # Create default deployment configuration
                self._create_default_deployment_config()
        except Exception as e:
            logger.error(f"Error loading deployment config: {e}")
            self._create_default_deployment_config()
    
    def _load_system_config(self):
        """Load system configuration."""
        try:
            if self.system_config_file.exists():
                with open(self.system_config_file, 'r') as f:
                    self.system_config = json.load(f)
                logger.info("Loaded system configuration")
            else:
                # Create default system configuration
                self._create_default_system_config()
        except Exception as e:
            logger.error(f"Error loading system config: {e}")
            self._create_default_system_config()
    
    def _create_default_model_paths(self):
        """Create default model paths configuration."""
        settings = get_settings()
        
        self.model_paths = ModelPathConfiguration(
            foundation_models={
                "xls-r-300m": "foundation/xls-r-300m",
                "hubert-base": "foundation/hubert-base",
                "wav2vec2-base": "foundation/wav2vec2-base"
            },
            classifiers={
                "english": "classifiers/english_classifier.h5",
                "tamil": "classifiers/tamil_classifier.h5",
                "hindi": "classifiers/hindi_classifier.h5",
                "malayalam": "classifiers/malayalam_classifier.h5",
                "telugu": "classifiers/telugu_classifier.h5"
            },
            explanation_models={
                "distilgpt2": "explanation/distilgpt2",
                "flan-t5-small": "explanation/flan-t5-small"
            },
            preprocessing_models={
                "feature_scalers": "preprocessing/feature_scalers"
            },
            base_path=settings.ml.models_base_path
        )
        
        self._save_model_paths()
        logger.info("Created default model paths configuration")
    
    def _create_default_deployment_config(self):
        """Create default deployment configuration."""
        settings = get_settings()
        
        self.deployment = DeploymentConfiguration(
            environment=DeploymentEnvironment(settings.environment.value),
            api_host=settings.api.host,
            api_port=settings.api.port,
            workers=settings.api.workers,
            log_level=settings.logging.log_level.value,
            enable_monitoring=True,
            enable_validation=True,
            enable_fallback=True,
            max_request_size=settings.security.max_request_size,
            request_timeout=settings.security.request_timeout,
            cors_origins=settings.api.cors_origins,
            ssl_enabled=settings.security.ssl_enabled,
            ssl_cert_path=settings.security.ssl_cert_file,
            ssl_key_path=settings.security.ssl_key_file
        )
        
        self._save_deployment_config()
        logger.info("Created default deployment configuration")
    
    def _create_default_system_config(self):
        """Create default system configuration."""
        settings = get_settings()
        
        self.system_config = {
            "monitoring": {
                "window_size": 1000,
                "degradation_threshold": 0.05,
                "alert_thresholds": {
                    "warning": 0.03,
                    "critical": 0.08
                },
                "retention_days": 30
            },
            "validation": {
                "startup_validation": True,
                "periodic_validation": True,
                "validation_interval_hours": 24,
                "min_confidence_threshold": 0.6
            },
            "fallback": {
                "confidence_threshold": 0.7,
                "memory_limit_mb": 2048,
                "max_processing_time_seconds": 30,
                "enable_traditional_features": True
            },
            "optimization": {
                "auto_threshold_optimization": True,
                "optimization_interval_hours": 168,  # Weekly
                "min_samples_for_optimization": 100
            },
            "registry": {
                "auto_registration": True,
                "backup_retention_days": 30,
                "enable_ab_testing": True
            }
        }
        
        self._save_system_config()
        logger.info("Created default system configuration")
    
    # Threshold Management
    
    def set_threshold(self, language: str, threshold: float, precision: float,
                     recall: float, f1_score: float, model_version: str = "1.0.0") -> None:
        """
        Set optimized threshold for a language.
        
        Args:
            language: Target language
            threshold: Optimized threshold value
            precision: Precision achieved with this threshold
            recall: Recall achieved with this threshold
            f1_score: F1 score achieved with this threshold
            model_version: Model version
        """
        config = ThresholdConfiguration(
            language=language,
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            optimization_date=datetime.now(),
            model_version=model_version,
            is_optimized=True
        )
        
        self.thresholds[language] = config
        self._save_thresholds()
        
        logger.info(f"Set threshold for {language}: {threshold:.4f} (F1: {f1_score:.3f})")
    
    def get_threshold(self, language: str) -> Optional[ThresholdConfiguration]:
        """Get threshold configuration for a language."""
        return self.thresholds.get(language)
    
    def get_all_thresholds(self) -> Dict[str, ThresholdConfiguration]:
        """Get all threshold configurations."""
        return self.thresholds.copy()
    
    def reset_threshold(self, language: str) -> None:
        """Reset threshold to default for a language."""
        if language in self.thresholds:
            del self.thresholds[language]
            self._save_thresholds()
            logger.info(f"Reset threshold for {language} to default")
    
    # Model Path Management
    
    def update_model_path(self, model_type: str, model_name: str, path: str) -> None:
        """
        Update model path configuration.
        
        Args:
            model_type: Type of model ("foundation", "classifier", "explanation", "preprocessing")
            model_name: Name/identifier of the model
            path: Relative path from base path
        """
        if not self.model_paths:
            self._create_default_model_paths()
        
        if model_type == "foundation":
            self.model_paths.foundation_models[model_name] = path
        elif model_type == "classifier":
            self.model_paths.classifiers[model_name] = path
        elif model_type == "explanation":
            self.model_paths.explanation_models[model_name] = path
        elif model_type == "preprocessing":
            self.model_paths.preprocessing_models[model_name] = path
        else:
            raise ConfigurationError(f"Unknown model type: {model_type}")
        
        self._save_model_paths()
        logger.info(f"Updated {model_type} model path for {model_name}: {path}")
    
    def get_model_path(self, model_type: str, model_name: str) -> Optional[str]:
        """Get model path for a specific model."""
        if not self.model_paths:
            return None
        
        if model_type == "foundation":
            return self.model_paths.foundation_models.get(model_name)
        elif model_type == "classifier":
            return self.model_paths.classifiers.get(model_name)
        elif model_type == "explanation":
            return self.model_paths.explanation_models.get(model_name)
        elif model_type == "preprocessing":
            return self.model_paths.preprocessing_models.get(model_name)
        else:
            return None
    
    def validate_model_paths(self) -> List[str]:
        """Validate all configured model paths."""
        if not self.model_paths:
            return ["Model paths not configured"]
        
        return self.model_paths.validate_paths()
    
    # Deployment Configuration Management
    
    def update_deployment_config(self, **kwargs) -> None:
        """Update deployment configuration."""
        if not self.deployment:
            self._create_default_deployment_config()
        
        for key, value in kwargs.items():
            if hasattr(self.deployment, key):
                if key == 'environment' and isinstance(value, str):
                    value = DeploymentEnvironment(value)
                setattr(self.deployment, key, value)
            else:
                logger.warning(f"Unknown deployment config key: {key}")
        
        self._save_deployment_config()
        logger.info(f"Updated deployment configuration: {list(kwargs.keys())}")
    
    def validate_deployment_config(self) -> List[str]:
        """Validate deployment configuration."""
        if not self.deployment:
            return ["Deployment configuration not loaded"]
        
        return self.deployment.validate()
    
    # System Configuration Management
    
    def update_system_config(self, section: str, key: str, value: Any) -> None:
        """Update system configuration value."""
        if section not in self.system_config:
            self.system_config[section] = {}
        
        self.system_config[section][key] = value
        self._save_system_config()
        
        logger.info(f"Updated system config {section}.{key}: {value}")
    
    def get_system_config(self, section: str, key: str, default: Any = None) -> Any:
        """Get system configuration value."""
        return self.system_config.get(section, {}).get(key, default)
    
    # Persistence Methods
    
    def _save_thresholds(self):
        """Save threshold configurations to file."""
        try:
            data = {
                lang: config.to_dict()
                for lang, config in self.thresholds.items()
            }
            with open(self.thresholds_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving thresholds: {e}")
            raise ConfigurationError(f"Failed to save thresholds: {e}")
    
    def _save_model_paths(self):
        """Save model paths configuration to file."""
        try:
            if self.model_paths:
                with open(self.model_paths_file, 'w') as f:
                    json.dump(asdict(self.model_paths), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model paths: {e}")
            raise ConfigurationError(f"Failed to save model paths: {e}")
    
    def _save_deployment_config(self):
        """Save deployment configuration to file."""
        try:
            if self.deployment:
                data = asdict(self.deployment)
                # Convert enum to string
                data['environment'] = self.deployment.environment.value
                with open(self.deployment_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving deployment config: {e}")
            raise ConfigurationError(f"Failed to save deployment config: {e}")
    
    def _save_system_config(self):
        """Save system configuration to file."""
        try:
            with open(self.system_config_file, 'w') as f:
                json.dump(self.system_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving system config: {e}")
            raise ConfigurationError(f"Failed to save system config: {e}")
    
    # Utility Methods
    
    def export_configuration(self, output_file: str) -> None:
        """
        Export all configurations to a single file.
        
        Args:
            output_file: Path to output file
        """
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "thresholds": {
                    lang: config.to_dict()
                    for lang, config in self.thresholds.items()
                },
                "model_paths": asdict(self.model_paths) if self.model_paths else None,
                "deployment": asdict(self.deployment) if self.deployment else None,
                "system_config": self.system_config
            }
            
            # Convert deployment environment enum to string
            if export_data["deployment"] and "environment" in export_data["deployment"]:
                export_data["deployment"]["environment"] = self.deployment.environment.value
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported configuration to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            raise ConfigurationError(f"Failed to export configuration: {e}")
    
    def import_configuration(self, input_file: str, overwrite: bool = False) -> None:
        """
        Import configurations from a file.
        
        Args:
            input_file: Path to input file
            overwrite: Whether to overwrite existing configurations
        """
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            # Import thresholds
            if "thresholds" in import_data:
                for lang, config_data in import_data["thresholds"].items():
                    if overwrite or lang not in self.thresholds:
                        self.thresholds[lang] = ThresholdConfiguration.from_dict(config_data)
                self._save_thresholds()
            
            # Import model paths
            if "model_paths" in import_data and import_data["model_paths"]:
                if overwrite or not self.model_paths:
                    self.model_paths = ModelPathConfiguration(**import_data["model_paths"])
                    self._save_model_paths()
            
            # Import deployment config
            if "deployment" in import_data and import_data["deployment"]:
                if overwrite or not self.deployment:
                    deployment_data = import_data["deployment"]
                    if "environment" in deployment_data:
                        deployment_data["environment"] = DeploymentEnvironment(deployment_data["environment"])
                    self.deployment = DeploymentConfiguration(**deployment_data)
                    self._save_deployment_config()
            
            # Import system config
            if "system_config" in import_data:
                if overwrite:
                    self.system_config = import_data["system_config"]
                else:
                    # Merge configurations
                    for section, config in import_data["system_config"].items():
                        if section not in self.system_config:
                            self.system_config[section] = config
                        else:
                            self.system_config[section].update(config)
                self._save_system_config()
            
            logger.info(f"Imported configuration from {input_file}")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            raise ConfigurationError(f"Failed to import configuration: {e}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations."""
        return {
            "thresholds": {
                "count": len(self.thresholds),
                "languages": list(self.thresholds.keys()),
                "optimized_count": sum(1 for config in self.thresholds.values() if config.is_optimized)
            },
            "model_paths": {
                "configured": self.model_paths is not None,
                "foundation_models": len(self.model_paths.foundation_models) if self.model_paths else 0,
                "classifiers": len(self.model_paths.classifiers) if self.model_paths else 0,
                "missing_paths": len(self.validate_model_paths())
            },
            "deployment": {
                "configured": self.deployment is not None,
                "environment": self.deployment.environment.value if self.deployment else None,
                "validation_errors": len(self.validate_deployment_config())
            },
            "system_config": {
                "sections": list(self.system_config.keys()),
                "monitoring_enabled": self.get_system_config("monitoring", "window_size") is not None,
                "validation_enabled": self.get_system_config("validation", "startup_validation", False),
                "fallback_enabled": self.get_system_config("fallback", "confidence_threshold") is not None
            }
        }