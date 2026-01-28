"""
Model compatibility validation for AI voice detection system.

This module implements dimensional compatibility checks for model switches,
feature consistency validation between models, and runtime error prevention
for mismatched architectures.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, List, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.exceptions import ValidationError, ModelLoadingError, ConfigurationError
from src.models.schemas import ValidationResult

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Levels of model compatibility."""
    FULLY_COMPATIBLE = "fully_compatible"
    COMPATIBLE_WITH_ADAPTATION = "compatible_with_adaptation"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue between models."""
    issue_type: str
    severity: ValidationSeverity
    description: str
    affected_component: str
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelCompatibilityResult:
    """Result of model compatibility validation."""
    compatibility_level: CompatibilityLevel
    is_compatible: bool
    issues: List[CompatibilityIssue]
    adaptation_required: bool
    adaptation_suggestions: List[str]
    validation_metadata: Dict[str, Any]
    
    def __post_init__(self):
        # Determine if compatible based on issues
        critical_issues = [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
        error_issues = [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
        
        if critical_issues:
            self.is_compatible = False
            self.compatibility_level = CompatibilityLevel.INCOMPATIBLE
        elif error_issues:
            self.is_compatible = False
            self.compatibility_level = CompatibilityLevel.INCOMPATIBLE
        elif self.adaptation_required:
            self.is_compatible = True
            self.compatibility_level = CompatibilityLevel.COMPATIBLE_WITH_ADAPTATION
        else:
            self.is_compatible = True
            self.compatibility_level = CompatibilityLevel.FULLY_COMPATIBLE


@dataclass
class ModelArchitecture:
    """Represents model architecture information."""
    model_type: str  # "foundation" or "classifier"
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    embedding_dimension: Optional[int] = None
    model_format: str = "tensorflow"  # "tensorflow", "pytorch", "onnx"
    version: str = "unknown"
    language: Optional[str] = None
    framework_version: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelCompatibilityValidator:
    """
    Validates compatibility between models to prevent runtime errors.
    
    Implements dimensional compatibility checks for model switches,
    feature consistency validation between models, and runtime error
    prevention for mismatched architectures.
    
    Requirements addressed:
    - 4.4: Model compatibility validation for switches
    - Dimensional compatibility checks
    - Feature consistency validation
    - Runtime error prevention for mismatched architectures
    """
    
    def __init__(self, settings=None):
        """
        Initialize the ModelCompatibilityValidator.
        
        Args:
            settings: Application settings object containing configuration
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.model_base_path = Path(settings.ml.models_base_path)
        
        # Cache for model architectures
        self.architecture_cache: Dict[str, ModelArchitecture] = {}
        
        # Compatibility rules
        self.compatibility_rules = self._initialize_compatibility_rules()
        
        logger.info("ModelCompatibilityValidator initialized")
    
    def validate_model_switch_compatibility(self, current_model_path: str, 
                                          new_model_path: str,
                                          model_type: str = "classifier",
                                          language: Optional[str] = None) -> ModelCompatibilityResult:
        """
        Validate compatibility when switching between models.
        
        Args:
            current_model_path: Path to current model
            new_model_path: Path to new model
            model_type: Type of model ("foundation" or "classifier")
            language: Target language (for classifiers)
            
        Returns:
            ModelCompatibilityResult with validation results
        """
        try:
            logger.info(f"Validating model switch compatibility: {current_model_path} -> {new_model_path}")
            
            # Get architectures for both models
            current_arch = self._get_model_architecture(current_model_path, model_type, language)
            new_arch = self._get_model_architecture(new_model_path, model_type, language)
            
            # Perform compatibility checks
            issues = []
            adaptation_suggestions = []
            
            # Check dimensional compatibility
            dimension_issues, dimension_adaptations = self._check_dimensional_compatibility(
                current_arch, new_arch
            )
            issues.extend(dimension_issues)
            adaptation_suggestions.extend(dimension_adaptations)
            
            # Check feature consistency
            feature_issues, feature_adaptations = self._check_feature_consistency(
                current_arch, new_arch
            )
            issues.extend(feature_issues)
            adaptation_suggestions.extend(feature_adaptations)
            
            # Check framework compatibility
            framework_issues, framework_adaptations = self._check_framework_compatibility(
                current_arch, new_arch
            )
            issues.extend(framework_issues)
            adaptation_suggestions.extend(framework_adaptations)
            
            # Check version compatibility
            version_issues, version_adaptations = self._check_version_compatibility(
                current_arch, new_arch
            )
            issues.extend(version_issues)
            adaptation_suggestions.extend(version_adaptations)
            
            # Determine if adaptation is required
            adaptation_required = len(adaptation_suggestions) > 0
            
            validation_metadata = {
                "current_model": current_model_path,
                "new_model": new_model_path,
                "model_type": model_type,
                "language": language,
                "current_architecture": self._architecture_to_dict(current_arch),
                "new_architecture": self._architecture_to_dict(new_arch),
                "total_issues": len(issues),
                "critical_issues": len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]),
                "error_issues": len([i for i in issues if i.severity == ValidationSeverity.ERROR]),
                "warning_issues": len([i for i in issues if i.severity == ValidationSeverity.WARNING])
            }
            
            result = ModelCompatibilityResult(
                compatibility_level=CompatibilityLevel.UNKNOWN,  # Will be set in __post_init__
                is_compatible=False,  # Will be set in __post_init__
                issues=issues,
                adaptation_required=adaptation_required,
                adaptation_suggestions=adaptation_suggestions,
                validation_metadata=validation_metadata
            )
            
            logger.info(
                f"Model switch compatibility validation completed: "
                f"{result.compatibility_level.value} with {len(issues)} issues"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Model switch compatibility validation failed: {str(e)}")
            
            # Return incompatible result with error
            return ModelCompatibilityResult(
                compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                is_compatible=False,
                issues=[
                    CompatibilityIssue(
                        issue_type="validation_error",
                        severity=ValidationSeverity.CRITICAL,
                        description=f"Compatibility validation failed: {str(e)}",
                        affected_component="compatibility_validator"
                    )
                ],
                adaptation_required=False,
                adaptation_suggestions=[],
                validation_metadata={"error": str(e)}
            )
    
    def validate_foundation_classifier_compatibility(self, foundation_model_path: str,
                                                   classifier_path: str,
                                                   language: str) -> ModelCompatibilityResult:
        """
        Validate compatibility between foundation model and classifier.
        
        Args:
            foundation_model_path: Path to foundation model
            classifier_path: Path to classifier model
            language: Target language
            
        Returns:
            ModelCompatibilityResult with validation results
        """
        try:
            logger.info(f"Validating foundation-classifier compatibility for {language}")
            
            # Get architectures
            foundation_arch = self._get_model_architecture(foundation_model_path, "foundation")
            classifier_arch = self._get_model_architecture(classifier_path, "classifier", language)
            
            issues = []
            adaptation_suggestions = []
            
            # Check if foundation model output matches classifier input
            if foundation_arch.embedding_dimension and classifier_arch.input_shape:
                expected_input_dim = classifier_arch.input_shape[0] if classifier_arch.input_shape else None
                
                if expected_input_dim and foundation_arch.embedding_dimension != expected_input_dim:
                    issues.append(
                        CompatibilityIssue(
                            issue_type="dimension_mismatch",
                            severity=ValidationSeverity.CRITICAL,
                            description=(
                                f"Foundation model embedding dimension ({foundation_arch.embedding_dimension}) "
                                f"does not match classifier input dimension ({expected_input_dim})"
                            ),
                            affected_component="feature_pipeline",
                            suggested_fix="Use dimension adaptation layer or retrain classifier",
                            metadata={
                                "foundation_embedding_dim": foundation_arch.embedding_dimension,
                                "classifier_input_dim": expected_input_dim
                            }
                        )
                    )
                    
                    adaptation_suggestions.append(
                        f"Add dimension adaptation layer from {foundation_arch.embedding_dimension} "
                        f"to {expected_input_dim} dimensions"
                    )
            
            # Check framework compatibility
            if foundation_arch.model_format != classifier_arch.model_format:
                issues.append(
                    CompatibilityIssue(
                        issue_type="framework_mismatch",
                        severity=ValidationSeverity.ERROR,
                        description=(
                            f"Foundation model format ({foundation_arch.model_format}) "
                            f"differs from classifier format ({classifier_arch.model_format})"
                        ),
                        affected_component="model_loading",
                        suggested_fix="Convert models to same format or use framework bridges"
                    )
                )
                
                adaptation_suggestions.append(
                    f"Convert models to common format or implement {foundation_arch.model_format} "
                    f"to {classifier_arch.model_format} bridge"
                )
            
            # Check language compatibility
            if classifier_arch.language and classifier_arch.language != language:
                issues.append(
                    CompatibilityIssue(
                        issue_type="language_mismatch",
                        severity=ValidationSeverity.ERROR,
                        description=(
                            f"Classifier trained for {classifier_arch.language} "
                            f"but requested for {language}"
                        ),
                        affected_component="classifier",
                        suggested_fix=f"Use classifier trained for {language}"
                    )
                )
            
            validation_metadata = {
                "foundation_model": foundation_model_path,
                "classifier_model": classifier_path,
                "language": language,
                "foundation_architecture": self._architecture_to_dict(foundation_arch),
                "classifier_architecture": self._architecture_to_dict(classifier_arch)
            }
            
            result = ModelCompatibilityResult(
                compatibility_level=CompatibilityLevel.UNKNOWN,  # Will be set in __post_init__
                is_compatible=False,  # Will be set in __post_init__
                issues=issues,
                adaptation_required=len(adaptation_suggestions) > 0,
                adaptation_suggestions=adaptation_suggestions,
                validation_metadata=validation_metadata
            )
            
            logger.info(
                f"Foundation-classifier compatibility validation completed: "
                f"{result.compatibility_level.value}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Foundation-classifier compatibility validation failed: {str(e)}")
            
            return ModelCompatibilityResult(
                compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                is_compatible=False,
                issues=[
                    CompatibilityIssue(
                        issue_type="validation_error",
                        severity=ValidationSeverity.CRITICAL,
                        description=f"Compatibility validation failed: {str(e)}",
                        affected_component="compatibility_validator"
                    )
                ],
                adaptation_required=False,
                adaptation_suggestions=[],
                validation_metadata={"error": str(e)}
            )
    
    def validate_runtime_compatibility(self, model_configs: Dict[str, Any]) -> ModelCompatibilityResult:
        """
        Validate runtime compatibility of current model configuration.
        
        Args:
            model_configs: Dictionary of model configurations
            
        Returns:
            ModelCompatibilityResult with validation results
        """
        try:
            logger.info("Validating runtime model compatibility")
            
            issues = []
            adaptation_suggestions = []
            
            # Check each model configuration
            for config_name, config in model_configs.items():
                try:
                    # Validate individual model configuration
                    config_issues = self._validate_model_config(config_name, config)
                    issues.extend(config_issues)
                    
                except Exception as config_error:
                    issues.append(
                        CompatibilityIssue(
                            issue_type="config_validation_error",
                            severity=ValidationSeverity.ERROR,
                            description=f"Failed to validate {config_name}: {str(config_error)}",
                            affected_component=config_name
                        )
                    )
            
            # Check cross-model compatibility
            cross_model_issues = self._validate_cross_model_compatibility(model_configs)
            issues.extend(cross_model_issues)
            
            validation_metadata = {
                "total_models": len(model_configs),
                "validated_models": list(model_configs.keys()),
                "runtime_check": True
            }
            
            result = ModelCompatibilityResult(
                compatibility_level=CompatibilityLevel.UNKNOWN,  # Will be set in __post_init__
                is_compatible=False,  # Will be set in __post_init__
                issues=issues,
                adaptation_required=len(adaptation_suggestions) > 0,
                adaptation_suggestions=adaptation_suggestions,
                validation_metadata=validation_metadata
            )
            
            logger.info(f"Runtime compatibility validation completed with {len(issues)} issues")
            
            return result
            
        except Exception as e:
            logger.error(f"Runtime compatibility validation failed: {str(e)}")
            
            return ModelCompatibilityResult(
                compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                is_compatible=False,
                issues=[
                    CompatibilityIssue(
                        issue_type="runtime_validation_error",
                        severity=ValidationSeverity.CRITICAL,
                        description=f"Runtime validation failed: {str(e)}",
                        affected_component="runtime_validator"
                    )
                ],
                adaptation_required=False,
                adaptation_suggestions=[],
                validation_metadata={"error": str(e)}
            )
    
    def _get_model_architecture(self, model_path: str, model_type: str, 
                              language: Optional[str] = None) -> ModelArchitecture:
        """
        Get architecture information for a model.
        
        Args:
            model_path: Path to model file
            model_type: Type of model ("foundation" or "classifier")
            language: Target language (for classifiers)
            
        Returns:
            ModelArchitecture with model information
        """
        try:
            # Check cache first
            cache_key = f"{model_path}_{model_type}_{language or 'none'}"
            if cache_key in self.architecture_cache:
                return self.architecture_cache[cache_key]
            
            # Load model to inspect architecture
            model_path_obj = Path(model_path)
            
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Try to load model and extract architecture
            try:
                if model_path.endswith('.h5') or model_path.endswith('.keras'):
                    # TensorFlow/Keras model
                    model = tf.keras.models.load_model(model_path)
                    
                    input_shape = model.input_shape[1:] if model.input_shape else (None,)
                    output_shape = model.output_shape[1:] if model.output_shape else (None,)
                    
                    # For foundation models, embedding dimension is output dimension
                    embedding_dim = None
                    if model_type == "foundation" and output_shape:
                        embedding_dim = output_shape[0] if len(output_shape) == 1 else output_shape[-1]
                    
                    architecture = ModelArchitecture(
                        model_type=model_type,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        embedding_dimension=embedding_dim,
                        model_format="tensorflow",
                        language=language,
                        metadata={
                            "layers": len(model.layers),
                            "trainable_params": model.count_params(),
                            "model_size_mb": model_path_obj.stat().st_size / (1024 * 1024)
                        }
                    )
                    
                    # Clean up model to free memory
                    del model
                    
                else:
                    # Unknown format - create minimal architecture
                    architecture = ModelArchitecture(
                        model_type=model_type,
                        input_shape=(None,),
                        output_shape=(None,),
                        model_format="unknown",
                        language=language,
                        metadata={
                            "model_size_mb": model_path_obj.stat().st_size / (1024 * 1024)
                        }
                    )
                
                # Cache the architecture
                self.architecture_cache[cache_key] = architecture
                
                return architecture
                
            except Exception as load_error:
                logger.warning(f"Failed to load model for architecture inspection: {load_error}")
                
                # Create minimal architecture based on file inspection
                architecture = ModelArchitecture(
                    model_type=model_type,
                    input_shape=(None,),
                    output_shape=(None,),
                    model_format="unknown",
                    language=language,
                    metadata={
                        "load_error": str(load_error),
                        "model_size_mb": model_path_obj.stat().st_size / (1024 * 1024)
                    }
                )
                
                return architecture
                
        except Exception as e:
            logger.error(f"Failed to get model architecture for {model_path}: {str(e)}")
            
            # Return minimal architecture with error
            return ModelArchitecture(
                model_type=model_type,
                input_shape=(None,),
                output_shape=(None,),
                model_format="error",
                language=language,
                metadata={"error": str(e)}
            )
    
    def _check_dimensional_compatibility(self, current_arch: ModelArchitecture,
                                       new_arch: ModelArchitecture) -> Tuple[List[CompatibilityIssue], List[str]]:
        """Check dimensional compatibility between models."""
        issues = []
        adaptations = []
        
        # Check input shape compatibility
        if current_arch.input_shape != new_arch.input_shape:
            if current_arch.input_shape != (None,) and new_arch.input_shape != (None,):
                issues.append(
                    CompatibilityIssue(
                        issue_type="input_shape_mismatch",
                        severity=ValidationSeverity.ERROR,
                        description=(
                            f"Input shape mismatch: {current_arch.input_shape} -> {new_arch.input_shape}"
                        ),
                        affected_component="model_input",
                        suggested_fix="Add input reshaping layer or retrain model"
                    )
                )
                
                adaptations.append(f"Add input reshaping from {current_arch.input_shape} to {new_arch.input_shape}")
        
        # Check output shape compatibility
        if current_arch.output_shape != new_arch.output_shape:
            if current_arch.output_shape != (None,) and new_arch.output_shape != (None,):
                issues.append(
                    CompatibilityIssue(
                        issue_type="output_shape_mismatch",
                        severity=ValidationSeverity.WARNING,
                        description=(
                            f"Output shape mismatch: {current_arch.output_shape} -> {new_arch.output_shape}"
                        ),
                        affected_component="model_output",
                        suggested_fix="Update downstream components to handle new output shape"
                    )
                )
                
                adaptations.append(f"Update output handling from {current_arch.output_shape} to {new_arch.output_shape}")
        
        # Check embedding dimension compatibility (for foundation models)
        if (current_arch.embedding_dimension and new_arch.embedding_dimension and
            current_arch.embedding_dimension != new_arch.embedding_dimension):
            
            issues.append(
                CompatibilityIssue(
                    issue_type="embedding_dimension_mismatch",
                    severity=ValidationSeverity.CRITICAL,
                    description=(
                        f"Embedding dimension mismatch: {current_arch.embedding_dimension} -> "
                        f"{new_arch.embedding_dimension}"
                    ),
                    affected_component="embedding_pipeline",
                    suggested_fix="Add dimension adaptation layer or retrain downstream models"
                )
            )
            
            adaptations.append(
                f"Add embedding dimension adaptation from {current_arch.embedding_dimension} "
                f"to {new_arch.embedding_dimension}"
            )
        
        return issues, adaptations
    
    def _check_feature_consistency(self, current_arch: ModelArchitecture,
                                 new_arch: ModelArchitecture) -> Tuple[List[CompatibilityIssue], List[str]]:
        """Check feature consistency between models."""
        issues = []
        adaptations = []
        
        # Check model type consistency
        if current_arch.model_type != new_arch.model_type:
            issues.append(
                CompatibilityIssue(
                    issue_type="model_type_mismatch",
                    severity=ValidationSeverity.CRITICAL,
                    description=(
                        f"Model type mismatch: {current_arch.model_type} -> {new_arch.model_type}"
                    ),
                    affected_component="model_pipeline",
                    suggested_fix="Use models of the same type"
                )
            )
        
        # Check language consistency (for classifiers)
        if (current_arch.language and new_arch.language and
            current_arch.language != new_arch.language):
            
            issues.append(
                CompatibilityIssue(
                    issue_type="language_mismatch",
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Language mismatch: {current_arch.language} -> {new_arch.language}"
                    ),
                    affected_component="classifier",
                    suggested_fix="Use classifier trained for the same language"
                )
            )
        
        return issues, adaptations
    
    def _check_framework_compatibility(self, current_arch: ModelArchitecture,
                                     new_arch: ModelArchitecture) -> Tuple[List[CompatibilityIssue], List[str]]:
        """Check framework compatibility between models."""
        issues = []
        adaptations = []
        
        # Check model format compatibility
        if current_arch.model_format != new_arch.model_format:
            if current_arch.model_format != "unknown" and new_arch.model_format != "unknown":
                severity = ValidationSeverity.ERROR if current_arch.model_format != "tensorflow" else ValidationSeverity.WARNING
                
                issues.append(
                    CompatibilityIssue(
                        issue_type="framework_mismatch",
                        severity=severity,
                        description=(
                            f"Framework mismatch: {current_arch.model_format} -> {new_arch.model_format}"
                        ),
                        affected_component="model_loading",
                        suggested_fix="Convert models to same framework or implement bridge"
                    )
                )
                
                adaptations.append(
                    f"Implement {current_arch.model_format} to {new_arch.model_format} conversion"
                )
        
        return issues, adaptations
    
    def _check_version_compatibility(self, current_arch: ModelArchitecture,
                                   new_arch: ModelArchitecture) -> Tuple[List[CompatibilityIssue], List[str]]:
        """Check version compatibility between models."""
        issues = []
        adaptations = []
        
        # Check framework version compatibility
        if (current_arch.framework_version and new_arch.framework_version and
            current_arch.framework_version != new_arch.framework_version):
            
            issues.append(
                CompatibilityIssue(
                    issue_type="framework_version_mismatch",
                    severity=ValidationSeverity.WARNING,
                    description=(
                        f"Framework version mismatch: {current_arch.framework_version} -> "
                        f"{new_arch.framework_version}"
                    ),
                    affected_component="framework",
                    suggested_fix="Test compatibility or update framework versions"
                )
            )
        
        return issues, adaptations
    
    def _validate_model_config(self, config_name: str, config: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Validate individual model configuration."""
        issues = []
        
        try:
            # Check required configuration fields
            required_fields = ["classifier_path", "input_shape", "version"]
            for field in required_fields:
                if field not in config:
                    issues.append(
                        CompatibilityIssue(
                            issue_type="missing_config_field",
                            severity=ValidationSeverity.ERROR,
                            description=f"Missing required field '{field}' in {config_name} configuration",
                            affected_component=config_name,
                            suggested_fix=f"Add '{field}' to model configuration"
                        )
                    )
            
            # Check if model file exists
            if "classifier_path" in config:
                model_path = Path(config["classifier_path"])
                if not model_path.exists():
                    issues.append(
                        CompatibilityIssue(
                            issue_type="model_file_missing",
                            severity=ValidationSeverity.CRITICAL,
                            description=f"Model file not found: {model_path}",
                            affected_component=config_name,
                            suggested_fix="Ensure model file exists or update path"
                        )
                    )
            
            # Check input shape validity
            if "input_shape" in config:
                input_shape = config["input_shape"]
                if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
                    issues.append(
                        CompatibilityIssue(
                            issue_type="invalid_input_shape",
                            severity=ValidationSeverity.ERROR,
                            description=f"Invalid input shape in {config_name}: {input_shape}",
                            affected_component=config_name,
                            suggested_fix="Provide valid input shape as tuple or list"
                        )
                    )
            
        except Exception as e:
            issues.append(
                CompatibilityIssue(
                    issue_type="config_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Error validating {config_name} configuration: {str(e)}",
                    affected_component=config_name
                )
            )
        
        return issues
    
    def _validate_cross_model_compatibility(self, model_configs: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Validate compatibility across multiple models."""
        issues = []
        
        try:
            # Check for consistent input shapes across classifiers
            input_shapes = {}
            for config_name, config in model_configs.items():
                if "input_shape" in config:
                    input_shape = tuple(config["input_shape"])
                    if input_shape in input_shapes:
                        input_shapes[input_shape].append(config_name)
                    else:
                        input_shapes[input_shape] = [config_name]
            
            # If there are multiple different input shapes, flag as potential issue
            if len(input_shapes) > 1:
                shape_info = {str(shape): models for shape, models in input_shapes.items()}
                issues.append(
                    CompatibilityIssue(
                        issue_type="inconsistent_input_shapes",
                        severity=ValidationSeverity.WARNING,
                        description=f"Inconsistent input shapes across models: {shape_info}",
                        affected_component="cross_model",
                        suggested_fix="Ensure all models use consistent input dimensions",
                        metadata={"shape_distribution": shape_info}
                    )
                )
            
        except Exception as e:
            issues.append(
                CompatibilityIssue(
                    issue_type="cross_model_validation_error",
                    severity=ValidationSeverity.ERROR,
                    description=f"Error in cross-model validation: {str(e)}",
                    affected_component="cross_model"
                )
            )
        
        return issues
    
    def _architecture_to_dict(self, arch: ModelArchitecture) -> Dict[str, Any]:
        """Convert ModelArchitecture to dictionary for serialization."""
        return {
            "model_type": arch.model_type,
            "input_shape": arch.input_shape,
            "output_shape": arch.output_shape,
            "embedding_dimension": arch.embedding_dimension,
            "model_format": arch.model_format,
            "version": arch.version,
            "language": arch.language,
            "framework_version": arch.framework_version,
            "metadata": arch.metadata
        }
    
    def _initialize_compatibility_rules(self) -> Dict[str, Any]:
        """Initialize compatibility rules and constraints."""
        return {
            "dimension_tolerance": 0,  # No tolerance for dimension mismatches
            "framework_compatibility": {
                "tensorflow": ["tensorflow", "keras"],
                "pytorch": ["pytorch"],
                "onnx": ["onnx", "tensorflow", "pytorch"]
            },
            "critical_mismatches": [
                "embedding_dimension_mismatch",
                "model_type_mismatch",
                "model_file_missing"
            ],
            "adaptable_mismatches": [
                "input_shape_mismatch",
                "framework_mismatch",
                "framework_version_mismatch"
            ]
        }
    
    def get_compatibility_summary(self) -> Dict[str, Any]:
        """
        Get summary of compatibility validation capabilities.
        
        Returns:
            Dictionary containing compatibility validation summary
        """
        return {
            "validator_type": "ModelCompatibilityValidator",
            "cached_architectures": len(self.architecture_cache),
            "supported_formats": ["tensorflow", "keras", "h5"],
            "validation_types": [
                "model_switch_compatibility",
                "foundation_classifier_compatibility", 
                "runtime_compatibility"
            ],
            "compatibility_levels": [level.value for level in CompatibilityLevel],
            "severity_levels": [severity.value for severity in ValidationSeverity],
            "compatibility_rules": self.compatibility_rules
        }
    
    def clear_architecture_cache(self):
        """Clear the architecture cache."""
        self.architecture_cache.clear()
        logger.info("Architecture cache cleared")