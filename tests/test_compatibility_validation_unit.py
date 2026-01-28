"""
Unit tests for model compatibility validation.

Tests dimensional compatibility checks, feature consistency validation,
and runtime error prevention for mismatched architectures.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.validation.compatibility_validator import (
    ModelCompatibilityValidator,
    ModelArchitecture,
    CompatibilityLevel,
    ValidationSeverity,
    CompatibilityIssue,
    ModelCompatibilityResult
)
from src.exceptions import ValidationError, ModelLoadingError


class TestModelCompatibilityValidator:
    """Test cases for ModelCompatibilityValidator."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.ml.models_base_path = "/tmp/test_models"
        return settings
    
    @pytest.fixture
    def validator(self, mock_settings):
        """Create validator instance for testing."""
        return ModelCompatibilityValidator(mock_settings)
    
    @pytest.fixture
    def sample_foundation_architecture(self):
        """Sample foundation model architecture."""
        return ModelArchitecture(
            model_type="foundation",
            input_shape=(16000,),
            output_shape=(1024,),
            embedding_dimension=1024,
            model_format="tensorflow",
            version="1.0.0",
            language=None
        )
    
    @pytest.fixture
    def sample_classifier_architecture(self):
        """Sample classifier architecture."""
        return ModelArchitecture(
            model_type="classifier",
            input_shape=(1024,),
            output_shape=(1,),
            embedding_dimension=None,
            model_format="tensorflow",
            version="1.0.0",
            language="English"
        )
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.architecture_cache == {}
        assert validator.compatibility_rules is not None
        assert "dimension_tolerance" in validator.compatibility_rules
    
    def test_dimensional_compatibility_check_compatible(self, validator, sample_foundation_architecture):
        """Test dimensional compatibility check with compatible models."""
        # Create two compatible architectures
        arch1 = sample_foundation_architecture
        arch2 = ModelArchitecture(
            model_type="foundation",
            input_shape=(16000,),
            output_shape=(1024,),
            embedding_dimension=1024,
            model_format="tensorflow",
            version="1.1.0"
        )
        
        issues, adaptations = validator._check_dimensional_compatibility(arch1, arch2)
        
        assert len(issues) == 0
        assert len(adaptations) == 0
    
    def test_dimensional_compatibility_check_incompatible(self, validator, sample_foundation_architecture):
        """Test dimensional compatibility check with incompatible models."""
        # Create incompatible architectures
        arch1 = sample_foundation_architecture
        arch2 = ModelArchitecture(
            model_type="foundation",
            input_shape=(8000,),  # Different input shape
            output_shape=(512,),  # Different output shape
            embedding_dimension=512,  # Different embedding dimension
            model_format="tensorflow",
            version="1.0.0"
        )
        
        issues, adaptations = validator._check_dimensional_compatibility(arch1, arch2)
        
        # Should have issues for input shape, output shape, and embedding dimension
        assert len(issues) >= 2  # At least input and embedding dimension issues
        assert len(adaptations) >= 2
        
        # Check for specific issue types
        issue_types = [issue.issue_type for issue in issues]
        assert "input_shape_mismatch" in issue_types
        assert "embedding_dimension_mismatch" in issue_types
    
    def test_feature_consistency_check_compatible(self, validator, sample_foundation_architecture):
        """Test feature consistency check with compatible models."""
        arch1 = sample_foundation_architecture
        arch2 = ModelArchitecture(
            model_type="foundation",  # Same type
            input_shape=(16000,),
            output_shape=(1024,),
            embedding_dimension=1024,
            model_format="tensorflow",
            version="1.0.0"
        )
        
        issues, adaptations = validator._check_feature_consistency(arch1, arch2)
        
        assert len(issues) == 0
        assert len(adaptations) == 0
    
    def test_feature_consistency_check_incompatible(self, validator, sample_foundation_architecture, sample_classifier_architecture):
        """Test feature consistency check with incompatible models."""
        # Different model types
        arch1 = sample_foundation_architecture
        arch2 = sample_classifier_architecture
        
        issues, adaptations = validator._check_feature_consistency(arch1, arch2)
        
        assert len(issues) >= 1
        issue_types = [issue.issue_type for issue in issues]
        assert "model_type_mismatch" in issue_types
    
    def test_framework_compatibility_check_compatible(self, validator, sample_foundation_architecture):
        """Test framework compatibility check with compatible models."""
        arch1 = sample_foundation_architecture
        arch2 = ModelArchitecture(
            model_type="foundation",
            input_shape=(16000,),
            output_shape=(1024,),
            embedding_dimension=1024,
            model_format="tensorflow",  # Same framework
            version="1.0.0"
        )
        
        issues, adaptations = validator._check_framework_compatibility(arch1, arch2)
        
        assert len(issues) == 0
        assert len(adaptations) == 0
    
    def test_framework_compatibility_check_incompatible(self, validator, sample_foundation_architecture):
        """Test framework compatibility check with incompatible models."""
        arch1 = sample_foundation_architecture
        arch2 = ModelArchitecture(
            model_type="foundation",
            input_shape=(16000,),
            output_shape=(1024,),
            embedding_dimension=1024,
            model_format="pytorch",  # Different framework
            version="1.0.0"
        )
        
        issues, adaptations = validator._check_framework_compatibility(arch1, arch2)
        
        assert len(issues) >= 1
        assert len(adaptations) >= 1
        
        issue_types = [issue.issue_type for issue in issues]
        assert "framework_mismatch" in issue_types
    
    @patch('tensorflow.keras.models.load_model')
    def test_get_model_architecture_tensorflow(self, mock_load_model, validator):
        """Test getting model architecture for TensorFlow model."""
        # Mock TensorFlow model
        mock_model = Mock()
        mock_model.input_shape = (None, 1024)
        mock_model.output_shape = (None, 1)
        mock_model.layers = [Mock(), Mock(), Mock()]
        mock_model.count_params.return_value = 1000000
        mock_load_model.return_value = mock_model
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            architecture = validator._get_model_architecture(tmp_path, "classifier", "English")
            
            assert architecture.model_type == "classifier"
            assert architecture.input_shape == (1024,)
            assert architecture.output_shape == (1,)
            assert architecture.model_format == "tensorflow"
            assert architecture.language == "English"
            assert "layers" in architecture.metadata
            assert "trainable_params" in architecture.metadata
            
        finally:
            os.unlink(tmp_path)
    
    def test_get_model_architecture_file_not_found(self, validator):
        """Test getting model architecture for non-existent file."""
        architecture = validator._get_model_architecture("/nonexistent/model.h5", "classifier", "English")
        
        assert architecture.model_type == "classifier"
        assert architecture.model_format == "error"
        assert architecture.language == "English"
        assert "error" in architecture.metadata
    
    def test_validate_model_config_valid(self, validator):
        """Test validation of valid model configuration."""
        config = {
            "classifier_path": "/tmp/test_model.h5",
            "input_shape": (1024,),
            "version": "1.0.0"
        }
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            issues = validator._validate_model_config("test_config", config)
        
        assert len(issues) == 0
    
    def test_validate_model_config_missing_fields(self, validator):
        """Test validation of model configuration with missing fields."""
        config = {
            "classifier_path": "/tmp/test_model.h5"
            # Missing input_shape and version
        }
        
        issues = validator._validate_model_config("test_config", config)
        
        assert len(issues) >= 2  # Missing input_shape and version
        issue_types = [issue.issue_type for issue in issues]
        assert "missing_config_field" in issue_types
    
    def test_validate_model_config_missing_file(self, validator):
        """Test validation of model configuration with missing file."""
        config = {
            "classifier_path": "/nonexistent/model.h5",
            "input_shape": (1024,),
            "version": "1.0.0"
        }
        
        issues = validator._validate_model_config("test_config", config)
        
        assert len(issues) >= 1
        issue_types = [issue.issue_type for issue in issues]
        assert "model_file_missing" in issue_types
        
        # Check severity
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) >= 1
    
    def test_validate_model_config_invalid_input_shape(self, validator):
        """Test validation of model configuration with invalid input shape."""
        config = {
            "classifier_path": "/tmp/test_model.h5",
            "input_shape": "invalid",  # Should be tuple or list
            "version": "1.0.0"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            issues = validator._validate_model_config("test_config", config)
        
        assert len(issues) >= 1
        issue_types = [issue.issue_type for issue in issues]
        assert "invalid_input_shape" in issue_types
    
    def test_validate_cross_model_compatibility_consistent(self, validator):
        """Test cross-model compatibility validation with consistent models."""
        model_configs = {
            "english": {
                "classifier_path": "/tmp/english.h5",
                "input_shape": (1024,),
                "version": "1.0.0"
            },
            "spanish": {
                "classifier_path": "/tmp/spanish.h5",
                "input_shape": (1024,),  # Same input shape
                "version": "1.0.0"
            }
        }
        
        issues = validator._validate_cross_model_compatibility(model_configs)
        
        # Should have no issues for consistent input shapes
        inconsistent_issues = [issue for issue in issues if issue.issue_type == "inconsistent_input_shapes"]
        assert len(inconsistent_issues) == 0
    
    def test_validate_cross_model_compatibility_inconsistent(self, validator):
        """Test cross-model compatibility validation with inconsistent models."""
        model_configs = {
            "english": {
                "classifier_path": "/tmp/english.h5",
                "input_shape": (1024,),
                "version": "1.0.0"
            },
            "spanish": {
                "classifier_path": "/tmp/spanish.h5",
                "input_shape": (512,),  # Different input shape
                "version": "1.0.0"
            }
        }
        
        issues = validator._validate_cross_model_compatibility(model_configs)
        
        # Should have warning for inconsistent input shapes
        inconsistent_issues = [issue for issue in issues if issue.issue_type == "inconsistent_input_shapes"]
        assert len(inconsistent_issues) >= 1
        assert inconsistent_issues[0].severity == ValidationSeverity.WARNING
    
    def test_foundation_classifier_compatibility_compatible(self, validator):
        """Test foundation-classifier compatibility validation with compatible models."""
        with patch.object(validator, '_get_model_architecture') as mock_get_arch:
            # Mock compatible architectures
            foundation_arch = ModelArchitecture(
                model_type="foundation",
                input_shape=(16000,),
                output_shape=(1024,),
                embedding_dimension=1024,
                model_format="tensorflow"
            )
            
            classifier_arch = ModelArchitecture(
                model_type="classifier",
                input_shape=(1024,),  # Matches foundation output
                output_shape=(1,),
                model_format="tensorflow",
                language="English"
            )
            
            mock_get_arch.side_effect = [foundation_arch, classifier_arch]
            
            result = validator.validate_foundation_classifier_compatibility(
                "/tmp/foundation.h5", "/tmp/classifier.h5", "English"
            )
            
            assert result.is_compatible
            assert result.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE
            assert len(result.issues) == 0
    
    def test_foundation_classifier_compatibility_incompatible(self, validator):
        """Test foundation-classifier compatibility validation with incompatible models."""
        with patch.object(validator, '_get_model_architecture') as mock_get_arch:
            # Mock incompatible architectures
            foundation_arch = ModelArchitecture(
                model_type="foundation",
                input_shape=(16000,),
                output_shape=(1024,),
                embedding_dimension=1024,
                model_format="tensorflow"
            )
            
            classifier_arch = ModelArchitecture(
                model_type="classifier",
                input_shape=(512,),  # Doesn't match foundation output
                output_shape=(1,),
                model_format="pytorch",  # Different framework
                language="Spanish"  # Different language
            )
            
            mock_get_arch.side_effect = [foundation_arch, classifier_arch]
            
            result = validator.validate_foundation_classifier_compatibility(
                "/tmp/foundation.h5", "/tmp/classifier.h5", "English"
            )
            
            assert not result.is_compatible
            assert result.compatibility_level == CompatibilityLevel.INCOMPATIBLE
            assert len(result.issues) >= 2  # Dimension and framework mismatch
            
            issue_types = [issue.issue_type for issue in result.issues]
            assert "dimension_mismatch" in issue_types
            assert "framework_mismatch" in issue_types
    
    def test_runtime_compatibility_validation(self, validator):
        """Test runtime compatibility validation."""
        model_configs = {
            "english": {
                "classifier_path": "/tmp/english.h5",
                "input_shape": (1024,),
                "version": "1.0.0"
            },
            "spanish": {
                "classifier_path": "/tmp/spanish.h5",
                "input_shape": (1024,),
                "version": "1.0.0"
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validator.validate_runtime_compatibility(model_configs)
        
        assert result is not None
        assert isinstance(result, ModelCompatibilityResult)
        assert "total_models" in result.validation_metadata
        assert result.validation_metadata["total_models"] == 2
    
    def test_architecture_to_dict(self, validator, sample_foundation_architecture):
        """Test conversion of ModelArchitecture to dictionary."""
        arch_dict = validator._architecture_to_dict(sample_foundation_architecture)
        
        assert isinstance(arch_dict, dict)
        assert arch_dict["model_type"] == "foundation"
        assert arch_dict["input_shape"] == (16000,)
        assert arch_dict["output_shape"] == (1024,)
        assert arch_dict["embedding_dimension"] == 1024
        assert arch_dict["model_format"] == "tensorflow"
    
    def test_compatibility_summary(self, validator):
        """Test getting compatibility summary."""
        summary = validator.get_compatibility_summary()
        
        assert isinstance(summary, dict)
        assert "validator_type" in summary
        assert "cached_architectures" in summary
        assert "supported_formats" in summary
        assert "validation_types" in summary
        assert "compatibility_levels" in summary
        assert "severity_levels" in summary
    
    def test_clear_architecture_cache(self, validator):
        """Test clearing architecture cache."""
        # Add something to cache
        validator.architecture_cache["test_key"] = Mock()
        assert len(validator.architecture_cache) == 1
        
        # Clear cache
        validator.clear_architecture_cache()
        assert len(validator.architecture_cache) == 0
    
    def test_compatibility_issue_creation(self):
        """Test CompatibilityIssue creation and properties."""
        issue = CompatibilityIssue(
            issue_type="test_issue",
            severity=ValidationSeverity.ERROR,
            description="Test issue description",
            affected_component="test_component",
            suggested_fix="Test fix",
            metadata={"key": "value"}
        )
        
        assert issue.issue_type == "test_issue"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.description == "Test issue description"
        assert issue.affected_component == "test_component"
        assert issue.suggested_fix == "Test fix"
        assert issue.metadata == {"key": "value"}
    
    def test_model_compatibility_result_post_init(self):
        """Test ModelCompatibilityResult post-initialization logic."""
        # Test with critical issues
        critical_issue = CompatibilityIssue(
            issue_type="critical_test",
            severity=ValidationSeverity.CRITICAL,
            description="Critical issue",
            affected_component="test"
        )
        
        result = ModelCompatibilityResult(
            compatibility_level=CompatibilityLevel.UNKNOWN,
            is_compatible=True,  # Will be overridden
            issues=[critical_issue],
            adaptation_required=False,
            adaptation_suggestions=[],
            validation_metadata={}
        )
        
        # Should be set to incompatible due to critical issue
        assert not result.is_compatible
        assert result.compatibility_level == CompatibilityLevel.INCOMPATIBLE
        
        # Test with adaptation required
        warning_issue = CompatibilityIssue(
            issue_type="warning_test",
            severity=ValidationSeverity.WARNING,
            description="Warning issue",
            affected_component="test"
        )
        
        result2 = ModelCompatibilityResult(
            compatibility_level=CompatibilityLevel.UNKNOWN,
            is_compatible=True,
            issues=[warning_issue],
            adaptation_required=True,
            adaptation_suggestions=["Test adaptation"],
            validation_metadata={}
        )
        
        # Should be compatible with adaptation
        assert result2.is_compatible
        assert result2.compatibility_level == CompatibilityLevel.COMPATIBLE_WITH_ADAPTATION


class TestModelArchitecture:
    """Test cases for ModelArchitecture dataclass."""
    
    def test_model_architecture_creation(self):
        """Test ModelArchitecture creation with all fields."""
        arch = ModelArchitecture(
            model_type="foundation",
            input_shape=(16000,),
            output_shape=(1024,),
            embedding_dimension=1024,
            model_format="tensorflow",
            version="1.0.0",
            language="English",
            framework_version="2.8.0",
            metadata={"test": "value"}
        )
        
        assert arch.model_type == "foundation"
        assert arch.input_shape == (16000,)
        assert arch.output_shape == (1024,)
        assert arch.embedding_dimension == 1024
        assert arch.model_format == "tensorflow"
        assert arch.version == "1.0.0"
        assert arch.language == "English"
        assert arch.framework_version == "2.8.0"
        assert arch.metadata == {"test": "value"}
    
    def test_model_architecture_defaults(self):
        """Test ModelArchitecture creation with default values."""
        arch = ModelArchitecture(
            model_type="classifier",
            input_shape=(1024,),
            output_shape=(1,)
        )
        
        assert arch.model_type == "classifier"
        assert arch.input_shape == (1024,)
        assert arch.output_shape == (1,)
        assert arch.embedding_dimension is None
        assert arch.model_format == "tensorflow"
        assert arch.version == "unknown"
        assert arch.language is None
        assert arch.framework_version is None
        assert arch.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__])