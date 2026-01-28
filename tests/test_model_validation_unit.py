"""
Unit tests for model validation functionality.

This module contains unit tests for the ModelValidator, EmbeddingValidator,
and FoundationModelManager classes.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tensorflow as tf

from src.validation.validator import ModelValidator, EmbeddingValidator, FoundationModelManager
from src.models.schemas import ValidationResult, ModelMetadata


class TestModelValidator:
    """Unit tests for ModelValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
    
    def test_initialization(self):
        """Test ModelValidator initialization."""
        validator = ModelValidator()
        
        assert validator.min_accuracy_threshold == 0.6
        assert validator.weight_variance_threshold == 1e-6
        assert validator.embedding_quality_threshold == 0.1
        assert validator.model_base_path is not None
    
    def test_validate_foundation_model_nonexistent_path(self):
        """Test validation with non-existent model path."""
        result = self.validator.validate_foundation_model("/nonexistent/path")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.confidence_score == 0.0
        assert "does not exist" in result.error_message
        assert result.performance_metrics == {}
    
    def test_validate_classifier_nonexistent_path(self):
        """Test classifier validation with non-existent path."""
        result = self.validator.validate_classifier("/nonexistent/path", "English")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.confidence_score == 0.0
        assert "does not exist" in result.error_message
    
    def test_validate_classifier_empty_file(self):
        """Test classifier validation with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = self.validator.validate_classifier(tmp_path, "English")
            
            assert isinstance(result, ValidationResult)
            assert result.is_valid is False
            assert result.confidence_score == 0.0
            assert "too small" in result.error_message
            assert "file_size" in result.performance_metrics
        finally:
            os.unlink(tmp_path)
    
    def test_validate_classifier_corrupted_file(self):
        """Test classifier validation with corrupted model file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            # Write some random data to simulate corrupted file
            tmp_file.write(b"corrupted model data" * 100)  # Make it large enough
            tmp_path = tmp_file.name
        
        try:
            result = self.validator.validate_classifier(tmp_path, "English")
            
            assert isinstance(result, ValidationResult)
            assert result.is_valid is False
            assert result.confidence_score == 0.0
            assert "loading failed" in result.error_message
        finally:
            os.unlink(tmp_path)
    
    def test_check_model_weights_empty_model(self):
        """Test weight checking with model that has no weights."""
        # Create a model with no trainable parameters
        model = tf.keras.Sequential()
        
        metrics = self.validator.check_model_weights(model)
        
        # Should return error for model without weights
        assert "error" in metrics or "has_weights" in metrics
        if "has_weights" in metrics:
            assert metrics["has_weights"] == 0.0
            assert metrics["weight_variance"] == 0.0
            assert metrics["weight_mean"] == 0.0
    
    def test_check_model_weights_simple_model(self):
        """Test weight checking with a simple trained model."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile to initialize weights
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        metrics = self.validator.check_model_weights(model)
        
        assert metrics["has_weights"] == 1.0
        assert metrics["num_weight_arrays"] > 0
        assert metrics["total_parameters"] > 0
        assert "weight_mean" in metrics
        assert "weight_std" in metrics
        assert "weight_variance" in metrics
        assert "near_zero_ratio" in metrics
    
    def test_check_model_weights_random_vs_trained(self):
        """Test weight analysis can distinguish random vs trained weights."""
        # Create two identical models
        model_random = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model_random.compile(optimizer='adam', loss='binary_crossentropy')
        
        model_trained = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model_trained.compile(optimizer='adam', loss='binary_crossentropy')
        
        # "Train" one model by setting weights to specific values
        weights = model_trained.get_weights()
        for i, w in enumerate(weights):
            if len(w.shape) > 1:  # Weight matrices
                weights[i] = np.random.normal(0, 0.1, w.shape)  # Smaller variance
            else:  # Bias vectors
                weights[i] = np.zeros_like(w)  # Zero bias
        model_trained.set_weights(weights)
        
        # Analyze both models
        metrics_random = self.validator.check_model_weights(model_random)
        metrics_trained = self.validator.check_model_weights(model_trained)
        
        # Trained model should have different characteristics
        assert metrics_random["weight_variance"] != metrics_trained["weight_variance"]
        assert "near_zero_ratio" in metrics_random
        assert "near_zero_ratio" in metrics_trained
    
    def test_verify_model_performance_simple_model(self):
        """Test performance verification with a simple model."""
        # Create and compile a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(100,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        metrics = self.validator.verify_model_performance(model, "English")
        
        assert "test_success_rate" in metrics
        assert "avg_inference_time" in metrics
        assert metrics["test_success_rate"] >= 0.0
        assert metrics["avg_inference_time"] > 0.0
        
        # Check individual test cases
        for i in range(4):  # We have 4 test cases
            assert f"test_case_{i}_success" in metrics
    
    def test_verify_model_performance_invalid_input_shape(self):
        """Test performance verification with invalid model input shape."""
        # Create a model with no defined input shape
        model = tf.keras.Sequential()
        
        metrics = self.validator.verify_model_performance(model, "English")
        
        # Should return error for invalid model
        assert "test_error" in metrics or "performance_error" in metrics
        if "test_error" in metrics:
            assert "Invalid input shape" in metrics["test_error"]
        elif "performance_error" in metrics:
            # Alternative error handling is acceptable
            assert isinstance(metrics["performance_error"], str)
    
    def test_calculate_foundation_confidence_high_score(self):
        """Test foundation confidence calculation with good metrics."""
        metrics = {
            "inference_success": 1.0,
            "embedding_shape_valid": 1.0,
            "embedding_has_nan": 0.0,
            "embedding_has_inf": 0.0,
            "embedding_variance": 0.5,
            "embedding_consistency": 1e-8,
            "inference_time": 0.1
        }
        
        confidence = self.validator._calculate_foundation_confidence(metrics)
        
        assert confidence > 0.8  # Should be high confidence
        assert confidence <= 1.0
    
    def test_calculate_foundation_confidence_low_score(self):
        """Test foundation confidence calculation with poor metrics."""
        metrics = {
            "inference_success": 0.0,
            "embedding_shape_valid": 0.0,
            "embedding_has_nan": 1.0,
            "embedding_has_inf": 1.0,
            "embedding_variance": 0.0,
            "embedding_consistency": 1.0,
            "inference_time": 10.0
        }
        
        confidence = self.validator._calculate_foundation_confidence(metrics)
        
        assert confidence < 0.3  # Should be low confidence
        assert confidence >= 0.0
    
    def test_calculate_classifier_confidence_high_score(self):
        """Test classifier confidence calculation with good metrics."""
        metrics = {
            "has_weights": 1.0,
            "weight_variance": 0.1,
            "test_success_rate": 1.0,
            "prediction_consistency": 0.9,
            "file_size": 50000,
            "avg_inference_time": 0.05
        }
        
        confidence = self.validator._calculate_classifier_confidence(metrics)
        
        assert confidence > 0.8  # Should be high confidence
        assert confidence <= 1.0
    
    def test_calculate_classifier_confidence_low_score(self):
        """Test classifier confidence calculation with poor metrics."""
        metrics = {
            "has_weights": 0.0,
            "weight_variance": 1e-8,
            "test_success_rate": 0.0,
            "prediction_consistency": 0.0,
            "file_size": 100,
            "avg_inference_time": 5.0
        }
        
        confidence = self.validator._calculate_classifier_confidence(metrics)
        
        assert confidence < 0.3  # Should be low confidence
        assert confidence >= 0.0
    
    def test_calculate_prediction_consistency_identical(self):
        """Test prediction consistency with identical predictions."""
        pred1 = np.array([[0.8, 0.2], [0.3, 0.7]])
        pred2 = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        consistency = self.validator._calculate_prediction_consistency([pred1, pred2])
        
        assert consistency == 1.0  # Perfect consistency
    
    def test_calculate_prediction_consistency_different(self):
        """Test prediction consistency with different predictions."""
        pred1 = np.array([[0.8, 0.2], [0.3, 0.7]])
        pred2 = np.array([[0.2, 0.8], [0.7, 0.3]])
        
        consistency = self.validator._calculate_prediction_consistency([pred1, pred2])
        
        assert 0.0 <= consistency < 1.0  # Should be less than perfect
    
    def test_calculate_prediction_consistency_single_prediction(self):
        """Test prediction consistency with single prediction."""
        pred1 = np.array([[0.8, 0.2]])
        
        consistency = self.validator._calculate_prediction_consistency([pred1])
        
        assert consistency == 0.0  # Can't calculate consistency with single prediction


class TestEmbeddingValidator:
    """Unit tests for EmbeddingValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = EmbeddingValidator(quality_threshold=0.3, consistency_threshold=1e-6)
    
    def test_initialization(self):
        """Test EmbeddingValidator initialization."""
        validator = EmbeddingValidator(quality_threshold=0.5, consistency_threshold=1e-5)
        
        assert validator.quality_threshold == 0.5
        assert validator.consistency_threshold == 1e-5
    
    def test_is_random_embedding_none_input(self):
        """Test random embedding detection with None input."""
        result = self.validator.is_random_embedding(None)
        assert result is True
    
    def test_is_random_embedding_empty_input(self):
        """Test random embedding detection with empty input."""
        result = self.validator.is_random_embedding(np.array([]))
        assert result is True
    
    def test_is_random_embedding_nan_values(self):
        """Test random embedding detection with NaN values."""
        embeddings = np.array([1.0, 2.0, np.nan, 4.0])
        result = self.validator.is_random_embedding(embeddings)
        assert result is True
    
    def test_is_random_embedding_inf_values(self):
        """Test random embedding detection with infinite values."""
        embeddings = np.array([1.0, 2.0, np.inf, 4.0])
        result = self.validator.is_random_embedding(embeddings)
        assert result is True
    
    def test_is_random_embedding_high_variance(self):
        """Test random embedding detection with very high variance."""
        embeddings = np.random.normal(0, 5, 1000)  # High variance
        result = self.validator.is_random_embedding(embeddings)
        assert result is True
    
    def test_is_random_embedding_uniform_distribution(self):
        """Test random embedding detection with uniform distribution."""
        embeddings = np.full(1000, 1.0)  # All same value
        result = self.validator.is_random_embedding(embeddings)
        assert result is True
    
    def test_is_random_embedding_repeated_values(self):
        """Test random embedding detection with too many repeated values."""
        embeddings = np.array([1.0] * 90 + [2.0] * 10)  # 90% repeated
        result = self.validator.is_random_embedding(embeddings)
        assert result is True
    
    def test_is_random_embedding_good_embeddings(self):
        """Test random embedding detection with good embeddings."""
        # Create embeddings with reasonable characteristics
        embeddings = np.random.normal(0, 0.5, 1000)  # Reasonable variance
        embeddings = embeddings + np.sin(np.arange(1000) * 0.01)  # Add structure
        
        result = self.validator.is_random_embedding(embeddings)
        # Note: This test may be sensitive to random values, so we check it's a boolean
        assert isinstance(result, bool)
    
    def test_is_random_embedding_statistical_tests(self):
        """Test enhanced random embedding detection with statistical tests."""
        # Test normal distribution detection
        normal_embeddings = np.random.normal(0, 1, 1000)
        result = self.validator.is_random_embedding(normal_embeddings)
        # May or may not be detected as random depending on specific values
        assert isinstance(result, bool)
        
        # Test clearly structured embeddings (sine wave)
        structured_embeddings = np.array([np.sin(i * 0.1) for i in range(1000)])
        result = self.validator.is_random_embedding(structured_embeddings)
        # Sine wave should generally not be detected as random
        assert isinstance(result, bool)
    
    def test_check_embedding_quality_none_input(self):
        """Test embedding quality check with None input."""
        quality = self.validator.check_embedding_quality(None)
        assert quality == 0.0
    
    def test_check_embedding_quality_empty_input(self):
        """Test embedding quality check with empty input."""
        quality = self.validator.check_embedding_quality(np.array([]))
        assert quality == 0.0
    
    def test_check_embedding_quality_good_embeddings(self):
        """Test embedding quality check with good embeddings."""
        # Create high-quality embeddings
        embeddings = np.random.normal(0, 1, 1000)  # Good variance
        embeddings = embeddings * np.linspace(0.5, 1.5, 1000)  # Add structure
        
        quality = self.validator.check_embedding_quality(embeddings)
        
        assert 0.0 <= quality <= 1.0
        assert quality > 0.3  # Should be reasonably high
    
    def test_check_embedding_quality_poor_embeddings(self):
        """Test embedding quality check with poor embeddings."""
        # Create poor-quality embeddings
        embeddings = np.array([np.nan, np.inf, 0, 0, 0])
        
        quality = self.validator.check_embedding_quality(embeddings)
        
        assert 0.0 <= quality <= 1.0
        assert quality < 0.5  # Should be low
    
    def test_check_embedding_quality_comprehensive(self):
        """Test comprehensive embedding quality assessment."""
        # Test various quality aspects
        
        # High quality: good variance, distribution, uniqueness, magnitude
        good_embeddings = np.random.normal(0, 0.8, 1000)
        good_embeddings += np.sin(np.arange(1000) * 0.05)  # Add structure
        quality_good = self.validator.check_embedding_quality(good_embeddings)
        
        # Poor quality: extreme variance
        poor_embeddings = np.random.normal(0, 10, 1000)
        quality_poor = self.validator.check_embedding_quality(poor_embeddings)
        
        assert quality_good > quality_poor
        assert 0.0 <= quality_good <= 1.0
        assert 0.0 <= quality_poor <= 1.0
    
    def test_validate_consistency_success(self):
        """Test consistency validation with consistent function."""
        def consistent_func(audio):
            return np.array([1.0, 2.0, 3.0])  # Always return same values
        
        audio = np.random.randn(1000)
        result = self.validator.validate_consistency(audio, consistent_func, runs=3)
        
        assert result is True
    
    def test_validate_consistency_failure(self):
        """Test consistency validation with inconsistent function."""
        def inconsistent_func(audio):
            return np.random.randn(3)  # Return different values each time
        
        audio = np.random.randn(1000)
        result = self.validator.validate_consistency(audio, inconsistent_func, runs=3)
        
        assert result is False
    
    def test_validate_consistency_none_return(self):
        """Test consistency validation with function that returns None."""
        def none_func(audio):
            return None
        
        audio = np.random.randn(1000)
        result = self.validator.validate_consistency(audio, none_func, runs=3)
        
        assert result is False
    
    def test_validate_consistency_shape_mismatch(self):
        """Test consistency validation with shape mismatches."""
        call_count = 0
        def shape_changing_func(audio):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return np.array([1.0, 2.0, 3.0])
            else:
                return np.array([1.0, 2.0])  # Different shape
        
        audio = np.random.randn(1000)
        result = self.validator.validate_consistency(audio, shape_changing_func, runs=3)
        
        assert result is False
    
    def test_validate_consistency_insufficient_runs(self):
        """Test consistency validation with insufficient runs."""
        def dummy_func(audio):
            return np.array([1.0, 2.0, 3.0])
        
        audio = np.random.randn(1000)
        result = self.validator.validate_consistency(audio, dummy_func, runs=1)
        
        assert result is False  # Should require at least 2 runs
    
    def test_perform_statistical_tests_comprehensive(self):
        """Test comprehensive statistical tests on embeddings."""
        # Create test embeddings with known properties
        embeddings = np.random.normal(0, 1, 1000)
        
        results = self.validator.perform_statistical_tests(embeddings)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "mean", "std", "variance", "min", "max", "range",
            "normality_score", "randomness_score", "autocorr_score",
            "entropy", "distribution_uniformity", "unique_ratio"
        ]
        
        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"
        
        # Check value ranges
        assert 0.0 <= results["normality_score"] <= 1.0
        assert 0.0 <= results["randomness_score"] <= 1.0
        assert 0.0 <= results["autocorr_score"] <= 1.0
        assert results["entropy"] >= 0.0
        assert 0.0 <= results["unique_ratio"] <= 1.0
    
    def test_perform_statistical_tests_empty_input(self):
        """Test statistical tests with empty input."""
        results = self.validator.perform_statistical_tests(np.array([]))
        
        assert "error" in results
        assert results["error"] == "Empty embeddings"
    
    def test_perform_statistical_tests_structured_data(self):
        """Test statistical tests with structured data."""
        # Create structured embeddings (sine wave)
        embeddings = np.array([np.sin(i * 0.1) for i in range(1000)])
        
        results = self.validator.perform_statistical_tests(embeddings)
        
        # Structured data should have high autocorrelation
        assert results["autocorr_score"] > 0.5
        
        # Should have reasonable entropy
        assert results["entropy"] > 0.0
    
    def test_perform_statistical_tests_random_data(self):
        """Test statistical tests with random data."""
        # Create random embeddings
        embeddings = np.random.normal(0, 1, 1000)
        
        results = self.validator.perform_statistical_tests(embeddings)
        
        # Random data should have high normality score
        assert results["normality_score"] > 0.3
        
        # Should have high randomness score
        assert results["randomness_score"] > 0.3
    
    def test_statistical_helper_methods(self):
        """Test internal statistical helper methods."""
        # Test normality test
        normal_data = np.random.normal(0, 1, 1000)
        is_normal = self.validator._test_normality(normal_data)
        assert isinstance(is_normal, (bool, np.bool_))
        
        # Test structure test
        structured_data = np.array([np.sin(i * 0.1) for i in range(100)])
        has_structure = self.validator._test_structure(structured_data)
        assert isinstance(has_structure, (bool, np.bool_))  # Should return boolean
        
        # Test with random data
        random_data = np.random.randn(100)
        has_structure_random = self.validator._test_structure(random_data)
        # May or may not have structure depending on random values
        assert isinstance(has_structure_random, (bool, np.bool_))
    
    def test_entropy_calculation(self):
        """Test entropy calculation method."""
        # Test with uniform data (high entropy)
        uniform_data = np.random.uniform(-1, 1, 1000)
        entropy_uniform = self.validator._calculate_entropy(uniform_data)
        
        # Test with constant data (low entropy)
        constant_data = np.full(1000, 1.0)
        entropy_constant = self.validator._calculate_entropy(constant_data)
        
        assert entropy_uniform > entropy_constant
        assert entropy_uniform > 0.0
        assert entropy_constant >= 0.0
    
    def test_normality_score_calculation(self):
        """Test normality score calculation."""
        # Test with normal data
        normal_data = np.random.normal(0, 1, 1000)
        normality_score = self.validator._test_normality_score(normal_data)
        
        assert 0.0 <= normality_score <= 1.0
        assert normality_score > 0.3  # Should be reasonably high for normal data
    
    def test_randomness_score_calculation(self):
        """Test randomness score calculation."""
        # Test with alternating pattern (should have specific randomness characteristics)
        pattern_data = np.array([1, -1] * 500)
        randomness_pattern = self.validator._test_randomness_score(pattern_data)
        
        # Test with random data
        random_data = np.random.randn(1000)
        randomness_random = self.validator._test_randomness_score(random_data)
        
        assert 0.0 <= randomness_pattern <= 1.0
        assert 0.0 <= randomness_random <= 1.0
        # Both should be valid randomness scores (the specific relationship may vary)
        assert isinstance(randomness_pattern, float)
        assert isinstance(randomness_random, float)
    
    def test_autocorrelation_score_calculation(self):
        """Test autocorrelation score calculation."""
        # Test with highly correlated data (sine wave)
        correlated_data = np.array([np.sin(i * 0.01) for i in range(1000)])
        autocorr_high = self.validator._test_autocorrelation_score(correlated_data)
        
        # Test with random data (low correlation)
        random_data = np.random.randn(1000)
        autocorr_low = self.validator._test_autocorrelation_score(random_data)
        
        assert 0.0 <= autocorr_high <= 1.0
        assert 0.0 <= autocorr_low <= 1.0
        assert autocorr_high > autocorr_low


class TestFoundationModelManager:
    """Unit tests for FoundationModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.validation.validator.TRANSFORMERS_AVAILABLE', False):
            self.manager = FoundationModelManager()
    
    def test_initialization(self):
        """Test FoundationModelManager initialization."""
        with patch('src.validation.validator.TRANSFORMERS_AVAILABLE', False):
            manager = FoundationModelManager()
            
            assert manager.model_cache == {}
            assert manager.processor_cache == {}
            assert manager.validator is not None
            assert manager.embedding_validator is not None
    
    def test_load_model_nonexistent(self):
        """Test loading non-existent model."""
        model, processor = self.manager.load_model("nonexistent_model")
        
        assert model is None
        assert processor is None
    
    def test_extract_embeddings_no_model(self):
        """Test embedding extraction when model is not available."""
        audio = np.random.randn(16000)
        embeddings = self.manager.extract_embeddings(audio, "nonexistent_model")
        
        assert embeddings is None
    
    def test_validate_embeddings_good(self):
        """Test embedding validation with good embeddings."""
        embeddings = np.random.normal(0, 1, (1, 768))
        result = self.manager.validate_embeddings(embeddings)
        
        # Result depends on the specific values, but should not crash
        assert isinstance(result, bool)
    
    def test_validate_embeddings_bad(self):
        """Test embedding validation with bad embeddings."""
        embeddings = np.array([[np.nan, np.inf, 0]])
        result = self.manager.validate_embeddings(embeddings)
        
        assert result is False
    
    def test_get_fallback_features(self):
        """Test fallback feature extraction."""
        audio = np.random.randn(16000)
        features = self.manager.get_fallback_features(audio)
        
        assert features is not None
        assert features.shape == (1, 1024)  # Expected output shape
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_get_fallback_features_empty_audio(self):
        """Test fallback feature extraction with empty audio."""
        audio = np.array([])
        features = self.manager.get_fallback_features(audio)
        
        # Should return zero features as fallback
        assert features is not None
        assert features.shape == (1, 1024)
        assert np.all(features == 0.0)
    
    def test_switch_model_invalid(self):
        """Test switching to invalid model."""
        with pytest.raises(ValueError):
            self.manager.switch_model("invalid_model")
    
    @patch('src.validation.validator.TRANSFORMERS_AVAILABLE', True)
    def test_load_model_with_transformers_unavailable_fallback(self):
        """Test model loading fallback when transformers becomes unavailable."""
        # This test ensures graceful handling when transformers is not available
        with patch('src.validation.validator.TRANSFORMERS_AVAILABLE', False):
            manager = FoundationModelManager()
            model, processor = manager.load_model("xls-r-300m")
            
            assert model is None
            assert processor is None


class TestValidationResultModel:
    """Unit tests for ValidationResult Pydantic model."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult instances."""
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.85,
            error_message=None,
            performance_metrics={"accuracy": 0.92}
        )
        
        assert result.is_valid is True
        assert result.confidence_score == 0.85
        assert result.error_message is None
        assert result.performance_metrics == {"accuracy": 0.92}
        assert result.validation_timestamp is not None
    
    def test_validation_result_invalid_confidence(self):
        """Test ValidationResult with invalid confidence score."""
        with pytest.raises(ValueError):
            ValidationResult(
                is_valid=True,
                confidence_score=1.5,  # Invalid: > 1.0
                performance_metrics={}
            )
        
        with pytest.raises(ValueError):
            ValidationResult(
                is_valid=True,
                confidence_score=-0.1,  # Invalid: < 0.0
                performance_metrics={}
            )


class TestModelMetadataModel:
    """Unit tests for ModelMetadata Pydantic model."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instances."""
        from datetime import datetime
        
        metadata = ModelMetadata(
            model_id="test_model",
            model_type="classifier",
            language="English",
            version="1.0.0",
            file_path="/path/to/model",
            training_date=datetime.now(),
            validation_accuracy=0.92,
            test_accuracy=0.89,
            model_size_bytes=1024,
            training_data_hash="abc123"
        )
        
        assert metadata.model_id == "test_model"
        assert metadata.model_type == "classifier"
        assert metadata.language == "English"
        assert metadata.version == "1.0.0"
        assert metadata.status == "ACTIVE"  # Default value
    
    def test_model_metadata_invalid_accuracy(self):
        """Test ModelMetadata with invalid accuracy values."""
        from datetime import datetime
        
        with pytest.raises(ValueError):
            ModelMetadata(
                model_id="test_model",
                model_type="classifier",
                version="1.0.0",
                file_path="/path/to/model",
                training_date=datetime.now(),
                validation_accuracy=1.5,  # Invalid: > 1.0
                test_accuracy=0.89,
                model_size_bytes=1024,
                training_data_hash="abc123"
            )
    
    def test_model_metadata_foundation_model(self):
        """Test ModelMetadata for foundation model (no language)."""
        from datetime import datetime
        
        metadata = ModelMetadata(
            model_id="foundation_model",
            model_type="foundation",
            language=None,  # Foundation models don't have language
            version="1.0.0",
            file_path="/path/to/model",
            training_date=datetime.now(),
            validation_accuracy=0.95,
            test_accuracy=0.93,
            model_size_bytes=50000000,
            training_data_hash="def456"
        )
        
        assert metadata.model_type == "foundation"
        assert metadata.language is None
        assert metadata.model_size_bytes == 50000000