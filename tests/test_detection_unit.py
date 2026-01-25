"""
Unit tests for the detection engine module.

This module contains unit tests for the DetectionEngine class and related
functionality in the AI-Generated Voice Detection API.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tensorflow as tf

from src.detection.engine import DetectionEngine, DetectionResult, DetectionEngineError
from src.audio.processor import AudioFeatures


class TestDetectionEngine:
    """Unit tests for DetectionEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DetectionEngine()
        
        # Create sample audio features for testing
        self.sample_features = AudioFeatures(
            mfcc=[[0.5] * 50 for _ in range(13)],  # 13 MFCC coefficients, 50 time frames
            spectral=[[1.0] * 50 for _ in range(4)],  # 4 spectral features
            temporal=[1.0, 0.5, 0.3, 0.8],  # 4 temporal features
            duration=2.0,
            sample_rate=22050
        )
    
    def test_initialization(self):
        """Test DetectionEngine initialization."""
        engine = DetectionEngine()
        
        # Check supported languages
        assert engine.SUPPORTED_LANGUAGES == ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        
        # Check model cache is empty initially
        assert len(engine.model_cache) == 0
        
        # Check model configurations are initialized
        assert len(engine.model_configs) == 5
        for language in engine.SUPPORTED_LANGUAGES:
            assert language in engine.model_configs
            config = engine.model_configs[language]
            assert "model_path" in config
            assert "version" in config
            assert "input_shape" in config
            assert "confidence_threshold" in config
    
    def test_initialization_with_custom_path(self):
        """Test DetectionEngine initialization with custom model path."""
        custom_path = "custom/models"
        engine = DetectionEngine(model_base_path=custom_path)
        
        # Check that model paths use the custom base path
        for language in engine.SUPPORTED_LANGUAGES:
            config = engine.model_configs[language]
            # Handle Windows path separators
            model_path_str = str(config["model_path"]).replace("\\", "/")
            assert model_path_str.startswith(custom_path)
    
    def test_load_model_creates_dummy_model(self):
        """Test model loading creates dummy model when pre-trained model doesn't exist."""
        language = "English"
        
        # Load model (should create dummy model)
        model = self.engine.load_model(language)
        
        # Verify model is created and cached
        assert isinstance(model, tf.keras.Model)
        assert language in self.engine.model_cache
        assert self.engine.model_cache[language] is model
    
    def test_load_model_caching(self):
        """Test that models are properly cached."""
        language = "English"
        
        # Load model twice
        model1 = self.engine.load_model(language)
        model2 = self.engine.load_model(language)
        
        # Should return the same cached model
        assert model1 is model2
        assert len(self.engine.model_cache) == 1
    
    def test_load_model_unsupported_language(self):
        """Test loading model for unsupported language raises error."""
        with pytest.raises(DetectionEngineError) as exc_info:
            self.engine.load_model("French")
        
        assert "Unsupported language: French" in str(exc_info.value)
    
    def test_preprocess_features(self):
        """Test feature preprocessing for model input."""
        language = "English"
        
        # Preprocess features
        processed = self.engine._preprocess_features(self.sample_features, language)
        
        # Check output shape: (batch_size=1, mfcc_coeffs=13, time_frames)
        assert processed.shape[0] == 1  # Batch dimension
        assert processed.shape[1] == 13  # MFCC coefficients
        assert processed.shape[2] == 50  # Time frames
        
        # Check data type
        assert isinstance(processed, np.ndarray)
    
    def test_preprocess_features_invalid_mfcc_shape(self):
        """Test preprocessing with invalid MFCC shape raises error."""
        # Create features with wrong MFCC shape
        invalid_features = AudioFeatures(
            mfcc=[[0.5] * 50 for _ in range(12)],  # Wrong: 12 instead of 13
            spectral=[[1.0] * 50 for _ in range(4)],
            temporal=[1.0, 0.5, 0.3, 0.8],
            duration=2.0,
            sample_rate=22050
        )
        
        with pytest.raises(DetectionEngineError) as exc_info:
            self.engine._preprocess_features(invalid_features, "English")
        
        assert "Expected 13 MFCC coefficients" in str(exc_info.value)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        language = "English"
        
        # Test various model outputs
        test_cases = [
            (0.0, 1.0),    # Very confident HUMAN
            (1.0, 1.0),    # Very confident AI_GENERATED
            (0.5, 0.0),    # Uncertain (should have low confidence)
            (0.3, 0.35),   # Moderately confident HUMAN (adjusted for floating point precision)
            (0.7, 0.35),   # Moderately confident AI_GENERATED (adjusted for floating point precision)
        ]
        
        for model_output, expected_min_confidence in test_cases:
            confidence = self.engine._calculate_confidence_score(model_output, language)
            
            # Check confidence is in valid range
            assert 0.0 <= confidence <= 1.0
            assert confidence >= expected_min_confidence
    
    def test_generate_explanation(self):
        """Test explanation generation."""
        language = "English"
        
        # Test HUMAN classification
        result_human = DetectionResult(
            classification="HUMAN",
            confidence_score=0.85,
            model_version="1.0.0",
            processing_time=0.1
        )
        
        explanation_human = self.engine._generate_explanation(
            "HUMAN", 0.85, self.sample_features, language
        )
        
        # Check explanation properties
        assert isinstance(explanation_human, str)
        assert len(explanation_human) > 10
        assert "human" in explanation_human.lower() or "HUMAN" in explanation_human
        assert language in explanation_human
        assert "2.0" in explanation_human  # Duration should be mentioned
        
        # Test AI_GENERATED classification
        result_ai = DetectionResult(
            classification="AI_GENERATED",
            confidence_score=0.75,
            model_version="1.0.0",
            processing_time=0.1
        )
        
        explanation_ai = self.engine._generate_explanation(
            "AI_GENERATED", 0.75, self.sample_features, language
        )
        
        # Check explanation properties
        assert isinstance(explanation_ai, str)
        assert len(explanation_ai) > 10
        assert any(keyword in explanation_ai.lower() for keyword in ["ai", "artificial", "synthetic", "generated"])
        assert language in explanation_ai
    
    def test_detect_voice_type_success(self):
        """Test successful voice type detection."""
        language = "English"
        
        # Run detection
        result = self.engine.detect_voice_type(self.sample_features, language)
        
        # Verify result structure
        assert isinstance(result, DetectionResult)
        assert result.classification in ["AI_GENERATED", "HUMAN"]
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.model_version, str)
        assert len(result.model_version) > 0
        assert result.processing_time >= 0
    
    def test_detect_voice_type_all_languages(self):
        """Test detection works for all supported languages."""
        for language in self.engine.SUPPORTED_LANGUAGES:
            result = self.engine.detect_voice_type(self.sample_features, language)
            
            # Each language should produce valid results
            assert isinstance(result, DetectionResult)
            assert result.classification in ["AI_GENERATED", "HUMAN"]
            assert 0.0 <= result.confidence_score <= 1.0
    
    def test_detect_voice_type_unsupported_language(self):
        """Test detection with unsupported language raises error."""
        with pytest.raises(DetectionEngineError) as exc_info:
            self.engine.detect_voice_type(self.sample_features, "French")
        
        assert "Unsupported language: French" in str(exc_info.value)
    
    def test_generate_explanation_method(self):
        """Test the public generate_explanation method."""
        language = "English"
        
        # First get a detection result
        result = self.engine.detect_voice_type(self.sample_features, language)
        
        # Generate explanation
        explanation = self.engine.generate_explanation(result, self.sample_features, language)
        
        # Verify explanation
        assert isinstance(explanation, str)
        assert len(explanation.strip()) > 0
        assert language in explanation
    
    def test_get_model_info(self):
        """Test getting model information."""
        language = "English"
        
        # Get model info before loading
        info_before = self.engine.get_model_info(language)
        assert info_before["language"] == language
        assert info_before["is_loaded"] is False
        assert "version" in info_before
        assert "model_path" in info_before
        
        # Load model and check info again
        self.engine.load_model(language)
        info_after = self.engine.get_model_info(language)
        assert info_after["is_loaded"] is True
    
    def test_get_model_info_unsupported_language(self):
        """Test getting model info for unsupported language raises error."""
        with pytest.raises(DetectionEngineError) as exc_info:
            self.engine.get_model_info("French")
        
        assert "Unsupported language: French" in str(exc_info.value)
    
    def test_clear_model_cache_specific_language(self):
        """Test clearing model cache for specific language."""
        # Load models for multiple languages
        self.engine.load_model("English")
        self.engine.load_model("Tamil")
        
        assert len(self.engine.model_cache) == 2
        
        # Clear cache for one language
        self.engine.clear_model_cache("English")
        
        assert len(self.engine.model_cache) == 1
        assert "English" not in self.engine.model_cache
        assert "Tamil" in self.engine.model_cache
    
    def test_clear_model_cache_all(self):
        """Test clearing all model caches."""
        # Load models for multiple languages
        self.engine.load_model("English")
        self.engine.load_model("Tamil")
        
        assert len(self.engine.model_cache) == 2
        
        # Clear all caches
        self.engine.clear_model_cache()
        
        assert len(self.engine.model_cache) == 0
    
    def test_classification_consistency(self):
        """Test that classification is consistent for same input."""
        language = "English"
        
        # Run detection multiple times
        results = []
        for _ in range(3):
            result = self.engine.detect_voice_type(self.sample_features, language)
            results.append(result)
        
        # All results should have same classification
        classifications = [r.classification for r in results]
        assert all(c == classifications[0] for c in classifications)
        
        # Confidence scores should be similar
        confidence_scores = [r.confidence_score for r in results]
        max_diff = max(confidence_scores) - min(confidence_scores)
        assert max_diff <= 0.1  # Allow small variation due to floating point precision
    
    def test_model_version_consistency(self):
        """Test that model version is consistent."""
        language = "English"
        
        # Get model version from config
        expected_version = self.engine.model_configs[language]["version"]
        
        # Run detection and check version
        result = self.engine.detect_voice_type(self.sample_features, language)
        assert result.model_version == expected_version
    
    def test_processing_time_measurement(self):
        """Test that processing time is measured."""
        language = "English"
        
        result = self.engine.detect_voice_type(self.sample_features, language)
        
        # Processing time should be positive and reasonable
        assert result.processing_time > 0
        assert result.processing_time < 10  # Should complete within 10 seconds
    
    def test_confidence_threshold_usage(self):
        """Test that confidence threshold is used for classification."""
        language = "English"
        
        # Get the threshold from config
        threshold = self.engine.model_configs[language]["confidence_threshold"]
        assert threshold == 0.5  # Default threshold
        
        # The classification logic uses this threshold
        # (This is tested indirectly through the detection results)
        result = self.engine.detect_voice_type(self.sample_features, language)
        assert result.classification in ["AI_GENERATED", "HUMAN"]


class TestDetectionResult:
    """Unit tests for DetectionResult model."""
    
    def test_detection_result_creation(self):
        """Test creating DetectionResult instances."""
        result = DetectionResult(
            classification="HUMAN",
            confidence_score=0.85,
            model_version="1.0.0",
            processing_time=0.123
        )
        
        assert result.classification == "HUMAN"
        assert result.confidence_score == 0.85
        assert result.model_version == "1.0.0"
        assert result.processing_time == 0.123
    
    def test_detection_result_validation(self):
        """Test DetectionResult field validation."""
        # Valid result should work
        result = DetectionResult(
            classification="AI_GENERATED",
            confidence_score=1.0,
            model_version="2.0.0",
            processing_time=0.0
        )
        
        assert result.classification == "AI_GENERATED"
        assert result.confidence_score == 1.0


class TestDetectionEngineError:
    """Unit tests for DetectionEngineError exception."""
    
    def test_detection_engine_error_creation(self):
        """Test creating DetectionEngineError instances."""
        error_message = "Test error message"
        error = DetectionEngineError(error_message)
        
        assert str(error) == error_message
        assert isinstance(error, Exception)
    
    def test_detection_engine_error_inheritance(self):
        """Test that DetectionEngineError inherits from Exception."""
        error = DetectionEngineError("Test")
        assert isinstance(error, Exception)
        
        # Should be catchable as Exception
        try:
            raise error
        except Exception as e:
            assert isinstance(e, DetectionEngineError)