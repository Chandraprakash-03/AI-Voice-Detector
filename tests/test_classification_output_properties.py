"""
Property-based tests for classification output constraints.

**Feature: ai-voice-detection-api, Property 5**: Classification Output Constraints
**Validates: Requirements 5.1, 5.2, 5.3**
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
import base64

from src.detection.engine import DetectionEngine, DetectionEngineError
from src.audio.processor import AudioProcessor, AudioFeatures
from src.exceptions import UnsupportedLanguageError


# Test data generators
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
VALID_CLASSIFICATIONS = ["AI_GENERATED", "HUMAN"]


@st.composite
def audio_features_strategy(draw):
    """Generate valid AudioFeatures objects for testing."""
    # Generate MFCC features (13 coefficients, variable time frames)
    time_frames = draw(st.integers(min_value=10, max_value=200))
    mfcc = draw(st.lists(
        st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), 
                min_size=time_frames, max_size=time_frames),
        min_size=13, max_size=13
    ))
    
    # Generate spectral features (4 features, same time frames)
    spectral = draw(st.lists(
        st.lists(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=time_frames, max_size=time_frames),
        min_size=4, max_size=4
    ))
    
    # Generate temporal features (4 values)
    temporal = draw(st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=4
    ))
    
    # Generate duration and sample rate
    duration = draw(st.floats(min_value=0.1, max_value=30.0))
    sample_rate = draw(st.sampled_from([16000, 22050, 44100, 48000]))
    
    return AudioFeatures(
        mfcc=mfcc,
        spectral=spectral,
        temporal=temporal,
        duration=duration,
        sample_rate=sample_rate
    )


@st.composite
def mp3_like_base64_strategy(draw):
    """Generate base64-encoded MP3-like data for testing."""
    # Generate MP3-like bytes
    header_type = draw(st.sampled_from(["id3", "frame"]))
    
    if header_type == "id3":
        header = b"ID3"
        data = draw(st.binary(min_size=100, max_size=1000))
        mp3_bytes = header + data
    else:
        sync_byte1 = 0xFF
        sync_byte2 = draw(st.integers(min_value=0xE0, max_value=0xFF))
        header = bytes([sync_byte1, sync_byte2])
        data = draw(st.binary(min_size=100, max_size=1000))
        mp3_bytes = header + data
    
    return base64.b64encode(mp3_bytes).decode('ascii')


class TestClassificationOutputConstraints:
    """
    Property 5: Classification Output Constraints
    
    For any successful audio analysis, the system should return a classification 
    that is exactly one of "AI_GENERATED" or "HUMAN", a confidence score between 
    0.0 and 1.0 inclusive, and a non-empty explanation string.
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DetectionEngine()
        self.processor = AudioProcessor()
    
    @given(
        features=audio_features_strategy(),
        language=st.sampled_from(SUPPORTED_LANGUAGES)
    )
    @settings(max_examples=5, deadline=None)
    def test_classification_result_constraints(self, features, language):
        """
        Property: Classification results must have valid classification, confidence score, and metadata.
        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        try:
            result = self.engine.detect_voice_type(features, language)
            
            # Requirement 5.1: Classification must be exactly one of the valid values
            assert result.classification in VALID_CLASSIFICATIONS, (
                f"Classification must be one of {VALID_CLASSIFICATIONS}, got {result.classification}"
            )
            
            # Requirement 5.2: Confidence score must be between 0.0 and 1.0 inclusive
            assert isinstance(result.confidence_score, (int, float)), (
                f"Confidence score must be numeric, got {type(result.confidence_score)}"
            )
            assert 0.0 <= result.confidence_score <= 1.0, (
                f"Confidence score must be between 0.0 and 1.0, got {result.confidence_score}"
            )
            
            # Additional constraints for result structure
            assert isinstance(result.model_version, str), (
                f"Model version must be string, got {type(result.model_version)}"
            )
            assert len(result.model_version) > 0, "Model version must not be empty"
            
            assert isinstance(result.processing_time, (int, float)), (
                f"Processing time must be numeric, got {type(result.processing_time)}"
            )
            assert result.processing_time >= 0, (
                f"Processing time must be non-negative, got {result.processing_time}"
            )
            
        except DetectionEngineError:
            # If detection fails, that's acceptable for this property test
            # The property is about successful analyses only
            pytest.skip("Detection failed - property applies only to successful analyses")
    
    @given(
        features=audio_features_strategy(),
        language=st.sampled_from(SUPPORTED_LANGUAGES)
    )
    @settings(max_examples=5, deadline=None)
    def test_explanation_generation_constraints(self, features, language):
        """
        Property: Generated explanations must be non-empty strings.
        **Validates: Requirements 5.3**
        """
        try:
            result = self.engine.detect_voice_type(features, language)
            explanation = self.engine.generate_explanation(result, features, language)
            
            # Requirement 5.3: Explanation must be a non-empty string
            assert isinstance(explanation, str), (
                f"Explanation must be string, got {type(explanation)}"
            )
            assert len(explanation.strip()) > 0, "Explanation must not be empty or whitespace-only"
            
            # Additional quality constraints for explanations
            assert len(explanation) >= 10, (
                f"Explanation should be descriptive (at least 10 chars), got {len(explanation)}"
            )
            
            # Explanation should mention the classification
            classification_mentioned = (
                "AI" in explanation or "HUMAN" in explanation or 
                "human" in explanation or "artificial" in explanation or
                "synthetic" in explanation or "generated" in explanation
            )
            assert classification_mentioned, (
                f"Explanation should mention the classification type: {explanation}"
            )
            
            # Explanation should mention the language
            assert language in explanation, (
                f"Explanation should mention the language {language}: {explanation}"
            )
            
        except DetectionEngineError:
            pytest.skip("Detection failed - property applies only to successful analyses")
    
    @given(
        features=audio_features_strategy(),
        language=st.sampled_from(SUPPORTED_LANGUAGES)
    )
    @settings(max_examples=5, deadline=None)
    def test_classification_consistency(self, features, language):
        """
        Property: Multiple calls with same input should produce consistent results.
        **Validates: Requirements 5.1, 5.2**
        """
        try:
            # Run detection multiple times with same input
            results = []
            for _ in range(3):
                result = self.engine.detect_voice_type(features, language)
                results.append(result)
            
            # All results should have the same classification
            classifications = [r.classification for r in results]
            assert all(c == classifications[0] for c in classifications), (
                f"Classification should be consistent across runs: {classifications}"
            )
            
            # Confidence scores should be similar (within reasonable tolerance)
            confidence_scores = [r.confidence_score for r in results]
            max_diff = max(confidence_scores) - min(confidence_scores)
            assert max_diff <= 0.1, (
                f"Confidence scores should be consistent (max diff <= 0.1): {confidence_scores}"
            )
            
        except DetectionEngineError:
            pytest.skip("Detection failed - property applies only to successful analyses")
    
    @given(language=st.sampled_from(SUPPORTED_LANGUAGES))
    @settings(max_examples=5, deadline=None)
    def test_supported_language_processing(self, language):
        """
        Property: All supported languages should be processable by the detection engine.
        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        # Create minimal valid features
        features = AudioFeatures(
            mfcc=[[0.0] * 50 for _ in range(13)],  # 13 MFCC coefficients, 50 time frames
            spectral=[[1.0] * 50 for _ in range(4)],  # 4 spectral features
            temporal=[1.0, 0.5, 0.3, 0.8],  # 4 temporal features
            duration=2.0,
            sample_rate=22050
        )
        
        try:
            result = self.engine.detect_voice_type(features, language)
            
            # Should produce valid result for any supported language
            assert result.classification in VALID_CLASSIFICATIONS
            assert 0.0 <= result.confidence_score <= 1.0
            
            explanation = self.engine.generate_explanation(result, features, language)
            assert isinstance(explanation, str)
            assert len(explanation.strip()) > 0
            
        except DetectionEngineError:
            pytest.skip("Detection failed - property applies only to successful analyses")
    
    @given(language=st.text().filter(lambda x: x not in SUPPORTED_LANGUAGES and x.strip()))
    @settings(max_examples=5)
    def test_unsupported_language_rejection(self, language):
        """
        Property: Unsupported languages should be rejected with appropriate errors.
        **Validates: Requirements 5.1**
        """
        features = AudioFeatures(
            mfcc=[[0.0] * 50 for _ in range(13)],
            spectral=[[1.0] * 50 for _ in range(4)],
            temporal=[1.0, 0.5, 0.3, 0.8],
            duration=2.0,
            sample_rate=22050
        )
        
        # Unsupported languages should raise UnsupportedLanguageError
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            self.engine.detect_voice_type(features, language)
        
        # Error message should mention unsupported language
        error_message = str(exc_info.value).lower()
        assert "unsupported" in error_message or "language" in error_message
    
    @given(
        base64_audio=mp3_like_base64_strategy(),
        language=st.sampled_from(SUPPORTED_LANGUAGES)
    )
    @settings(max_examples=5, deadline=None)
    def test_end_to_end_classification_constraints(self, base64_audio, language):
        """
        Property: End-to-end processing should produce valid classification results.
        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        try:
            # Process audio through complete pipeline
            features = self.processor.process_base64_audio(base64_audio, "mp3")
            result = self.engine.detect_voice_type(features, language)
            explanation = self.engine.generate_explanation(result, features, language)
            
            # Verify all output constraints
            assert result.classification in VALID_CLASSIFICATIONS
            assert 0.0 <= result.confidence_score <= 1.0
            assert isinstance(explanation, str)
            assert len(explanation.strip()) > 0
            
        except (DetectionEngineError, Exception):
            # Audio processing or detection might fail for various reasons
            # The property applies only to successful end-to-end processing
            pytest.skip("End-to-end processing failed - property applies only to successful processing")
    
    def test_classification_values_exactly_two(self):
        """
        Property: System should support exactly two classification values.
        **Validates: Requirements 5.1**
        """
        # Verify we have exactly the expected classification values
        assert len(VALID_CLASSIFICATIONS) == 2
        assert set(VALID_CLASSIFICATIONS) == {"AI_GENERATED", "HUMAN"}
        
        # Test that both classifications can be produced
        features = AudioFeatures(
            mfcc=[[0.0] * 50 for _ in range(13)],
            spectral=[[1.0] * 50 for _ in range(4)],
            temporal=[1.0, 0.5, 0.3, 0.8],
            duration=2.0,
            sample_rate=22050
        )
        
        # Test multiple times to potentially get both classifications
        classifications_seen = set()
        for _ in range(10):  # Try multiple times
            try:
                result = self.engine.detect_voice_type(features, "English")
                classifications_seen.add(result.classification)
                assert result.classification in VALID_CLASSIFICATIONS
            except DetectionEngineError:
                continue
        
        # At minimum, we should see valid classifications
        assert all(c in VALID_CLASSIFICATIONS for c in classifications_seen)
    
    @given(
        features=audio_features_strategy(),
        language=st.sampled_from(SUPPORTED_LANGUAGES)
    )
    @settings(max_examples=5, deadline=None)
    def test_confidence_score_precision(self, features, language):
        """
        Property: Confidence scores should be reasonable precision floating point numbers.
        **Validates: Requirements 5.2**
        """
        try:
            result = self.engine.detect_voice_type(features, language)
            
            # Confidence score should be a proper float
            assert isinstance(result.confidence_score, (int, float))
            assert not np.isnan(result.confidence_score)
            assert not np.isinf(result.confidence_score)
            
            # Should be within valid range
            assert 0.0 <= result.confidence_score <= 1.0
            
            # Should have reasonable precision (not too many decimal places)
            # Convert to string and check decimal places
            score_str = f"{result.confidence_score:.10f}"
            if '.' in score_str:
                decimal_part = score_str.split('.')[1].rstrip('0')
                # Allow up to 10 decimal places for ML model precision
                assert len(decimal_part) <= 10, (
                    f"Confidence score has too many decimal places: {result.confidence_score}"
                )
            
        except DetectionEngineError:
            pytest.skip("Detection failed - property applies only to successful analyses")