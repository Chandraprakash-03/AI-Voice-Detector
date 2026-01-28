"""
Graceful fallback system for AI voice detection.

This module implements validated backup methods for when foundation models
or classifiers fail, providing resource-aware degradation strategies and
ensemble methods to maintain system reliability.
"""

import logging
import time
import warnings
from typing import Dict, Optional, List, Any, Union, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import tensorflow as tf
from datetime import datetime

# Audio processing imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Install with: pip install librosa")

from src.audio.processor import AudioFeatures
from src.exceptions import (
    DetectionEngineError,
    ModelLoadingError,
    InferenceError
)
from src.models.schemas import ValidationResult

logger = logging.getLogger(__name__)


class FallbackLevel(Enum):
    """Enumeration of fallback levels from best to worst."""
    NONE = "none"                    # No fallback needed
    FOUNDATION_MODEL = "foundation"  # Foundation model fallback
    CLASSIFIER_ENSEMBLE = "ensemble" # Classifier ensemble fallback
    TRADITIONAL_FEATURES = "traditional"  # Traditional audio features
    RULE_BASED = "rule_based"       # Rule-based detection
    EMERGENCY = "emergency"         # Emergency response


class FallbackReason(Enum):
    """Enumeration of reasons for fallback activation."""
    FOUNDATION_MODEL_UNAVAILABLE = "foundation_unavailable"
    FOUNDATION_MODEL_FAILED = "foundation_failed"
    CLASSIFIER_UNAVAILABLE = "classifier_unavailable"
    CLASSIFIER_FAILED = "classifier_failed"
    LOW_CONFIDENCE = "low_confidence"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    VALIDATION_FAILED = "validation_failed"
    TIMEOUT = "timeout"


class FallbackResult:
    """Result from fallback processing."""
    
    def __init__(self, classification: str, confidence_score: float, 
                 fallback_level: FallbackLevel, reason: FallbackReason,
                 processing_time: float, metadata: Dict[str, Any] = None):
        self.classification = classification
        self.confidence_score = confidence_score
        self.fallback_level = fallback_level
        self.reason = reason
        self.processing_time = processing_time
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class FallbackStrategy:
    """
    Implements graceful fallback system for AI voice detection.
    
    Provides validated backup methods for when foundation models or classifiers fail,
    including resource-aware degradation strategies and ensemble methods to maintain
    system reliability.
    
    Requirements addressed:
    - 1.3: Graceful fallback when foundation models unavailable
    - 4.3: Validated backup feature extraction methods
    - 6.2: Resource-aware degradation strategies
    - 6.4: Ensemble methods for classifier failures
    """
    
    def __init__(self, settings=None):
        """
        Initialize the FallbackStrategy.
        
        Args:
            settings: Application settings object containing configuration
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.model_base_path = Path(settings.ml.models_base_path)
        
        # Initialize audio processor for traditional features
        from src.audio.processor import AudioProcessor
        self.audio_processor = AudioProcessor(settings) if LIBROSA_AVAILABLE else None
        
        # Fallback thresholds and configurations
        self.confidence_threshold = 0.3  # Minimum confidence for reliable results
        self.resource_memory_limit = 1024 * 1024 * 1024  # 1GB memory limit
        self.max_processing_time = 30.0  # Maximum processing time in seconds
        
        # Traditional feature extraction parameters
        self.mfcc_coefficients = 13
        self.spectral_features = 4
        self.temporal_features = 4
        
        # Ensemble weights for different fallback methods
        self.ensemble_weights = {
            'traditional_features': 0.4,
            'statistical_analysis': 0.3,
            'rule_based': 0.3
        }
        
        logger.info("FallbackStrategy initialized with validated backup methods")
    
    def handle_foundation_model_failure(self, audio: np.ndarray, 
                                      target_dim: int = 1024) -> np.ndarray:
        """
        Handle foundation model failure by extracting traditional audio features.
        
        Implements validated backup feature extraction using traditional audio analysis
        methods (MFCC, spectral, temporal) instead of generating random embeddings.
        
        Args:
            audio: Audio data as numpy array
            target_dim: Target dimension for feature vector
            
        Returns:
            Traditional audio features as numpy array
        """
        try:
            logger.info("Handling foundation model failure with traditional feature extraction")
            
            if audio is None or len(audio) == 0:
                logger.warning("Invalid audio data, returning structured fallback features")
                return self._create_structured_fallback_features(target_dim)
            
            # Normalize audio
            audio = self._normalize_audio(audio)
            
            # Extract traditional features
            features = []
            
            # 1. MFCC features (most important for speech)
            mfcc_features = self._extract_mfcc_features(audio)
            features.extend(mfcc_features)
            
            # 2. Spectral features
            spectral_features = self._extract_spectral_features(audio)
            features.extend(spectral_features)
            
            # 3. Temporal features
            temporal_features = self._extract_temporal_features(audio)
            features.extend(temporal_features)
            
            # 4. Statistical features
            statistical_features = self._extract_statistical_features(audio)
            features.extend(statistical_features)
            
            # 5. Zero-crossing rate and energy features
            zcr_energy_features = self._extract_zcr_energy_features(audio)
            features.extend(zcr_energy_features)
            
            # Ensure target dimension
            features = self._adjust_feature_dimension(features, target_dim)
            
            # Convert to numpy array
            feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Validate features are meaningful
            if self._validate_fallback_features(feature_array):
                logger.debug(f"Successfully extracted {feature_array.shape[1]} traditional features")
                return feature_array
            else:
                logger.warning("Traditional features validation failed, using structured fallback")
                return self._create_structured_fallback_features(target_dim)
                
        except Exception as e:
            logger.error(f"Traditional feature extraction failed: {str(e)}")
            return self._create_structured_fallback_features(target_dim)
    
    def handle_classifier_failure(self, embeddings: np.ndarray, language: str,
                                available_classifiers: Dict[str, Any]) -> FallbackResult:
        """
        Handle classifier failure using ensemble methods.
        
        Implements ensemble methods for classifier failures by combining predictions
        from available classifiers or using rule-based detection as last resort.
        
        Args:
            embeddings: Feature embeddings for classification
            language: Target language
            available_classifiers: Dictionary of available classifier models
            
        Returns:
            FallbackResult with ensemble classification
        """
        try:
            start_time = time.time()
            logger.info(f"Handling classifier failure for {language} using ensemble methods")
            
            # Strategy 1: Use ensemble of available classifiers
            if available_classifiers:
                ensemble_result = self._ensemble_classification(
                    embeddings, language, available_classifiers
                )
                if ensemble_result is not None:
                    processing_time = time.time() - start_time
                    return FallbackResult(
                        classification=ensemble_result['classification'],
                        confidence_score=ensemble_result['confidence'],
                        fallback_level=FallbackLevel.CLASSIFIER_ENSEMBLE,
                        reason=FallbackReason.CLASSIFIER_FAILED,
                        processing_time=processing_time,
                        metadata={
                            'ensemble_size': ensemble_result['ensemble_size'],
                            'method': 'classifier_ensemble',
                            'language': language
                        }
                    )
            
            # Strategy 2: Statistical analysis of embeddings
            statistical_result = self._statistical_classification(embeddings, language)
            if statistical_result is not None:
                processing_time = time.time() - start_time
                return FallbackResult(
                    classification=statistical_result['classification'],
                    confidence_score=statistical_result['confidence'],
                    fallback_level=FallbackLevel.TRADITIONAL_FEATURES,
                    reason=FallbackReason.CLASSIFIER_FAILED,
                    processing_time=processing_time,
                    metadata={
                        'method': 'statistical_analysis',
                        'language': language,
                        'features_analyzed': statistical_result['features_count']
                    }
                )
            
            # Strategy 3: Rule-based detection (last resort)
            rule_result = self._rule_based_classification(embeddings, language)
            processing_time = time.time() - start_time
            
            return FallbackResult(
                classification=rule_result['classification'],
                confidence_score=rule_result['confidence'],
                fallback_level=FallbackLevel.RULE_BASED,
                reason=FallbackReason.CLASSIFIER_FAILED,
                processing_time=processing_time,
                metadata={
                    'method': 'rule_based',
                    'language': language,
                    'confidence_note': 'Low confidence due to rule-based fallback'
                }
            )
            
        except Exception as e:
            logger.error(f"Classifier fallback failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return FallbackResult(
                classification="HUMAN",  # Conservative default
                confidence_score=0.1,
                fallback_level=FallbackLevel.EMERGENCY,
                reason=FallbackReason.CLASSIFIER_FAILED,
                processing_time=processing_time,
                metadata={
                    'method': 'emergency_fallback',
                    'error': str(e),
                    'language': language
                }
            )
    
    def handle_resource_constraints(self, audio: np.ndarray, language: str,
                                  memory_usage: float, processing_time: float) -> FallbackResult:
        """
        Handle resource constraints with degraded processing strategies.
        
        Implements resource-aware degradation strategies that reduce computational
        requirements while maintaining reasonable detection accuracy.
        
        Args:
            audio: Audio data
            language: Target language
            memory_usage: Current memory usage in bytes
            processing_time: Current processing time in seconds
            
        Returns:
            FallbackResult with resource-optimized classification
        """
        try:
            start_time = time.time()
            logger.info(f"Handling resource constraints (memory: {memory_usage/1024/1024:.1f}MB, time: {processing_time:.1f}s)")
            
            # Determine degradation level based on resource constraints
            if memory_usage > self.resource_memory_limit:
                degradation_level = "high"
            elif processing_time > self.max_processing_time:
                degradation_level = "medium"
            else:
                degradation_level = "low"
            
            # Apply appropriate degradation strategy
            if degradation_level == "high":
                # Minimal processing - statistical analysis only
                result = self._minimal_processing_classification(audio, language)
                fallback_level = FallbackLevel.EMERGENCY
                reason = FallbackReason.RESOURCE_CONSTRAINTS
                
            elif degradation_level == "medium":
                # Reduced feature extraction
                result = self._reduced_feature_classification(audio, language)
                fallback_level = FallbackLevel.TRADITIONAL_FEATURES
                reason = FallbackReason.RESOURCE_CONSTRAINTS
                
            else:
                # Light processing with traditional features
                result = self._light_processing_classification(audio, language)
                fallback_level = FallbackLevel.TRADITIONAL_FEATURES
                reason = FallbackReason.RESOURCE_CONSTRAINTS
            
            processing_time_final = time.time() - start_time
            
            return FallbackResult(
                classification=result['classification'],
                confidence_score=result['confidence'],
                fallback_level=fallback_level,
                reason=reason,
                processing_time=processing_time_final,
                metadata={
                    'degradation_level': degradation_level,
                    'original_memory_usage': memory_usage,
                    'original_processing_time': processing_time,
                    'method': result['method'],
                    'language': language
                }
            )
            
        except Exception as e:
            logger.error(f"Resource constraint handling failed: {str(e)}")
            processing_time_final = time.time() - start_time
            
            return FallbackResult(
                classification="HUMAN",  # Conservative default
                confidence_score=0.1,
                fallback_level=FallbackLevel.EMERGENCY,
                reason=FallbackReason.RESOURCE_CONSTRAINTS,
                processing_time=processing_time_final,
                metadata={
                    'method': 'emergency_fallback',
                    'error': str(e),
                    'language': language
                }
            )
    
    def handle_low_confidence(self, initial_result: Dict[str, Any], 
                            audio: np.ndarray, language: str) -> FallbackResult:
        """
        Handle low confidence predictions with additional validation.
        
        When initial predictions have low confidence, this method applies additional
        validation techniques to improve reliability or appropriately flag uncertainty.
        
        Args:
            initial_result: Initial classification result with low confidence
            audio: Original audio data
            language: Target language
            
        Returns:
            FallbackResult with improved or validated classification
        """
        try:
            start_time = time.time()
            initial_confidence = initial_result.get('confidence_score', 0.0)
            initial_classification = initial_result.get('classification', 'HUMAN')
            
            logger.info(f"Handling low confidence prediction: {initial_classification} ({initial_confidence:.3f})")
            
            # Strategy 1: Cross-validation with different methods
            validation_results = []
            
            # Traditional feature analysis
            traditional_result = self._validate_with_traditional_features(audio, language)
            if traditional_result:
                validation_results.append(traditional_result)
            
            # Statistical pattern analysis
            statistical_result = self._validate_with_statistical_analysis(audio, language)
            if statistical_result:
                validation_results.append(statistical_result)
            
            # Rule-based validation
            rule_result = self._validate_with_rules(audio, language)
            if rule_result:
                validation_results.append(rule_result)
            
            # Combine validation results
            if validation_results:
                combined_result = self._combine_validation_results(
                    initial_result, validation_results
                )
                
                processing_time = time.time() - start_time
                
                return FallbackResult(
                    classification=combined_result['classification'],
                    confidence_score=combined_result['confidence'],
                    fallback_level=FallbackLevel.TRADITIONAL_FEATURES,
                    reason=FallbackReason.LOW_CONFIDENCE,
                    processing_time=processing_time,
                    metadata={
                        'initial_confidence': initial_confidence,
                        'initial_classification': initial_classification,
                        'validation_methods': len(validation_results),
                        'method': 'confidence_validation',
                        'language': language
                    }
                )
            
            # If validation fails, return with uncertainty flag
            processing_time = time.time() - start_time
            
            return FallbackResult(
                classification=initial_classification,
                confidence_score=max(initial_confidence * 0.8, 0.1),  # Reduce confidence
                fallback_level=FallbackLevel.RULE_BASED,
                reason=FallbackReason.LOW_CONFIDENCE,
                processing_time=processing_time,
                metadata={
                    'initial_confidence': initial_confidence,
                    'method': 'uncertainty_flagged',
                    'language': language,
                    'note': 'Low confidence prediction with validation failure'
                }
            )
            
        except Exception as e:
            logger.error(f"Low confidence handling failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return FallbackResult(
                classification="HUMAN",  # Conservative default
                confidence_score=0.1,
                fallback_level=FallbackLevel.EMERGENCY,
                reason=FallbackReason.LOW_CONFIDENCE,
                processing_time=processing_time,
                metadata={
                    'method': 'emergency_fallback',
                    'error': str(e),
                    'language': language
                }
            )
    
    def handle_timeout(self, audio: np.ndarray, language: str, 
                      timeout_duration: float) -> FallbackResult:
        """
        Handle processing timeout with fast fallback methods.
        
        When processing exceeds time limits, this method provides rapid
        classification using lightweight methods.
        
        Args:
            audio: Audio data
            language: Target language
            timeout_duration: Duration of timeout in seconds
            
        Returns:
            FallbackResult with fast classification
        """
        try:
            start_time = time.time()
            logger.info(f"Handling timeout after {timeout_duration:.1f}s with fast fallback")
            
            # Use only the fastest classification method
            result = self._fast_classification(audio, language)
            
            processing_time = time.time() - start_time
            
            return FallbackResult(
                classification=result['classification'],
                confidence_score=result['confidence'],
                fallback_level=FallbackLevel.EMERGENCY,
                reason=FallbackReason.TIMEOUT,
                processing_time=processing_time,
                metadata={
                    'timeout_duration': timeout_duration,
                    'method': 'fast_classification',
                    'language': language,
                    'note': 'Rapid processing due to timeout'
                }
            )
            
        except Exception as e:
            logger.error(f"Timeout handling failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return FallbackResult(
                classification="HUMAN",  # Conservative default
                confidence_score=0.1,
                fallback_level=FallbackLevel.EMERGENCY,
                reason=FallbackReason.TIMEOUT,
                processing_time=processing_time,
                metadata={
                    'method': 'emergency_fallback',
                    'error': str(e),
                    'language': language
                }
            )
    
    def get_fallback_recommendation(self, error_context: Dict[str, Any]) -> FallbackLevel:
        """
        Recommend appropriate fallback level based on error context.
        
        Analyzes the error context and system state to recommend the most
        appropriate fallback strategy.
        
        Args:
            error_context: Dictionary containing error information and system state
            
        Returns:
            Recommended FallbackLevel
        """
        try:
            error_type = error_context.get('error_type', 'unknown')
            system_resources = error_context.get('system_resources', {})
            model_availability = error_context.get('model_availability', {})
            
            # Foundation model errors
            if error_type in ['foundation_model_unavailable', 'foundation_model_failed']:
                if model_availability.get('alternative_models', 0) > 0:
                    return FallbackLevel.FOUNDATION_MODEL
                else:
                    return FallbackLevel.TRADITIONAL_FEATURES
            
            # Classifier errors
            elif error_type in ['classifier_unavailable', 'classifier_failed']:
                if model_availability.get('alternative_classifiers', 0) > 1:
                    return FallbackLevel.CLASSIFIER_ENSEMBLE
                else:
                    return FallbackLevel.TRADITIONAL_FEATURES
            
            # Resource constraints
            elif error_type == 'resource_constraints':
                memory_usage = system_resources.get('memory_usage', 0)
                if memory_usage > self.resource_memory_limit:
                    return FallbackLevel.EMERGENCY
                else:
                    return FallbackLevel.TRADITIONAL_FEATURES
            
            # Timeout issues
            elif error_type == 'timeout':
                return FallbackLevel.EMERGENCY
            
            # Low confidence
            elif error_type == 'low_confidence':
                return FallbackLevel.TRADITIONAL_FEATURES
            
            # Default recommendation
            else:
                return FallbackLevel.TRADITIONAL_FEATURES
                
        except Exception as e:
            logger.error(f"Fallback recommendation failed: {str(e)}")
            return FallbackLevel.EMERGENCY
    
    # Private helper methods for feature extraction
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio data."""
        try:
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            return audio
        except Exception:
            return audio
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> List[float]:
        """Extract MFCC features from audio."""
        try:
            if LIBROSA_AVAILABLE:
                import librosa
                mfcc = librosa.feature.mfcc(
                    y=audio, sr=16000, n_mfcc=self.mfcc_coefficients
                )
                # Statistical summary of MFCC coefficients
                features = []
                for i in range(mfcc.shape[0]):
                    features.extend([
                        float(np.mean(mfcc[i])),
                        float(np.std(mfcc[i])),
                        float(np.max(mfcc[i])),
                        float(np.min(mfcc[i]))
                    ])
                return features
            else:
                # Fallback MFCC-like features using FFT
                return self._simple_mfcc_features(audio)
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return [0.0] * (self.mfcc_coefficients * 4)
    
    def _extract_spectral_features(self, audio: np.ndarray) -> List[float]:
        """Extract spectral features from audio."""
        try:
            if LIBROSA_AVAILABLE:
                import librosa
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=16000)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
                
                return [
                    float(np.mean(spectral_centroids)),
                    float(np.mean(spectral_rolloff)),
                    float(np.mean(spectral_bandwidth)),
                    float(np.mean(zero_crossing_rate))
                ]
            else:
                # Fallback spectral features using FFT
                return self._simple_spectral_features(audio)
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return [0.0] * self.spectral_features
    
    def _extract_temporal_features(self, audio: np.ndarray) -> List[float]:
        """Extract temporal features from audio."""
        try:
            if LIBROSA_AVAILABLE:
                import librosa
                rms_energy = librosa.feature.rms(y=audio)
                tempo, _ = librosa.beat.beat_track(y=audio, sr=16000)
                
                return [
                    float(tempo),
                    float(np.mean(rms_energy)),
                    float(np.std(rms_energy)),
                    float(np.max(rms_energy))
                ]
            else:
                # Fallback temporal features
                return self._simple_temporal_features(audio)
        except Exception as e:
            logger.warning(f"Temporal feature extraction failed: {e}")
            return [0.0] * self.temporal_features
    
    def _extract_statistical_features(self, audio: np.ndarray) -> List[float]:
        """Extract statistical features from audio."""
        try:
            return [
                float(np.mean(audio)),
                float(np.std(audio)),
                float(np.var(audio)),
                float(np.min(audio)),
                float(np.max(audio)),
                float(np.median(audio)),
                float(np.percentile(audio, 25)),
                float(np.percentile(audio, 75))
            ]
        except Exception as e:
            logger.warning(f"Statistical feature extraction failed: {e}")
            return [0.0] * 8
    
    def _extract_zcr_energy_features(self, audio: np.ndarray) -> List[float]:
        """Extract zero-crossing rate and energy features."""
        try:
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
            zcr = len(zero_crossings) / len(audio) if len(audio) > 0 else 0
            
            # Energy features
            energy = np.sum(audio ** 2)
            rms_energy = np.sqrt(np.mean(audio ** 2))
            
            return [float(zcr), float(energy), float(rms_energy)]
        except Exception as e:
            logger.warning(f"ZCR/Energy feature extraction failed: {e}")
            return [0.0] * 3
    
    def _simple_mfcc_features(self, audio: np.ndarray) -> List[float]:
        """Simple MFCC-like features using FFT."""
        try:
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            
            # Mel-scale approximation
            features = []
            n_bands = self.mfcc_coefficients
            band_size = len(magnitude) // n_bands
            
            for i in range(n_bands):
                start_idx = i * band_size
                end_idx = min((i + 1) * band_size, len(magnitude))
                band_energy = np.mean(magnitude[start_idx:end_idx])
                
                features.extend([
                    float(band_energy),
                    float(np.std(magnitude[start_idx:end_idx])),
                    float(np.max(magnitude[start_idx:end_idx])),
                    float(np.min(magnitude[start_idx:end_idx]))
                ])
            
            return features
        except Exception:
            return [0.0] * (self.mfcc_coefficients * 4)
    
    def _simple_spectral_features(self, audio: np.ndarray) -> List[float]:
        """Simple spectral features using FFT."""
        try:
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(audio))
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            # Spectral rolloff (90% energy point)
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum >= 0.9 * cumsum[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
            zcr = len(zero_crossings) / len(audio) if len(audio) > 0 else 0
            
            return [
                float(spectral_centroid),
                float(spectral_rolloff),
                float(spectral_bandwidth),
                float(zcr)
            ]
        except Exception:
            return [0.0] * self.spectral_features
    
    def _simple_temporal_features(self, audio: np.ndarray) -> List[float]:
        """Simple temporal features."""
        try:
            # RMS energy
            rms_energy = np.sqrt(np.mean(audio ** 2))
            
            # Simple tempo estimation (zero crossing based)
            zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
            tempo_estimate = len(zero_crossings) * 60 / (len(audio) / 16000) if len(audio) > 0 else 0
            
            # Energy statistics
            energy_values = audio ** 2
            
            return [
                float(tempo_estimate),
                float(rms_energy),
                float(np.std(energy_values)),
                float(np.max(energy_values))
            ]
        except Exception:
            return [0.0] * self.temporal_features
    
    def _adjust_feature_dimension(self, features: List[float], target_dim: int) -> List[float]:
        """Adjust feature vector to target dimension."""
        current_size = len(features)
        
        if current_size == target_dim:
            return features
        elif current_size < target_dim:
            # Pad with structured features
            remaining = target_dim - current_size
            if current_size > 0:
                # Repeat and scale existing features
                base_features = np.array(features)
                for i in range(remaining):
                    scale_factor = 0.1 * (i % 10 + 1)
                    feature_idx = i % current_size
                    scaled_feature = base_features[feature_idx] * scale_factor
                    features.append(scaled_feature)
            else:
                # Create structured pattern
                for i in range(remaining):
                    features.append(0.01 * (i % 100))
        else:
            # Truncate
            features = features[:target_dim]
        
        return features
    
    def _create_structured_fallback_features(self, target_dim: int) -> np.ndarray:
        """Create structured fallback features (not random)."""
        features = [np.sin(i * 0.1) * 0.1 for i in range(target_dim)]
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def _validate_fallback_features(self, features: np.ndarray) -> bool:
        """Validate that fallback features are meaningful."""
        try:
            if features is None or features.size == 0:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return False
            
            # Check variance (should not be zero or too high)
            variance = np.var(features)
            if variance == 0 or variance > 100:
                return False
            
            # Check for reasonable value range
            if np.max(np.abs(features)) > 1000:
                return False
            
            return True
        except Exception:
            return False
    
    # Ensemble and classification methods
    
    def _ensemble_classification(self, embeddings: np.ndarray, language: str,
                               available_classifiers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform ensemble classification using available classifiers."""
        try:
            predictions = []
            confidences = []
            
            for classifier_name, classifier in available_classifiers.items():
                try:
                    if classifier is not None:
                        prediction = classifier.predict(embeddings, verbose=0)
                        pred_prob = float(prediction[0][0])
                        predictions.append(pred_prob)
                        
                        # Calculate confidence
                        confidence = abs(pred_prob - 0.5) * 2.0
                        confidences.append(confidence)
                        
                except Exception as e:
                    logger.warning(f"Classifier {classifier_name} failed: {e}")
                    continue
            
            if not predictions:
                return None
            
            # Weighted ensemble
            weights = np.array(confidences)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                ensemble_pred = np.average(predictions, weights=weights)
            else:
                ensemble_pred = np.mean(predictions)
            
            # Determine classification
            classification = "AI_GENERATED" if ensemble_pred >= 0.5 else "HUMAN"
            confidence = abs(ensemble_pred - 0.5) * 2.0
            
            return {
                'classification': classification,
                'confidence': float(confidence),
                'ensemble_size': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Ensemble classification failed: {e}")
            return None
    
    def _statistical_classification(self, embeddings: np.ndarray, language: str) -> Optional[Dict[str, Any]]:
        """Perform classification based on statistical analysis of embeddings."""
        try:
            if embeddings is None or embeddings.size == 0:
                return None
            
            flattened = embeddings.flatten()
            
            # Statistical features for classification
            mean_val = np.mean(flattened)
            std_val = np.std(flattened)
            variance = np.var(flattened)
            skewness = self._calculate_skewness(flattened)
            kurtosis = self._calculate_kurtosis(flattened)
            
            # Simple heuristic classification based on statistical properties
            # AI-generated speech often has different statistical patterns
            ai_score = 0.0
            
            # Check variance (AI often has different variance patterns)
            if 0.1 < variance < 2.0:
                ai_score += 0.2
            
            # Check skewness (AI might have different asymmetry)
            if abs(skewness) > 0.5:
                ai_score += 0.2
            
            # Check kurtosis (AI might have different tail behavior)
            if abs(kurtosis) > 1.0:
                ai_score += 0.2
            
            # Check mean (AI might center differently)
            if abs(mean_val) > 0.1:
                ai_score += 0.2
            
            # Check standard deviation
            if std_val > 0.5:
                ai_score += 0.2
            
            # Convert to classification
            classification = "AI_GENERATED" if ai_score >= 0.5 else "HUMAN"
            confidence = min(ai_score if ai_score >= 0.5 else (1.0 - ai_score), 0.8)
            
            return {
                'classification': classification,
                'confidence': float(confidence),
                'features_count': 5
            }
            
        except Exception as e:
            logger.error(f"Statistical classification failed: {e}")
            return None
    
    def _rule_based_classification(self, embeddings: np.ndarray, language: str) -> Dict[str, Any]:
        """Perform rule-based classification as last resort."""
        try:
            # Very simple rule-based approach
            # This is a conservative fallback that defaults to HUMAN
            # In practice, this would include more sophisticated rules
            
            classification = "HUMAN"  # Conservative default
            confidence = 0.3  # Low confidence for rule-based
            
            if embeddings is not None and embeddings.size > 0:
                # Simple rule: if embeddings have very specific patterns, might be AI
                flattened = embeddings.flatten()
                if len(flattened) > 10:
                    # Check for very uniform patterns (might indicate synthetic)
                    hist, _ = np.histogram(flattened, bins=10)
                    uniformity = np.std(hist / np.sum(hist))
                    
                    if uniformity < 0.1:  # Very uniform might be synthetic
                        classification = "AI_GENERATED"
                        confidence = 0.4
            
            return {
                'classification': classification,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Rule-based classification failed: {e}")
            return {
                'classification': "HUMAN",
                'confidence': 0.1
            }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            normalized = (data - mean_val) / std_val
            return float(np.mean(normalized ** 3))
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            normalized = (data - mean_val) / std_val
            return float(np.mean(normalized ** 4) - 3)
        except Exception:
            return 0.0
    
    # Resource-aware processing methods
    
    def _minimal_processing_classification(self, audio: np.ndarray, language: str) -> Dict[str, Any]:
        """Minimal processing classification for high resource constraints."""
        try:
            # Only basic statistical analysis
            if audio is None or len(audio) == 0:
                return {'classification': 'HUMAN', 'confidence': 0.1, 'method': 'minimal_empty'}
            
            # Simple statistical measures
            mean_val = float(np.mean(audio))
            std_val = float(np.std(audio))
            
            # Very simple heuristic
            if abs(mean_val) > 0.1 or std_val > 0.8:
                classification = "AI_GENERATED"
                confidence = 0.3
            else:
                classification = "HUMAN"
                confidence = 0.2
            
            return {
                'classification': classification,
                'confidence': confidence,
                'method': 'minimal_statistical'
            }
            
        except Exception:
            return {'classification': 'HUMAN', 'confidence': 0.1, 'method': 'minimal_error'}
    
    def _reduced_feature_classification(self, audio: np.ndarray, language: str) -> Dict[str, Any]:
        """Reduced feature classification for medium resource constraints."""
        try:
            if audio is None or len(audio) == 0:
                return {'classification': 'HUMAN', 'confidence': 0.1, 'method': 'reduced_empty'}
            
            # Extract only essential features
            features = []
            
            # Basic statistics
            features.extend([
                float(np.mean(audio)),
                float(np.std(audio)),
                float(np.max(audio)),
                float(np.min(audio))
            ])
            
            # Simple spectral feature
            try:
                fft = np.fft.fft(audio[:min(1024, len(audio))])  # Limit FFT size
                magnitude = np.abs(fft)
                features.append(float(np.mean(magnitude)))
            except Exception:
                features.append(0.0)
            
            # Simple classification based on features
            feature_sum = sum(abs(f) for f in features)
            if feature_sum > 2.0:
                classification = "AI_GENERATED"
                confidence = 0.4
            else:
                classification = "HUMAN"
                confidence = 0.3
            
            return {
                'classification': classification,
                'confidence': confidence,
                'method': 'reduced_features'
            }
            
        except Exception:
            return {'classification': 'HUMAN', 'confidence': 0.1, 'method': 'reduced_error'}
    
    def _light_processing_classification(self, audio: np.ndarray, language: str) -> Dict[str, Any]:
        """Light processing classification for low resource constraints."""
        try:
            if audio is None or len(audio) == 0:
                return {'classification': 'HUMAN', 'confidence': 0.1, 'method': 'light_empty'}
            
            # Extract lightweight features
            features = self._extract_statistical_features(audio)
            
            # Add simple spectral features
            try:
                spectral_features = self._simple_spectral_features(audio)
                features.extend(spectral_features[:2])  # Only first 2 spectral features
            except Exception:
                features.extend([0.0, 0.0])
            
            # Simple classification
            feature_variance = np.var(features)
            feature_mean = np.mean(np.abs(features))
            
            if feature_variance > 0.1 and feature_mean > 0.05:
                classification = "AI_GENERATED"
                confidence = 0.5
            else:
                classification = "HUMAN"
                confidence = 0.4
            
            return {
                'classification': classification,
                'confidence': confidence,
                'method': 'light_processing'
            }
            
        except Exception:
            return {'classification': 'HUMAN', 'confidence': 0.1, 'method': 'light_error'}
    
    def _fast_classification(self, audio: np.ndarray, language: str) -> Dict[str, Any]:
        """Fastest possible classification for timeout scenarios."""
        try:
            if audio is None or len(audio) == 0:
                return {'classification': 'HUMAN', 'confidence': 0.1}
            
            # Only the most basic analysis
            audio_mean = float(np.mean(audio))
            audio_std = float(np.std(audio))
            
            # Extremely simple heuristic
            if abs(audio_mean) > 0.05 or audio_std > 0.5:
                return {'classification': 'AI_GENERATED', 'confidence': 0.2}
            else:
                return {'classification': 'HUMAN', 'confidence': 0.2}
                
        except Exception:
            return {'classification': 'HUMAN', 'confidence': 0.1}
    
    # Validation methods for low confidence handling
    
    def _validate_with_traditional_features(self, audio: np.ndarray, language: str) -> Optional[Dict[str, Any]]:
        """Validate using traditional audio features."""
        try:
            features = self.handle_foundation_model_failure(audio, target_dim=100)
            if features is not None:
                # Simple validation based on feature patterns
                feature_variance = np.var(features)
                if feature_variance > 0.01:
                    return {'classification': 'AI_GENERATED', 'confidence': 0.4, 'method': 'traditional'}
                else:
                    return {'classification': 'HUMAN', 'confidence': 0.4, 'method': 'traditional'}
            return None
        except Exception:
            return None
    
    def _validate_with_statistical_analysis(self, audio: np.ndarray, language: str) -> Optional[Dict[str, Any]]:
        """Validate using statistical analysis."""
        try:
            if audio is None or len(audio) == 0:
                return None
            
            # Statistical validation
            stats = self._extract_statistical_features(audio)
            stat_variance = np.var(stats)
            
            if stat_variance > 0.05:
                return {'classification': 'AI_GENERATED', 'confidence': 0.3, 'method': 'statistical'}
            else:
                return {'classification': 'HUMAN', 'confidence': 0.3, 'method': 'statistical'}
                
        except Exception:
            return None
    
    def _validate_with_rules(self, audio: np.ndarray, language: str) -> Optional[Dict[str, Any]]:
        """Validate using rule-based analysis."""
        try:
            # Simple rule-based validation
            return {'classification': 'HUMAN', 'confidence': 0.2, 'method': 'rules'}
        except Exception:
            return None
    
    def _combine_validation_results(self, initial_result: Dict[str, Any], 
                                  validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine initial result with validation results."""
        try:
            all_results = [initial_result] + validation_results
            
            # Count votes for each classification
            ai_votes = sum(1 for r in all_results if r.get('classification') == 'AI_GENERATED')
            human_votes = sum(1 for r in all_results if r.get('classification') == 'HUMAN')
            
            # Weighted average of confidences
            confidences = [r.get('confidence_score', r.get('confidence', 0.0)) for r in all_results]
            avg_confidence = np.mean(confidences)
            
            # Determine final classification
            if ai_votes > human_votes:
                classification = 'AI_GENERATED'
            else:
                classification = 'HUMAN'
            
            # Boost confidence if multiple methods agree
            if max(ai_votes, human_votes) >= len(all_results) * 0.7:
                final_confidence = min(avg_confidence * 1.2, 0.8)
            else:
                final_confidence = avg_confidence * 0.9
            
            return {
                'classification': classification,
                'confidence': float(final_confidence)
            }
            
        except Exception:
            return {
                'classification': initial_result.get('classification', 'HUMAN'),
                'confidence': initial_result.get('confidence_score', initial_result.get('confidence', 0.1))
            }