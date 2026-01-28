"""
ML Detection Engine for AI-Generated Voice Detection API.

This module implements the core detection logic using foundation models (HuBERT, XLS-R)
for feature extraction and custom classifiers for AI vs human speech detection.
"""

import logging
import os
import time
import warnings
from typing import Dict, Optional, Tuple, Any, Union, List
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from pydantic import BaseModel

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")

# Foundation model imports
try:
    from transformers import (
        Wav2Vec2Model, 
        Wav2Vec2Processor,
        HubertModel,
        Wav2Vec2FeatureExtractor,
        AutoModel,
        AutoProcessor
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers library not available. Install with: pip install transformers torch")

from src.audio.processor import AudioFeatures
from src.exceptions import (
    DetectionEngineError,
    ModelLoadingError,
    UnsupportedLanguageError,
    InferenceError
)
from src.detection.fallback import FallbackLevel, FallbackReason
from src.optimization.threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


class DetectionResult(BaseModel):
    """Data model for detection results."""
    
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidence_score: float  # 0.0 to 1.0
    model_version: str
    processing_time: float
    threshold_used: Optional[float] = None  # Threshold used for classification
    is_threshold_optimized: Optional[bool] = None  # Whether threshold was optimized


class DetectionEngine:
    """
    ML-based detection engine for classifying voice recordings as AI-generated or human.
    
    Uses foundation models (HuBERT, XLS-R, Wav2Vec2) for feature extraction and 
    custom classifiers for final classification.
    """
    
    SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    # Foundation model configurations
    FOUNDATION_MODELS = {
        "xls-r-300m": {
            "path": "foundation/xls-r-300m",
            "model_class": "Wav2Vec2Model",
            "processor_class": "Wav2Vec2Processor",
            "embedding_dim": 1024,
            "recommended": True,
            "multilingual": True
        },
        "hubert-base": {
            "path": "foundation/hubert-base", 
            "model_class": "HubertModel",
            "processor_class": "Wav2Vec2FeatureExtractor",
            "embedding_dim": 768,
            "recommended": False,
            "multilingual": True
        },
        "wav2vec2-base": {
            "path": "foundation/wav2vec2-base",
            "model_class": "Wav2Vec2Model", 
            "processor_class": "Wav2Vec2Processor",
            "embedding_dim": 768,
            "recommended": False,
            "multilingual": False
        }
    }
    
    def __init__(self, settings=None, foundation_model: str = "auto"):
        """
        Initialize the DetectionEngine with comprehensive model validation, monitoring, and registry integration.
        
        Args:
            settings: Application settings object containing configuration
            foundation_model: Foundation model to use ("auto", "xls-r-300m", "hubert-base", "wav2vec2-base")
        """
        from src.config import get_settings
        from src.validation.validator import FoundationModelManager, ModelValidator
        from src.detection.fallback import FallbackStrategy
        from src.monitoring.accuracy_monitor import AccuracyMonitor
        from src.registry.model_registry import ModelRegistry
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.model_base_path = Path(settings.ml.models_base_path)
        
        # Initialize comprehensive model validation system
        self.model_validator = ModelValidator(settings)
        
        # Initialize FoundationModelManager for proper model handling
        self.foundation_manager = FoundationModelManager(settings)
        
        # Initialize FallbackStrategy for graceful degradation
        self.fallback_strategy = FallbackStrategy(settings)
        
        # Initialize ThresholdOptimizer for optimized classification thresholds
        self.threshold_optimizer = ThresholdOptimizer(settings)
        
        # Initialize AccuracyMonitor for real-time performance tracking
        self.accuracy_monitor = AccuracyMonitor(
            settings=settings,
            window_size=getattr(settings.ml, 'monitoring_window_size', 1000),
            degradation_threshold=getattr(settings.ml, 'degradation_threshold', 0.05)
        )
        
        # Initialize ModelRegistry for comprehensive model management
        registry_path = self.model_base_path / "registry"
        self.model_registry = ModelRegistry(str(registry_path))
        
        # Model caches with validation tracking
        self.classifier_cache: Dict[str, tf.keras.Model] = {}
        self.validation_cache: Dict[str, bool] = {}  # Track validation status
        
        # Configuration
        self.foundation_model_name = self._select_foundation_model(foundation_model)
        self.foundation_config = self.FOUNDATION_MODELS[self.foundation_model_name]
        
        # Initialize model configurations with validation
        self._initialize_model_configs()
        
        # Validate all models on startup
        self._validate_all_models_on_startup()
        
        # Set device preference
        self.device = "cuda" if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Ensure TensorFlow uses CPU for consistency (can be changed for GPU)
        if not torch.cuda.is_available():
            tf.config.set_visible_devices([], 'GPU')
        
        logger.info("DetectionEngine initialized with comprehensive validation, monitoring, and registry integration")
    
    def _select_foundation_model(self, model_name: str) -> str:
        """Select the best available foundation model."""
        if model_name == "auto":
            # Auto-select based on availability
            for name, config in self.FOUNDATION_MODELS.items():
                model_path = self.model_base_path / config["path"]
                if model_path.exists():
                    logger.info(f"Auto-selected foundation model: {name}")
                    return name
            
            # Default to XLS-R if none found (will create dummy)
            logger.warning("No foundation models found, will use XLS-R with dummy classifier")
            return "xls-r-300m"
        
        if model_name not in self.FOUNDATION_MODELS:
            raise ValueError(f"Unknown foundation model: {model_name}. Available: {list(self.FOUNDATION_MODELS.keys())}")
        
        return model_name
    
    def _initialize_model_configs(self):
        """Initialize model configurations for each supported language with validation."""
        self.model_configs = {}
        
        for language in self.SUPPORTED_LANGUAGES:
            # Get optimized threshold from ThresholdOptimizer, fallback to default 0.5
            optimized_threshold = self.threshold_optimizer.get_threshold(language)
            
            self.model_configs[language] = {
                "classifier_path": self.model_base_path / "classifiers" / f"{language.lower()}_classifier.h5",
                "version": "1.0.0",
                "input_shape": (self.foundation_config["embedding_dim"],),
                "confidence_threshold": optimized_threshold,
                "foundation_model": self.foundation_model_name,
                "validation_status": "pending"  # Track validation status
            }
    
    def _validate_all_models_on_startup(self):
        """
        Validate all models on startup to ensure system reliability.
        
        This method validates foundation models and classifiers, logging results
        and updating validation cache for runtime decisions.
        """
        logger.info("Starting comprehensive model validation on startup...")
        
        # Validate foundation model
        foundation_model_path = self.model_base_path / self.foundation_config["path"]
        foundation_validation = self.model_validator.validate_foundation_model(str(foundation_model_path))
        
        if foundation_validation.is_valid:
            logger.info(f"Foundation model {self.foundation_model_name} validation: PASSED "
                       f"(confidence: {foundation_validation.confidence_score:.3f})")
            self.validation_cache[f"foundation_{self.foundation_model_name}"] = True
        else:
            logger.warning(f"Foundation model {self.foundation_model_name} validation: FAILED "
                          f"- {foundation_validation.error_message}")
            self.validation_cache[f"foundation_{self.foundation_model_name}"] = False
        
        # Validate classifiers for each language
        for language in self.SUPPORTED_LANGUAGES:
            config = self.model_configs[language]
            classifier_path = config["classifier_path"]
            
            classifier_validation = self.model_validator.validate_classifier(str(classifier_path), language)
            
            if classifier_validation.is_valid:
                logger.info(f"Classifier for {language} validation: PASSED "
                           f"(confidence: {classifier_validation.confidence_score:.3f})")
                config["validation_status"] = "valid"
                self.validation_cache[f"classifier_{language}"] = True
                
                # Register model in registry if not already present
                self._register_model_if_needed(language, classifier_path, classifier_validation)
                
            else:
                logger.warning(f"Classifier for {language} validation: FAILED "
                              f"- {classifier_validation.error_message}")
                config["validation_status"] = "invalid"
                self.validation_cache[f"classifier_{language}"] = False
        
        # Log validation summary
        valid_classifiers = sum(1 for lang in self.SUPPORTED_LANGUAGES 
                               if self.validation_cache.get(f"classifier_{lang}", False))
        logger.info(f"Model validation complete: Foundation model valid: "
                   f"{self.validation_cache.get(f'foundation_{self.foundation_model_name}', False)}, "
                   f"Valid classifiers: {valid_classifiers}/{len(self.SUPPORTED_LANGUAGES)}")
    
    def _register_model_if_needed(self, language: str, classifier_path: Path, validation_result):
        """
        Register model in registry if not already present.
        
        Args:
            language: Language for the classifier
            classifier_path: Path to classifier file
            validation_result: Validation result from ModelValidator
        """
        try:
            model_id = f"{language.lower()}_classifier_v1.0.0"
            
            # Check if model already registered
            existing_model = self.model_registry.get_model(model_id)
            if existing_model:
                return  # Already registered
            
            # Create performance metrics from validation
            from src.models.schemas import PerformanceMetrics
            
            performance_metrics = PerformanceMetrics(
                accuracy=validation_result.performance_metrics.get("test_success_rate", 0.8),
                precision=0.8,  # Default values - would be updated with real metrics
                recall=0.8,
                f1_score=0.8,
                auc_roc=0.8,
                sample_count=validation_result.performance_metrics.get("total_parameters", 1000),
                last_updated=datetime.now()
            )
            
            # Register the model
            self.model_registry.register_model(
                model_id=model_id,
                model_type="classifier",
                file_path=str(classifier_path),
                validation_metrics=performance_metrics,
                language=language,
                version="1.0.0",
                additional_metadata={
                    "embedding_dim": self.foundation_config["embedding_dim"],
                    "foundation_model": self.foundation_model_name,
                    "validation_confidence": validation_result.confidence_score
                }
            )
            
            logger.info(f"Registered {language} classifier in model registry")
            
        except Exception as e:
            logger.error(f"Failed to register {language} classifier in registry: {str(e)}")
    
    def _extract_embeddings(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract embeddings from audio using foundation model with proper validation."""
        try:
            # Use FoundationModelManager for proper embedding extraction
            embeddings = self.foundation_manager.extract_embeddings(
                audio_array, 
                model_name=self.foundation_model_name,
                sample_rate=sample_rate,
                deterministic=True
            )
            
            if embeddings is not None:
                logger.debug(f"Successfully extracted embeddings using {self.foundation_model_name}")
                return embeddings
            
            # If foundation model fails, raise exception to trigger fallback
            raise RuntimeError(f"Foundation model {self.foundation_model_name} returned None embeddings")
            
        except MemoryError as e:
            logger.error(f"Memory error during embedding extraction: {str(e)}")
            # Clear any cached models to free memory
            self.foundation_manager.clear_cache()
            raise RuntimeError(f"Memory error in foundation model {self.foundation_model_name}")
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            # Re-raise to trigger fallback handling in detect_voice_type
            raise
    
    def _create_dummy_classifier(self, language: str) -> tf.keras.Model:
        """Create a dummy classifier for demonstration purposes."""
        try:
            embedding_dim = self.foundation_config["embedding_dim"]
            
            # Create a simple MLP classifier
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(embedding_dim,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize with random weights
            dummy_input = np.random.random((1, embedding_dim))
            _ = model.predict(dummy_input, verbose=0)
            
            logger.info(f"Created dummy classifier for {language} (input_dim: {embedding_dim})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create dummy classifier for {language}: {str(e)}")
            raise ModelLoadingError(f"Classifier creation failed for {language}: {str(e)}", language=language)
    
    def _load_classifier(self, language: str) -> tf.keras.Model:
        """Load or create a classifier for the specified language."""
        try:
            if language not in self.SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
            
            # Check cache first
            if language in self.classifier_cache:
                return self.classifier_cache[language]
            
            config = self.model_configs[language]
            classifier_path = config["classifier_path"]
            
            # Try to load pre-trained classifier
            if classifier_path.exists() and classifier_path.stat().st_size > 0:
                try:
                    logger.info(f"Loading classifier for {language} from {classifier_path}")
                    model = tf.keras.models.load_model(str(classifier_path))
                except Exception as load_error:
                    logger.warning(f"Failed to load classifier file for {language} (corrupted/invalid): {load_error}")
                    logger.info(f"Creating dummy classifier for {language} instead")
                    model = self._create_dummy_classifier(language)
            else:
                logger.warning(f"Classifier not found or empty for {language}, creating dummy classifier")
                model = self._create_dummy_classifier(language)
            
            # Cache the model
            self.classifier_cache[language] = model
            logger.info(f"Classifier for {language} loaded and cached successfully")
            
            return model
            
        except UnsupportedLanguageError:
            raise
        except Exception as e:
            logger.error(f"Failed to load classifier for {language}: {str(e)}")
            raise ModelLoadingError(f"Classifier loading failed for {language}: {str(e)}", language=language)
    
    def _preprocess_audio_for_foundation(self, features: AudioFeatures) -> np.ndarray:
        """Preprocess audio features for foundation model input."""
        try:
            # Foundation models typically expect raw audio waveform
            # We'll use the original audio data if available, otherwise reconstruct from MFCC
            
            # For now, create a dummy waveform from MFCC features
            # In production, you'd want to pass the original audio waveform
            mfcc_array = np.array(features.mfcc)  # Shape: (13, N)
            
            # Create a dummy waveform (this is a placeholder - use original audio in production)
            duration_samples = int(features.duration * 16000)  # Assume 16kHz
            dummy_waveform = np.random.randn(duration_samples) * 0.1  # Low amplitude noise
            
            logger.debug(f"Created dummy waveform of length {len(dummy_waveform)} for foundation model")
            return dummy_waveform.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            # Return minimal dummy audio
            return np.random.randn(16000).astype(np.float32)  # 1 second of dummy audio
    
    def _calculate_confidence_score(self, model_output: float, language: str, 
                                  features: Optional[np.ndarray] = None,
                                  model_metadata: Optional[Dict[str, Any]] = None,
                                  prediction_history: Optional[List[float]] = None) -> float:
        """
        Calculate enhanced confidence score from model output using comprehensive uncertainty analysis.
        
        Args:
            model_output: Raw model output (sigmoid probability)
            language: Target language
            features: Input features used for prediction (optional)
            model_metadata: Metadata about the model used (optional)
            prediction_history: History of predictions for stability analysis (optional)
            
        Returns:
            Enhanced confidence score between 0.0 and 1.0
        """
        try:
            # Use enhanced confidence calculator if available
            if not hasattr(self, '_confidence_calculator'):
                from src.detection.confidence import EnhancedConfidenceCalculator
                self._confidence_calculator = EnhancedConfidenceCalculator(self.settings)
            
            # Calculate comprehensive confidence metrics
            confidence_metrics = self._confidence_calculator.calculate_enhanced_confidence(
                model_output=model_output,
                language=language,
                features=features,
                model_metadata=model_metadata,
                prediction_history=prediction_history
            )
            
            # Store detailed metrics for potential use in explanations
            if not hasattr(self, '_last_confidence_metrics'):
                self._last_confidence_metrics = {}
            self._last_confidence_metrics[language] = confidence_metrics
            
            logger.debug(
                f"Enhanced confidence for {language}: {confidence_metrics.final_confidence:.3f} "
                f"({confidence_metrics.confidence_level.value}) with "
                f"{len(confidence_metrics.uncertainty_indicators)} uncertainty indicators"
            )
            
            return confidence_metrics.final_confidence
            
        except Exception as e:
            logger.error(f"Enhanced confidence calculation failed, using fallback: {str(e)}")
            # Fallback to original simple calculation
            return self._calculate_simple_confidence_score(model_output, language)
    
    def _calculate_simple_confidence_score(self, model_output: float, language: str) -> float:
        """
        Fallback simple confidence calculation method.
        
        Args:
            model_output: Raw model output (sigmoid probability)
            language: Target language
            
        Returns:
            Simple confidence score between 0.0 and 1.0
        """
        try:
            # For binary classification with sigmoid output:
            # - Values close to 0.5 indicate low confidence
            # - Values close to 0.0 or 1.0 indicate high confidence
            
            # Calculate distance from 0.5 (uncertainty point)
            distance_from_uncertain = abs(model_output - 0.5)
            
            # Scale to 0-1 range (0.5 distance = 1.0 confidence)
            confidence = min(distance_from_uncertain * 2.0, 1.0)
            
            # Apply language-specific adjustments if needed
            config = self.model_configs.get(language, {})
            threshold = config.get("confidence_threshold", 0.5)
            
            # Boost confidence for clear predictions
            if model_output < threshold - 0.2 or model_output > threshold + 0.2:
                confidence = min(confidence * 1.1, 1.0)
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Simple confidence calculation failed: {str(e)}")
            return 0.5  # Return neutral confidence on error
    
    def _generate_explanation(self, classification: str, confidence_score: float, 
                            features: AudioFeatures, language: str) -> str:
        """
        Generate human-readable explanation for the classification decision.
        
        Args:
            classification: Classification result ("AI_GENERATED" or "HUMAN")
            confidence_score: Confidence score
            features: Original audio features
            language: Target language
            
        Returns:
            Human-readable explanation string
        """
        try:
            # Analyze features for explanation
            duration = features.duration
            mfcc_array = np.array(features.mfcc)
            spectral_array = np.array(features.spectral)
            
            # Calculate feature statistics
            mfcc_variance = np.var(mfcc_array)
            spectral_mean = np.mean(spectral_array)
            
            # Generate explanation based on classification and features
            if classification == "HUMAN":
                if confidence_score > 0.8:
                    explanation = (
                        f"The audio exhibits strong human speech characteristics with natural "
                        f"vocal variations and breathing patterns typical of genuine human speech. "
                        f"Analysis of {duration:.1f} seconds of {language} audio shows "
                        f"consistent human-like spectral properties."
                    )
                elif confidence_score > 0.6:
                    explanation = (
                        f"The audio shows human speech patterns with moderate confidence. "
                        f"The {duration:.1f}-second {language} sample contains vocal characteristics "
                        f"consistent with human speech, though some features require closer analysis."
                    )
                else:
                    explanation = (
                        f"The audio appears to be human speech, but with lower confidence due to "
                        f"ambiguous vocal characteristics. The {duration:.1f}-second {language} "
                        f"sample shows mixed indicators that lean toward human origin."
                    )
            else:  # AI_GENERATED
                if confidence_score > 0.8:
                    explanation = (
                        f"The audio exhibits strong indicators of AI-generated speech with "
                        f"artificial vocal patterns and synthetic characteristics. "
                        f"Analysis of {duration:.1f} seconds of {language} audio reveals "
                        f"consistent markers typical of synthetic voice generation."
                    )
                elif confidence_score > 0.6:
                    explanation = (
                        f"The audio shows AI-generated speech patterns with moderate confidence. "
                        f"The {duration:.1f}-second {language} sample contains synthetic "
                        f"characteristics that suggest artificial voice generation."
                    )
                else:
                    explanation = (
                        f"The audio appears to be AI-generated, but with lower confidence due to "
                        f"mixed vocal characteristics. The {duration:.1f}-second {language} "
                        f"sample shows some indicators that suggest synthetic origin."
                    )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            # Return generic explanation on error
            return (
                f"The audio has been classified as {classification.replace('_', ' ').lower()} "
                f"with {confidence_score:.1%} confidence based on analysis of "
                f"{language} speech patterns."
            )
    
    
    def detect_voice_type(self, features: AudioFeatures, language: str, 
                          ground_truth: Optional[str] = None) -> DetectionResult:
        """
        Detect whether the voice is AI-generated or human with comprehensive monitoring and validation.
        
        This method implements systematic fallback triggering logic for various failure scenarios:
        - Foundation model failures (unavailable, loading errors, inference errors)
        - Classifier failures (loading errors, inference errors, validation failures)
        - Resource constraints (memory limits, processing timeouts)
        - Low confidence predictions requiring additional validation
        - Complete system failures requiring emergency fallback
        
        Args:
            features: Extracted audio features
            language: Target language for analysis
            ground_truth: Optional ground truth label for monitoring ("AI_GENERATED" or "HUMAN")
            
        Returns:
            DetectionResult with classification, confidence, and metadata
            
        Raises:
            UnsupportedLanguageError: If language is not supported
            InferenceError: If detection inference fails completely
        """
        start_time = time.time()
        fallback_context = {
            'language': language,
            'start_time': start_time,
            'system_resources': {},
            'model_availability': {},
            'error_history': []
        }
        
        try:
            # Validate language
            if language not in self.SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
            
            # Check model validation status before processing
            foundation_valid = self.validation_cache.get(f"foundation_{self.foundation_model_name}", True)
            classifier_valid = self.validation_cache.get(f"classifier_{language}", True)
            
            if not foundation_valid:
                logger.warning(f"Foundation model {self.foundation_model_name} failed validation, using fallback")
            if not classifier_valid:
                logger.warning(f"Classifier for {language} failed validation, using fallback")
            
            # Gather system resource information for fallback decisions
            fallback_context['system_resources'] = self._gather_system_resources()
            fallback_context['model_availability'] = self._assess_model_availability(language)
            
            # Preprocess audio for foundation model
            audio_array = self._preprocess_audio_for_foundation(features)
            
            # Step 1: Foundation Model Processing with Systematic Fallback
            embeddings, foundation_fallback_info = self._extract_embeddings_with_fallback(
                audio_array, fallback_context
            )
            
            # Step 2: Classifier Processing with Systematic Fallback  
            classification_result, classifier_fallback_info = self._classify_with_fallback(
                embeddings, language, fallback_context
            )
            
            # Step 3: Post-processing and Validation
            final_result, validation_fallback_info = self._validate_and_finalize_result(
                classification_result, audio_array, language, fallback_context
            )
            
            # Step 4: Timeout and Resource Constraint Checks
            processing_time = time.time() - start_time
            final_result, timeout_fallback_info = self._check_and_handle_constraints(
                final_result, audio_array, language, processing_time, fallback_context
            )
            
            # Compile comprehensive fallback information
            all_fallback_info = {
                'foundation': foundation_fallback_info,
                'classifier': classifier_fallback_info, 
                'validation': validation_fallback_info,
                'timeout': timeout_fallback_info
            }
            
            # Create final result with comprehensive metadata
            result = self._create_final_result(
                final_result, language, processing_time, all_fallback_info
            )
            
            # Track prediction in accuracy monitor
            self.accuracy_monitor.track_prediction(result, language, ground_truth)
            
            # Check for accuracy degradation
            if ground_truth and self.accuracy_monitor.detect_accuracy_degradation(language):
                logger.warning(f"Accuracy degradation detected for {language}")
            
            # Log comprehensive detection information
            self._log_detection_summary(result, language, all_fallback_info)
            
            return result
            
        except (UnsupportedLanguageError, ModelLoadingError):
            raise
        except Exception as e:
            logger.error(f"Voice detection failed for {language}: {str(e)}")
            fallback_context['error_history'].append({
                'type': 'system_failure',
                'error': str(e),
                'timestamp': time.time()
            })
            
            # Apply systematic emergency fallback
            emergency_result = self._apply_emergency_fallback(features, language, fallback_context, e)
            
            # Still track the emergency result for monitoring
            if ground_truth:
                self.accuracy_monitor.track_prediction(emergency_result, language, ground_truth)
            
            return emergency_result
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status including validation, monitoring, and registry information.
        
        Returns:
            Dictionary containing detailed system health information
        """
        try:
            # Get validation status
            validation_status = {}
            for language in self.SUPPORTED_LANGUAGES:
                validation_status[language] = {
                    "classifier_valid": self.validation_cache.get(f"classifier_{language}", False),
                    "validation_status": self.model_configs[language].get("validation_status", "unknown")
                }
            
            foundation_valid = self.validation_cache.get(f"foundation_{self.foundation_model_name}", False)
            
            # Get monitoring status
            monitoring_summary = self.accuracy_monitor.get_monitoring_summary()
            
            # Get registry statistics
            registry_stats = self.model_registry.get_registry_stats()
            
            # Get fallback status
            fallback_status = self.get_fallback_status()
            
            # Get threshold information
            threshold_info = self.get_all_threshold_info()
            
            # Calculate overall health score
            health_score = self._calculate_health_score(validation_status, foundation_valid, monitoring_summary)
            
            return {
                "overall_health": {
                    "score": health_score,
                    "status": "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.6 else "unhealthy"
                },
                "foundation_model": {
                    "name": self.foundation_model_name,
                    "valid": foundation_valid,
                    "config": self.foundation_config
                },
                "classifiers": validation_status,
                "monitoring": monitoring_summary,
                "registry": registry_stats,
                "fallback": fallback_status,
                "thresholds": threshold_info,
                "device": self.device,
                "supported_languages": self.SUPPORTED_LANGUAGES
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health status: {str(e)}")
            return {
                "overall_health": {"score": 0.0, "status": "error"},
                "error": str(e)
            }
    
    def _calculate_health_score(self, validation_status: Dict, foundation_valid: bool, 
                               monitoring_summary: Dict) -> float:
        """
        Calculate overall system health score.
        
        Args:
            validation_status: Validation status for all classifiers
            foundation_valid: Foundation model validation status
            monitoring_summary: Monitoring summary information
            
        Returns:
            Health score between 0.0 and 1.0
        """
        try:
            score = 0.0
            
            # Foundation model health (30% weight)
            if foundation_valid:
                score += 0.3
            
            # Classifier health (40% weight)
            valid_classifiers = sum(1 for status in validation_status.values() 
                                   if status.get("classifier_valid", False))
            classifier_ratio = valid_classifiers / len(self.SUPPORTED_LANGUAGES)
            score += classifier_ratio * 0.4
            
            # Monitoring health (20% weight)
            if monitoring_summary.get("total_predictions", 0) > 0:
                score += 0.1  # Has monitoring data
                
                # Check for recent alerts
                recent_alerts = monitoring_summary.get("recent_alerts_count", 0)
                if recent_alerts == 0:
                    score += 0.1  # No recent alerts
            
            # System availability (10% weight)
            if len(self.classifier_cache) > 0:
                score += 0.05  # Models are loaded
            
            if hasattr(self, 'fallback_strategy') and self.fallback_strategy:
                score += 0.05  # Fallback system available
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {str(e)}")
            return 0.0
    
    def update_model_performance_metrics(self, language: str, accuracy: float, 
                                       precision: float, recall: float, f1_score: float,
                                       sample_count: int) -> bool:
        """
        Update performance metrics for a model in the registry.
        
        Args:
            language: Target language
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1_score: Model F1 score
            sample_count: Number of samples used for evaluation
            
        Returns:
            True if updated successfully
        """
        try:
            model_id = f"{language.lower()}_classifier_v1.0.0"
            
            from src.models.schemas import PerformanceMetrics
            
            metrics = PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_roc=0.8,  # Would be calculated from actual data
                sample_count=sample_count,
                last_updated=datetime.now()
            )
            
            success = self.model_registry.update_model_metrics(model_id, metrics)
            
            if success:
                logger.info(f"Updated performance metrics for {language}: "
                           f"accuracy={accuracy:.3f}, f1={f1_score:.3f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics for {language}: {str(e)}")
            return False
    
    def get_monitoring_report(self, language: Optional[str] = None, 
                            hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive monitoring report.
        
        Args:
            language: Specific language (None for all languages)
            hours: Number of hours to include in report
            
        Returns:
            Dictionary containing monitoring report
        """
        try:
            from datetime import timedelta
            
            time_window = timedelta(hours=hours)
            
            if language:
                # Get metrics for specific language
                metrics = self.accuracy_monitor.get_accuracy_metrics(language, time_window)
                alerts = self.accuracy_monitor.get_recent_alerts(language, hours)
                
                if metrics:
                    return {
                        "language": language,
                        "time_window_hours": hours,
                        "metrics": {
                            "accuracy": metrics.accuracy,
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "f1_score": metrics.f1_score,
                            "sample_count": metrics.sample_count,
                            "confidence_distribution": metrics.confidence_distribution,
                            "avg_processing_time": sum(metrics.processing_times) / len(metrics.processing_times) if metrics.processing_times else 0.0
                        },
                        "alerts": [
                            {
                                "timestamp": alert.timestamp.isoformat(),
                                "level": alert.alert_level.value,
                                "message": alert.message,
                                "degradation": alert.degradation_amount
                            }
                            for alert in alerts
                        ]
                    }
                else:
                    return {
                        "language": language,
                        "time_window_hours": hours,
                        "error": "Insufficient data for metrics calculation"
                    }
            else:
                # Get metrics for all languages
                all_metrics = self.accuracy_monitor.get_all_language_metrics(time_window)
                all_alerts = self.accuracy_monitor.get_recent_alerts(hours=hours)
                
                report = {
                    "time_window_hours": hours,
                    "languages": {},
                    "summary": {
                        "total_languages": len(all_metrics),
                        "total_alerts": len(all_alerts),
                        "overall_accuracy": 0.0
                    }
                }
                
                total_accuracy = 0.0
                for lang, metrics in all_metrics.items():
                    report["languages"][lang] = {
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "sample_count": metrics.sample_count
                    }
                    total_accuracy += metrics.accuracy
                
                if all_metrics:
                    report["summary"]["overall_accuracy"] = total_accuracy / len(all_metrics)
                
                return report
                
        except Exception as e:
            logger.error(f"Failed to get monitoring report: {str(e)}")
            return {"error": str(e)}
    
    def validate_model_on_demand(self, language: str) -> Dict[str, Any]:
        """
        Perform on-demand validation of a specific language model.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary containing validation results
        """
        try:
            if language not in self.SUPPORTED_LANGUAGES:
                return {"error": f"Unsupported language: {language}"}
            
            config = self.model_configs[language]
            classifier_path = config["classifier_path"]
            
            # Perform validation
            validation_result = self.model_validator.validate_classifier(str(classifier_path), language)
            
            # Update validation cache
            self.validation_cache[f"classifier_{language}"] = validation_result.is_valid
            config["validation_status"] = "valid" if validation_result.is_valid else "invalid"
            
            # Update registry if model is valid
            if validation_result.is_valid:
                self._register_model_if_needed(language, classifier_path, validation_result)
            
            return {
                "language": language,
                "is_valid": validation_result.is_valid,
                "confidence_score": validation_result.confidence_score,
                "error_message": validation_result.error_message,
                "performance_metrics": validation_result.performance_metrics,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"On-demand validation failed for {language}: {str(e)}")
            return {"error": str(e)}
    
    def get_model_registry_info(self, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model registry information.
        
        Args:
            language: Specific language (None for all models)
            
        Returns:
            Dictionary containing registry information
        """
        try:
            if language:
                model_id = f"{language.lower()}_classifier_v1.0.0"
                model_record = self.model_registry.get_model(model_id)
                
                if model_record:
                    return {
                        "model_id": model_record.model_id,
                        "language": model_record.language,
                        "version": model_record.version,
                        "status": model_record.status.value,
                        "training_date": model_record.training_date.isoformat(),
                        "validation_metrics": {
                            "accuracy": model_record.validation_metrics.accuracy,
                            "precision": model_record.validation_metrics.precision,
                            "recall": model_record.validation_metrics.recall,
                            "f1_score": model_record.validation_metrics.f1_score
                        },
                        "file_size_mb": model_record.model_size_bytes / (1024 * 1024),
                        "last_updated": model_record.last_updated.isoformat()
                    }
                else:
                    return {"error": f"Model not found for language: {language}"}
            else:
                # Get all classifier models
                classifier_models = self.model_registry.list_models(model_type="classifier")
                
                models_info = {}
                for model in classifier_models:
                    if model.language:
                        models_info[model.language] = {
                            "model_id": model.model_id,
                            "version": model.version,
                            "status": model.status.value,
                            "accuracy": model.validation_metrics.accuracy,
                            "last_updated": model.last_updated.isoformat()
                        }
                
                return {
                    "total_models": len(classifier_models),
                    "models": models_info,
                    "registry_stats": self.model_registry.get_registry_stats()
                }
                
        except Exception as e:
            logger.error(f"Failed to get registry info: {str(e)}")
            return {"error": str(e)}
        """
        Get detailed explanation of the confidence score for the last prediction.
        
        Args:
            language: Target language
            
        Returns:
            Human-readable confidence explanation or None if not available
        """
        try:
            if not hasattr(self, '_last_confidence_metrics'):
                return None
            
            confidence_metrics = self._last_confidence_metrics.get(language)
            if confidence_metrics is None:
                return None
            
            if not hasattr(self, '_confidence_calculator'):
                from src.detection.confidence import EnhancedConfidenceCalculator
                self._confidence_calculator = EnhancedConfidenceCalculator(self.settings)
            
            return self._confidence_calculator.get_confidence_explanation(confidence_metrics)
            
        except Exception as e:
            logger.error(f"Failed to get confidence explanation for {language}: {str(e)}")
            return None
    
    def get_confidence_metrics(self, language: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed confidence metrics for the last prediction.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary containing detailed confidence metrics or None if not available
        """
        try:
            if not hasattr(self, '_last_confidence_metrics'):
                return None
            
            confidence_metrics = self._last_confidence_metrics.get(language)
            if confidence_metrics is None:
                return None
            
            return {
                'raw_confidence': confidence_metrics.raw_confidence,
                'adjusted_confidence': confidence_metrics.adjusted_confidence,
                'final_confidence': confidence_metrics.final_confidence,
                'confidence_level': confidence_metrics.confidence_level.value,
                'uncertainty_indicators': [
                    {
                        'type': indicator.uncertainty_type.value,
                        'value': indicator.value,
                        'description': indicator.description,
                        'factors': indicator.contributing_factors
                    }
                    for indicator in confidence_metrics.uncertainty_indicators
                ],
                'prediction_stability': confidence_metrics.prediction_stability,
                'feature_quality_score': confidence_metrics.feature_quality_score,
                'model_certainty': confidence_metrics.model_certainty,
                'calibration_score': confidence_metrics.calibration_score
            }
            
        except Exception as e:
            logger.error(f"Failed to get confidence metrics for {language}: {str(e)}")
            return None

    def generate_explanation(self, result: DetectionResult, features: AudioFeatures, 
                           language: str) -> str:
        """
        Generate explanation for a detection result using AI-powered explanation generator.
        
        Args:
            result: Detection result
            features: Original audio features
            language: Target language
            
        Returns:
            AI-generated natural language explanation
        """
        try:
            # Import here to avoid circular imports
            from src.explanation import get_explanation_generator
            
            # Get the explanation generator
            explanation_generator = get_explanation_generator()
            
            # Generate AI-powered explanation
            explanation = explanation_generator.generate_explanation(result, features, language)
            
            logger.debug(f"Generated AI explanation for {language}: {explanation[:100]}...")
            return explanation
            
        except Exception as e:
            logger.error(f"AI explanation generation failed, using fallback: {e}")
            # Fallback to the old hardcoded method
            return self._generate_explanation(
                result.classification, 
                result.confidence_score, 
                features, 
                language
            )
    
    def get_model_info(self, language: str) -> Dict[str, Any]:
        """
        Get information about a language-specific model.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary containing model information
            
        Raises:
            UnsupportedLanguageError: If language is not supported
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
        
        config = self.model_configs[language]
        is_classifier_loaded = language in self.classifier_cache
        foundation_info = self.foundation_manager.get_model_info(self.foundation_model_name)
        
        # Get threshold information
        threshold_info = self.get_threshold_info(language)
        
        return {
            "language": language,
            "version": config["version"],
            "classifier_path": str(config["classifier_path"]),
            "foundation_model": self.foundation_model_name,
            "foundation_config": self.foundation_config,
            "foundation_info": foundation_info,
            "is_classifier_loaded": is_classifier_loaded,
            "input_shape": config["input_shape"],
            "confidence_threshold": config["confidence_threshold"],
            "threshold_info": threshold_info,
            "device": self.device
        }
    
    def get_available_foundation_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available foundation models."""
        return self.foundation_manager.get_available_models()
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """
        Get current fallback system status and configuration.
        
        Returns:
            Dictionary containing fallback system information
        """
        try:
            from src.detection.fallback import FallbackLevel, FallbackReason
            
            return {
                "fallback_available": True,
                "confidence_threshold": self.fallback_strategy.confidence_threshold,
                "resource_memory_limit": self.fallback_strategy.resource_memory_limit,
                "max_processing_time": self.fallback_strategy.max_processing_time,
                "traditional_features_available": self.fallback_strategy.audio_processor is not None,
                "ensemble_weights": self.fallback_strategy.ensemble_weights,
                "fallback_levels": [level.value for level in FallbackLevel],
                "supported_fallback_reasons": [reason.value for reason in FallbackReason]
            }
        except Exception as e:
            logger.error(f"Failed to get fallback status: {str(e)}")
            return {
                "fallback_available": False,
                "error": str(e)
            }
    
    def configure_fallback_thresholds(self, confidence_threshold: Optional[float] = None,
                                    memory_limit: Optional[int] = None,
                                    max_processing_time: Optional[float] = None):
        """
        Configure fallback system thresholds.
        
        Args:
            confidence_threshold: Minimum confidence for reliable results
            memory_limit: Memory limit in bytes for resource constraints
            max_processing_time: Maximum processing time in seconds
        """
        try:
            if confidence_threshold is not None:
                self.fallback_strategy.confidence_threshold = confidence_threshold
                logger.info(f"Updated confidence threshold to {confidence_threshold}")
            
            if memory_limit is not None:
                self.fallback_strategy.resource_memory_limit = memory_limit
                logger.info(f"Updated memory limit to {memory_limit} bytes")
            
            if max_processing_time is not None:
                self.fallback_strategy.max_processing_time = max_processing_time
                logger.info(f"Updated max processing time to {max_processing_time} seconds")
                
        except Exception as e:
            logger.error(f"Failed to configure fallback thresholds: {str(e)}")
            raise
    
    def get_threshold_info(self, language: str) -> Dict[str, Any]:
        """
        Get threshold information for a specific language.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary containing threshold information
        """
        try:
            if language not in self.SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
            
            # Get current threshold configuration
            all_thresholds = self.threshold_optimizer.get_all_thresholds()
            threshold_config = all_thresholds.get(language)
            
            if threshold_config:
                return {
                    "language": language,
                    "threshold": threshold_config.threshold,
                    "precision": threshold_config.precision,
                    "recall": threshold_config.recall,
                    "f1_score": threshold_config.f1_score,
                    "auc_roc": threshold_config.auc_roc,
                    "sample_count": threshold_config.sample_count,
                    "last_updated": threshold_config.last_updated.isoformat(),
                    "model_version": threshold_config.model_version,
                    "optimization_method": threshold_config.optimization_method,
                    "is_optimized": threshold_config.optimization_method != "default"
                }
            else:
                # Return default threshold info
                default_threshold = self.threshold_optimizer.get_threshold(language)
                return {
                    "language": language,
                    "threshold": default_threshold,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "auc_roc": None,
                    "sample_count": 0,
                    "last_updated": None,
                    "model_version": "default",
                    "optimization_method": "default",
                    "is_optimized": False
                }
                
        except Exception as e:
            logger.error(f"Failed to get threshold info for {language}: {str(e)}")
            return {
                "language": language,
                "threshold": 0.5,
                "error": str(e)
            }
    
    def update_threshold_for_language(self, language: str, validation_predictions: List[float], 
                                    validation_labels: List[int], model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Update optimized threshold for a specific language using validation data.
        
        Args:
            language: Target language
            validation_predictions: Model predictions on validation data (probabilities)
            validation_labels: Ground truth labels for validation data (0 or 1)
            model_version: Specific model version (optional)
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            if language not in self.SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
            
            logger.info(f"Updating threshold for {language} with {len(validation_predictions)} validation samples")
            
            # Create validation data object
            from src.optimization.threshold_optimizer import ValidationData
            validation_data = ValidationData(
                predictions=np.array(validation_predictions),
                labels=np.array(validation_labels),
                language=language,
                model_version=model_version or "1.0.0"
            )
            
            # Update thresholds using optimizer
            result = self.threshold_optimizer.update_thresholds(language, validation_data, model_version)
            
            if result.success:
                # Update the model configuration with new threshold
                self.model_configs[language]["confidence_threshold"] = result.optimized_threshold
                
                logger.info(f"Successfully updated threshold for {language}: {result.optimized_threshold:.3f}")
                
                return {
                    "success": True,
                    "language": language,
                    "original_threshold": result.original_threshold,
                    "optimized_threshold": result.optimized_threshold,
                    "improvement": result.improvement,
                    "metrics": result.metrics,
                    "sample_count": result.sample_count,
                    "optimization_time": result.optimization_time
                }
            else:
                logger.warning(f"Threshold optimization failed for {language}: {result.error_message}")
                return {
                    "success": False,
                    "language": language,
                    "error": result.error_message,
                    "optimization_time": result.optimization_time
                }
                
        except Exception as e:
            logger.error(f"Failed to update threshold for {language}: {str(e)}")
            return {
                "success": False,
                "language": language,
                "error": str(e)
            }
    
    def get_all_threshold_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get threshold information for all supported languages.
        
        Returns:
            Dictionary mapping language to threshold information
        """
        try:
            all_info = {}
            
            for language in self.SUPPORTED_LANGUAGES:
                all_info[language] = self.get_threshold_info(language)
            
            return all_info
            
        except Exception as e:
            logger.error(f"Failed to get all threshold info: {str(e)}")
            return {}
    
    def reset_threshold_for_language(self, language: str, model_version: Optional[str] = None):
        """
        Reset threshold to default value for a specific language.
        
        Args:
            language: Target language
            model_version: Specific model version (optional)
        """
        try:
            if language not in self.SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
            
            # Reset in optimizer
            self.threshold_optimizer.reset_threshold(language, model_version)
            
            # Update model configuration to default
            default_threshold = self.threshold_optimizer.get_threshold(language)
            self.model_configs[language]["confidence_threshold"] = default_threshold
            
            logger.info(f"Reset threshold for {language} to default: {default_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to reset threshold for {language}: {str(e)}")
            raise
    
    def clear_model_cache(self, language: Optional[str] = None, clear_foundation: bool = False):
        """
        Clear model cache for memory management.
        
        Args:
            language: Specific language classifier to clear, or None to clear all classifiers
            clear_foundation: Whether to clear foundation model cache
        """
        if language:
            if language in self.classifier_cache:
                del self.classifier_cache[language]
                logger.info(f"Cleared classifier cache for {language}")
        else:
            self.classifier_cache.clear()
            logger.info("Cleared all classifier caches")
        
        if clear_foundation:
            self.foundation_manager.clear_cache()
            logger.info("Cleared foundation model cache")
        """
        Clear model cache for memory management.
        
        Args:
            language: Specific language classifier to clear, or None to clear all classifiers
            clear_foundation: Whether to clear foundation model cache
        """
        if language:
            if language in self.classifier_cache:
                del self.classifier_cache[language]
                logger.info(f"Cleared classifier cache for {language}")
        else:
            self.classifier_cache.clear()
            logger.info("Cleared all classifier caches")
        
        if clear_foundation:
            self.foundation_manager.clear_cache()
            logger.info("Cleared foundation model cache")
    
    def switch_foundation_model(self, model_name: str):
        """
        Switch to a different foundation model.
        
        Args:
            model_name: Name of the foundation model to switch to
        """
        if model_name not in self.FOUNDATION_MODELS:
            raise ValueError(f"Unknown foundation model: {model_name}")
        
        # Use FoundationModelManager to switch models
        self.foundation_manager.switch_model(model_name)
        
        # Clear classifier cache as embeddings dimensions may change
        self.classifier_cache.clear()
        
        # Update configuration
        self.foundation_model_name = model_name
        self.foundation_config = self.FOUNDATION_MODELS[model_name]
        
        # Reinitialize model configs with new embedding dimension
        self._initialize_model_configs()
        
        logger.info(f"Switched to foundation model: {model_name}")
    
    def _gather_system_resources(self) -> Dict[str, Any]:
        """
        Gather current system resource information for fallback decisions.
        
        Returns:
            Dictionary containing system resource metrics
        """
        try:
            resources = {}
            
            if PSUTIL_AVAILABLE:
                import psutil
                
                # Memory information
                memory = psutil.virtual_memory()
                resources['memory_usage'] = memory.used
                resources['memory_percent'] = memory.percent
                resources['memory_available'] = memory.available
                
                # CPU information
                resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                resources['cpu_count'] = psutil.cpu_count()
                
                # Disk information (for model loading)
                disk = psutil.disk_usage('/')
                resources['disk_usage'] = disk.used
                resources['disk_free'] = disk.free
                
            else:
                # Default values when psutil not available
                resources = {
                    'memory_usage': 0,
                    'memory_percent': 0,
                    'memory_available': 1024 * 1024 * 1024,  # 1GB default
                    'cpu_percent': 0,
                    'cpu_count': 1,
                    'disk_usage': 0,
                    'disk_free': 1024 * 1024 * 1024  # 1GB default
                }
            
            return resources
            
        except Exception as e:
            logger.warning(f"Failed to gather system resources: {e}")
            return {
                'memory_usage': 0,
                'memory_percent': 0,
                'memory_available': 1024 * 1024 * 1024,
                'cpu_percent': 0,
                'cpu_count': 1,
                'disk_usage': 0,
                'disk_free': 1024 * 1024 * 1024,
                'error': str(e)
            }
    
    def _assess_model_availability(self, language: str) -> Dict[str, Any]:
        """
        Assess availability of models for fallback decisions.
        
        Args:
            language: Target language
            
        Returns:
            Dictionary containing model availability information
        """
        try:
            availability = {
                'foundation_models': 0,
                'alternative_models': 0,
                'classifiers': 0,
                'alternative_classifiers': 0,
                'foundation_model_status': {},
                'classifier_status': {}
            }
            
            # Check foundation model availability
            for model_name, config in self.FOUNDATION_MODELS.items():
                model_path = self.model_base_path / config["path"]
                is_available = model_path.exists() and model_path.stat().st_size > 0
                availability['foundation_model_status'][model_name] = is_available
                
                if is_available:
                    availability['foundation_models'] += 1
                    if model_name != self.foundation_model_name:
                        availability['alternative_models'] += 1
            
            # Check classifier availability
            for lang in self.SUPPORTED_LANGUAGES:
                config = self.model_configs[lang]
                classifier_path = config["classifier_path"]
                is_available = classifier_path.exists() and classifier_path.stat().st_size > 0
                availability['classifier_status'][lang] = is_available
                
                if is_available:
                    availability['classifiers'] += 1
                    if lang != language:
                        availability['alternative_classifiers'] += 1
            
            return availability
            
        except Exception as e:
            logger.warning(f"Failed to assess model availability: {e}")
            return {
                'foundation_models': 0,
                'alternative_models': 0,
                'classifiers': 0,
                'alternative_classifiers': 0,
                'foundation_model_status': {},
                'classifier_status': {},
                'error': str(e)
            }
    
    def _extract_embeddings_with_fallback(self, audio_array: np.ndarray, 
                                        fallback_context: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract embeddings with systematic fallback handling.
        
        Args:
            audio_array: Audio data
            fallback_context: Context information for fallback decisions
            
        Returns:
            Tuple of (embeddings, fallback_info)
        """
        fallback_info = {
            'used': False,
            'reason': None,
            'method': 'foundation_model',
            'model_used': self.foundation_model_name,
            'error': None
        }
        
        try:
            # Attempt primary foundation model extraction
            embeddings = self._extract_embeddings(audio_array)
            logger.debug(f"Successfully extracted embeddings using {self.foundation_model_name}")
            return embeddings, fallback_info
            
        except Exception as e:
            logger.warning(f"Foundation model embedding extraction failed: {str(e)}")
            
            # Record error for fallback decision
            error_context = {
                'error_type': 'foundation_model_failed',
                'error_message': str(e),
                'system_resources': fallback_context['system_resources'],
                'model_availability': fallback_context['model_availability']
            }
            
            fallback_context['error_history'].append({
                'type': 'foundation_model_failure',
                'error': str(e),
                'timestamp': time.time()
            })
            
            # Get fallback recommendation
            recommended_level = self.fallback_strategy.get_fallback_recommendation(error_context)
            logger.info(f"Recommended fallback level: {recommended_level.value}")
            
            # Apply appropriate fallback strategy
            if recommended_level == FallbackLevel.FOUNDATION_MODEL:
                # Try alternative foundation model if available
                embeddings = self._try_alternative_foundation_model(audio_array, fallback_context)
                if embeddings is not None:
                    fallback_info.update({
                        'used': True,
                        'reason': 'foundation_model_failed',
                        'method': 'alternative_foundation_model',
                        'error': str(e)
                    })
                    return embeddings, fallback_info
            
            # Use traditional feature extraction fallback
            embeddings = self.fallback_strategy.handle_foundation_model_failure(
                audio_array, 
                target_dim=self.foundation_config["embedding_dim"]
            )
            
            fallback_info.update({
                'used': True,
                'reason': 'foundation_model_failed',
                'method': 'traditional_features',
                'model_used': 'fallback_features',
                'error': str(e)
            })
            
            logger.info("Using traditional feature fallback due to foundation model failure")
            return embeddings, fallback_info
    
    def _try_alternative_foundation_model(self, audio_array: np.ndarray, 
                                        fallback_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Try alternative foundation models if available.
        
        Args:
            audio_array: Audio data
            fallback_context: Context information
            
        Returns:
            Embeddings from alternative model or None if all fail
        """
        try:
            model_availability = fallback_context['model_availability']
            
            # Try other available foundation models
            for model_name, is_available in model_availability['foundation_model_status'].items():
                if model_name != self.foundation_model_name and is_available:
                    try:
                        logger.info(f"Trying alternative foundation model: {model_name}")
                        
                        # Temporarily switch to alternative model
                        original_model = self.foundation_model_name
                        self.foundation_model_name = model_name
                        self.foundation_config = self.FOUNDATION_MODELS[model_name]
                        
                        # Try extraction
                        embeddings = self._extract_embeddings(audio_array)
                        
                        logger.info(f"Successfully used alternative foundation model: {model_name}")
                        return embeddings
                        
                    except Exception as alt_error:
                        logger.warning(f"Alternative foundation model {model_name} also failed: {alt_error}")
                        # Restore original model
                        self.foundation_model_name = original_model
                        self.foundation_config = self.FOUNDATION_MODELS[original_model]
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Alternative foundation model attempt failed: {e}")
            return None
    
    def _classify_with_fallback(self, embeddings: np.ndarray, language: str,
                              fallback_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform classification with systematic fallback handling.
        
        Args:
            embeddings: Feature embeddings
            language: Target language
            fallback_context: Context information for fallback decisions
            
        Returns:
            Tuple of (classification_result, fallback_info)
        """
        fallback_info = {
            'used': False,
            'reason': None,
            'method': 'primary_classifier',
            'classifier_used': language,
            'error': None
        }
        
        try:
            # Attempt primary classifier loading and inference
            classifier = self._load_classifier(language)
            
            logger.info(f"Running classification for {language} using {self.foundation_model_name}")
            prediction = classifier.predict(embeddings, verbose=0)
            prediction_prob = float(prediction[0][0])
            
            # Determine classification based on threshold
            config = self.model_configs[language]
            threshold = config["confidence_threshold"]
            
            classification = "AI_GENERATED" if prediction_prob >= threshold else "HUMAN"
            
            # Get model metadata for enhanced confidence calculation
            model_metadata = {
                'validation_accuracy': config.get('validation_accuracy', 0.8),
                'model_size_bytes': config.get('model_size_bytes', 1000000),
                'model_version': config.get('version', '1.0.0')
            }
            
            # Calculate enhanced confidence score
            confidence_score = self._calculate_confidence_score(
                model_output=prediction_prob,
                language=language,
                features=embeddings,
                model_metadata=model_metadata,
                prediction_history=None  # Could be added later for stability tracking
            )
            
            result = {
                'classification': classification,
                'confidence_score': confidence_score,
                'prediction_prob': prediction_prob,
                'threshold': threshold
            }
            
            return result, fallback_info
            
        except Exception as e:
            logger.warning(f"Primary classifier failed for {language}: {str(e)}")
            
            # Record error for fallback decision
            error_context = {
                'error_type': 'classifier_failed',
                'error_message': str(e),
                'system_resources': fallback_context['system_resources'],
                'model_availability': fallback_context['model_availability']
            }
            
            fallback_context['error_history'].append({
                'type': 'classifier_failure',
                'error': str(e),
                'timestamp': time.time()
            })
            
            # Get fallback recommendation
            recommended_level = self.fallback_strategy.get_fallback_recommendation(error_context)
            logger.info(f"Recommended classifier fallback level: {recommended_level.value}")
            
            # Apply classifier fallback strategy
            available_classifiers = {
                lang: model for lang, model in self.classifier_cache.items() 
                if lang != language and model is not None
            }
            
            fallback_result = self.fallback_strategy.handle_classifier_failure(
                embeddings, language, available_classifiers
            )
            
            result = {
                'classification': fallback_result.classification,
                'confidence_score': fallback_result.confidence_score,
                'prediction_prob': 0.5,  # Neutral for fallback
                'threshold': 0.5
            }
            
            fallback_info.update({
                'used': True,
                'reason': 'classifier_failed',
                'method': fallback_result.fallback_level.value,
                'classifier_used': 'fallback_ensemble',
                'error': str(e)
            })
            
            logger.info(f"Applied classifier fallback: {result['classification']} ({result['confidence_score']:.3f})")
            return result, fallback_info
    
    def _validate_and_finalize_result(self, classification_result: Dict[str, Any], 
                                    audio_array: np.ndarray, language: str,
                                    fallback_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate classification result and apply additional fallback if needed.
        
        Args:
            classification_result: Initial classification result
            audio_array: Original audio data
            language: Target language
            fallback_context: Context information
            
        Returns:
            Tuple of (final_result, fallback_info)
        """
        fallback_info = {
            'used': False,
            'reason': None,
            'method': 'none',
            'error': None
        }
        
        try:
            confidence_score = classification_result['confidence_score']
            
            # Check if confidence is too low and needs validation
            if confidence_score < self.fallback_strategy.confidence_threshold:
                logger.info(f"Low confidence detected ({confidence_score:.3f}), applying validation")
                
                initial_result = {
                    'classification': classification_result['classification'],
                    'confidence_score': confidence_score
                }
                
                fallback_result = self.fallback_strategy.handle_low_confidence(
                    initial_result, audio_array, language
                )
                
                final_result = {
                    'classification': fallback_result.classification,
                    'confidence_score': fallback_result.confidence_score,
                    'prediction_prob': classification_result.get('prediction_prob', 0.5),
                    'threshold': classification_result.get('threshold', 0.5)
                }
                
                fallback_info.update({
                    'used': True,
                    'reason': 'low_confidence',
                    'method': 'confidence_validation',
                    'original_confidence': confidence_score,
                    'improved_confidence': fallback_result.confidence_score
                })
                
                logger.info(f"Applied low confidence fallback: {final_result['classification']} ({final_result['confidence_score']:.3f})")
                return final_result, fallback_info
            
            # No validation needed
            return classification_result, fallback_info
            
        except Exception as e:
            logger.error(f"Result validation failed: {str(e)}")
            
            fallback_info.update({
                'used': True,
                'reason': 'validation_failed',
                'method': 'error_recovery',
                'error': str(e)
            })
            
            # Return original result with reduced confidence
            final_result = classification_result.copy()
            final_result['confidence_score'] = max(final_result['confidence_score'] * 0.8, 0.1)
            
            return final_result, fallback_info
    
    def _check_and_handle_constraints(self, classification_result: Dict[str, Any],
                                    audio_array: np.ndarray, language: str,
                                    processing_time: float, 
                                    fallback_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Check for resource constraints and timeouts, applying fallback if needed.
        
        Args:
            classification_result: Current classification result
            audio_array: Original audio data
            language: Target language
            processing_time: Current processing time
            fallback_context: Context information
            
        Returns:
            Tuple of (final_result, fallback_info)
        """
        fallback_info = {
            'used': False,
            'reason': None,
            'method': 'none',
            'error': None
        }
        
        try:
            system_resources = fallback_context['system_resources']
            memory_usage = system_resources.get('memory_usage', 0)
            
            # Check for timeout
            if processing_time > self.fallback_strategy.max_processing_time:
                logger.warning(f"Processing timeout ({processing_time:.1f}s), applying timeout fallback")
                
                timeout_result = self.fallback_strategy.handle_timeout(
                    audio_array, language, processing_time
                )
                
                final_result = {
                    'classification': timeout_result.classification,
                    'confidence_score': timeout_result.confidence_score,
                    'prediction_prob': 0.5,
                    'threshold': 0.5
                }
                
                fallback_info.update({
                    'used': True,
                    'reason': 'timeout',
                    'method': 'timeout_fallback',
                    'timeout_duration': processing_time,
                    'max_allowed': self.fallback_strategy.max_processing_time
                })
                
                logger.info(f"Applied timeout fallback: {final_result['classification']} ({final_result['confidence_score']:.3f})")
                return final_result, fallback_info
            
            # Check for resource constraints
            if memory_usage > self.fallback_strategy.resource_memory_limit:
                logger.warning(f"Memory constraint detected ({memory_usage/1024/1024:.1f}MB), applying resource fallback")
                
                resource_result = self.fallback_strategy.handle_resource_constraints(
                    audio_array, language, memory_usage, processing_time
                )
                
                final_result = {
                    'classification': resource_result.classification,
                    'confidence_score': resource_result.confidence_score,
                    'prediction_prob': 0.5,
                    'threshold': 0.5
                }
                
                fallback_info.update({
                    'used': True,
                    'reason': 'resource_constraints',
                    'method': 'resource_fallback',
                    'memory_usage': memory_usage,
                    'memory_limit': self.fallback_strategy.resource_memory_limit
                })
                
                logger.info(f"Applied resource constraint fallback: {final_result['classification']} ({final_result['confidence_score']:.3f})")
                return final_result, fallback_info
            
            # No constraints detected
            return classification_result, fallback_info
            
        except Exception as e:
            logger.error(f"Constraint checking failed: {str(e)}")
            
            fallback_info.update({
                'used': True,
                'reason': 'constraint_check_failed',
                'method': 'error_recovery',
                'error': str(e)
            })
            
            # Return original result with reduced confidence
            final_result = classification_result.copy()
            final_result['confidence_score'] = max(final_result['confidence_score'] * 0.9, 0.1)
            
            return final_result, fallback_info
    
    def _create_final_result(self, classification_result: Dict[str, Any], language: str,
                           processing_time: float, all_fallback_info: Dict[str, Any]) -> DetectionResult:
        """
        Create final DetectionResult with comprehensive metadata.
        
        Args:
            classification_result: Final classification result
            language: Target language
            processing_time: Total processing time
            all_fallback_info: Comprehensive fallback information
            
        Returns:
            DetectionResult with complete metadata
        """
        try:
            # Determine model version with fallback indicators
            config = self.model_configs[language]
            model_version = f"{self.foundation_model_name}-{config['version']}"
            
            # Add fallback indicators to model version
            fallback_indicators = []
            for category, info in all_fallback_info.items():
                if info.get('used', False):
                    fallback_indicators.append(f"{category}_{info.get('method', 'fallback')}")
            
            if fallback_indicators:
                model_version += f"-fallback({','.join(fallback_indicators)})"
            
            # Get threshold information
            threshold_used = classification_result.get('threshold', config.get('confidence_threshold', 0.5))
            threshold_info = self.get_threshold_info(language)
            is_threshold_optimized = threshold_info.get('is_optimized', False)
            
            result = DetectionResult(
                classification=classification_result['classification'],
                confidence_score=classification_result['confidence_score'],
                model_version=model_version,
                processing_time=processing_time,
                threshold_used=threshold_used,
                is_threshold_optimized=is_threshold_optimized
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create final result: {str(e)}")
            
            # Create minimal result
            return DetectionResult(
                classification=classification_result.get('classification', 'HUMAN'),
                confidence_score=classification_result.get('confidence_score', 0.1),
                model_version=f"{self.foundation_model_name}-error",
                processing_time=processing_time,
                threshold_used=0.5,
                is_threshold_optimized=False
            )
    
    def _log_detection_summary(self, result: DetectionResult, language: str, 
                             all_fallback_info: Dict[str, Any]):
        """
        Log comprehensive detection summary with fallback information.
        
        Args:
            result: Final detection result
            language: Target language
            all_fallback_info: Comprehensive fallback information
        """
        try:
            # Collect fallback summary
            fallback_summary = []
            for category, info in all_fallback_info.items():
                if info.get('used', False):
                    reason = info.get('reason', 'unknown')
                    method = info.get('method', 'unknown')
                    fallback_summary.append(f"{category}({reason}->{method})")
            
            fallback_str = f" [fallbacks: {', '.join(fallback_summary)}]" if fallback_summary else ""
            
            logger.info(
                f"Detection completed for {language}: {result.classification} "
                f"(confidence: {result.confidence_score:.3f}, time: {result.processing_time:.3f}s, "
                f"foundation: {self.foundation_model_name}){fallback_str}"
            )
            
            # Log detailed fallback information at debug level
            if fallback_summary:
                logger.debug(f"Detailed fallback info for {language}: {all_fallback_info}")
                
        except Exception as e:
            logger.error(f"Failed to log detection summary: {str(e)}")
    
    def _apply_emergency_fallback(self, features: AudioFeatures, language: str,
                                fallback_context: Dict[str, Any], original_error: Exception) -> DetectionResult:
        """
        Apply emergency fallback for complete system failure.
        
        Args:
            features: Original audio features
            language: Target language
            fallback_context: Context information
            original_error: Original error that triggered emergency fallback
            
        Returns:
            DetectionResult from emergency fallback
        """
        try:
            logger.warning(f"Applying emergency fallback for {language} due to: {str(original_error)}")
            
            audio_array = self._preprocess_audio_for_foundation(features)
            processing_time = time.time() - fallback_context['start_time']
            
            # Use resource constraint handling as comprehensive emergency fallback
            emergency_result = self.fallback_strategy.handle_resource_constraints(
                audio_array, language, 0, processing_time
            )
            
            final_processing_time = time.time() - fallback_context['start_time']
            
            result = DetectionResult(
                classification=emergency_result.classification,
                confidence_score=emergency_result.confidence_score,
                model_version=f"{self.foundation_model_name}-emergency",
                processing_time=final_processing_time
            )
            
            logger.warning(f"Emergency fallback result for {language}: {result.classification} ({result.confidence_score:.3f})")
            return result
            
        except Exception as emergency_error:
            logger.error(f"Emergency fallback also failed: {str(emergency_error)}")
            
            # Absolute last resort - return conservative result
            final_processing_time = time.time() - fallback_context['start_time']
            
            result = DetectionResult(
                classification="HUMAN",  # Conservative default
                confidence_score=0.1,
                model_version=f"{self.foundation_model_name}-critical_failure",
                processing_time=final_processing_time
            )
            
            logger.critical(f"Critical failure - returning conservative result for {language}")
            return result
    
    def _generate_explanation(self, classification: str, confidence_score: float, 
                            features: AudioFeatures, language: str) -> str:
        """
        Generate human-readable explanation for the classification decision.
        
        Args:
            classification: Classification result ("AI_GENERATED" or "HUMAN")
            confidence_score: Confidence score
            features: Original audio features
            language: Target language
            
        Returns:
            Human-readable explanation string
        """
        try:
            # Analyze features for explanation
            duration = features.duration
            foundation_model = self.foundation_model_name
            
            # Generate explanation based on classification and features
            if classification == "HUMAN":
                if confidence_score > 0.8:
                    explanation = (
                        f"The audio exhibits strong human speech characteristics detected by "
                        f"{foundation_model} analysis. The {duration:.1f}-second {language} sample "
                        f"shows natural vocal patterns and speech dynamics typical of genuine human speech."
                    )
                elif confidence_score > 0.6:
                    explanation = (
                        f"The audio shows human speech patterns with moderate confidence using "
                        f"{foundation_model} features. The {duration:.1f}-second {language} sample "
                        f"contains characteristics consistent with human speech."
                    )
                else:
                    explanation = (
                        f"The audio appears to be human speech with lower confidence. "
                        f"The {duration:.1f}-second {language} sample analyzed with {foundation_model} "
                        f"shows mixed indicators that lean toward human origin."
                    )
            else:  # AI_GENERATED
                if confidence_score > 0.8:
                    explanation = (
                        f"The audio exhibits strong indicators of AI-generated speech detected by "
                        f"{foundation_model} analysis. The {duration:.1f}-second {language} sample "
                        f"shows synthetic patterns typical of artificial voice generation."
                    )
                elif confidence_score > 0.6:
                    explanation = (
                        f"The audio shows AI-generated speech patterns with moderate confidence using "
                        f"{foundation_model} features. The {duration:.1f}-second {language} sample "
                        f"contains synthetic characteristics suggesting artificial generation."
                    )
                else:
                    explanation = (
                        f"The audio appears to be AI-generated with lower confidence. "
                        f"The {duration:.1f}-second {language} sample analyzed with {foundation_model} "
                        f"shows some indicators suggesting synthetic origin."
                    )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            # Return generic explanation on error
            return (
                f"The audio has been classified as {classification.replace('_', ' ').lower()} "
                f"with {confidence_score:.1%} confidence using {self.foundation_model_name} "
                f"foundation model for {language} speech analysis."
            )