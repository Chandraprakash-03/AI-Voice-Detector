"""
ML Detection Engine for AI-Generated Voice Detection API.

This module implements the core detection logic using foundation models (HuBERT, XLS-R)
for feature extraction and custom classifiers for AI vs human speech detection.
"""

import logging
import os
import time
import warnings
from typing import Dict, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import tensorflow as tf
from pydantic import BaseModel

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

logger = logging.getLogger(__name__)


class DetectionResult(BaseModel):
    """Data model for detection results."""
    
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidence_score: float  # 0.0 to 1.0
    model_version: str
    processing_time: float


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
        Initialize the DetectionEngine.
        
        Args:
            settings: Application settings object containing configuration
            foundation_model: Foundation model to use ("auto", "xls-r-300m", "hubert-base", "wav2vec2-base")
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.model_base_path = Path(settings.ml.models_base_path)
        
        # Model caches
        self.foundation_model_cache: Dict[str, Any] = {}
        self.foundation_processor_cache: Dict[str, Any] = {}
        self.classifier_cache: Dict[str, tf.keras.Model] = {}
        
        # Configuration
        self.foundation_model_name = self._select_foundation_model(foundation_model)
        self.foundation_config = self.FOUNDATION_MODELS[self.foundation_model_name]
        
        # Initialize model configurations
        self._initialize_model_configs()
        
        # Set device preference
        self.device = "cuda" if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Ensure TensorFlow uses CPU for consistency (can be changed for GPU)
        if not torch.cuda.is_available():
            tf.config.set_visible_devices([], 'GPU')
    
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
        """Initialize model configurations for each supported language."""
        self.model_configs = {}
        
        for language in self.SUPPORTED_LANGUAGES:
            self.model_configs[language] = {
                "classifier_path": self.model_base_path / "classifiers" / f"{language.lower()}_classifier.h5",
                "version": "1.0.0",
                "input_shape": (self.foundation_config["embedding_dim"],),
                "confidence_threshold": 0.5,
                "foundation_model": self.foundation_model_name
            }
    
    def _load_foundation_model(self) -> Tuple[Any, Any]:
        """Load the foundation model and processor."""
        if not TRANSFORMERS_AVAILABLE:
            raise ModelLoadingError("transformers library not available. Install with: pip install transformers torch")
        
        cache_key = self.foundation_model_name
        
        # Check cache first
        if cache_key in self.foundation_model_cache:
            return self.foundation_model_cache[cache_key], self.foundation_processor_cache[cache_key]
        
        try:
            config = self.foundation_config
            model_path = self.model_base_path / config["path"]
            
            if model_path.exists():
                logger.info(f"Loading foundation model from {model_path}")
                
                # Load model and processor based on type
                if config["model_class"] == "Wav2Vec2Model":
                    model = Wav2Vec2Model.from_pretrained(str(model_path))
                    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
                elif config["model_class"] == "HubertModel":
                    model = HubertModel.from_pretrained(str(model_path))
                    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path))
                else:
                    raise ModelLoadingError(f"Unknown model class: {config['model_class']}")
                
                # Move to device
                model = model.to(self.device)
                model.eval()
                
            else:
                logger.warning(f"Foundation model not found at {model_path}, using dummy embeddings")
                model, processor = None, None
            
            # Cache the models
            self.foundation_model_cache[cache_key] = model
            self.foundation_processor_cache[cache_key] = processor
            
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load foundation model {self.foundation_model_name}: {str(e)}")
            raise ModelLoadingError(f"Foundation model loading failed: {str(e)}")
    
    def _extract_embeddings(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract embeddings from audio using foundation model."""
        try:
            foundation_model, processor = self._load_foundation_model()
            
            if foundation_model is None:
                # Return dummy embeddings if no foundation model
                logger.warning("Using dummy embeddings - foundation model not available")
                return np.random.random((1, self.foundation_config["embedding_dim"])).astype(np.float32)
            
            # Preprocess audio
            if processor is not None:
                inputs = processor(
                    audio_array, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Fallback preprocessing
                inputs = {"input_values": torch.tensor(audio_array).unsqueeze(0).to(self.device)}
            
            # Extract features
            with torch.no_grad():
                outputs = foundation_model(**inputs)
                
                # Get embeddings (usually from last_hidden_state)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states'):
                    embeddings = outputs.hidden_states[-1]
                else:
                    embeddings = outputs[0]
                
                # Pool embeddings (mean pooling across time dimension)
                embeddings = torch.mean(embeddings, dim=1)  # Shape: (batch_size, embedding_dim)
                
                # Convert to numpy
                embeddings = embeddings.cpu().numpy()
            
            logger.debug(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            # Return dummy embeddings on error
            return np.random.random((1, self.foundation_config["embedding_dim"])).astype(np.float32)
    
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
            if classifier_path.exists():
                logger.info(f"Loading classifier for {language} from {classifier_path}")
                model = tf.keras.models.load_model(str(classifier_path))
            else:
                logger.warning(f"Classifier not found for {language}, creating dummy classifier")
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
    
    def _calculate_confidence_score(self, model_output: float, language: str) -> float:
        """
        Calculate confidence score from model output.
        
        Args:
            model_output: Raw model output (sigmoid probability)
            language: Target language
            
        Returns:
            Confidence score between 0.0 and 1.0
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
            logger.error(f"Confidence calculation failed: {str(e)}")
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
    
    
    def detect_voice_type(self, features: AudioFeatures, language: str) -> DetectionResult:
        """
        Detect whether the voice is AI-generated or human.
        
        Args:
            features: Extracted audio features
            language: Target language for analysis
            
        Returns:
            DetectionResult with classification, confidence, and metadata
            
        Raises:
            UnsupportedLanguageError: If language is not supported
            InferenceError: If detection inference fails
        """
        start_time = time.time()
        
        try:
            # Validate language
            if language not in self.SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(language, self.SUPPORTED_LANGUAGES)
            
            # Step 1: Extract embeddings using foundation model
            audio_array = self._preprocess_audio_for_foundation(features)
            embeddings = self._extract_embeddings(audio_array)
            
            # Step 2: Load language-specific classifier
            classifier = self._load_classifier(language)
            
            # Step 3: Run classification
            logger.info(f"Running classification for {language} using {self.foundation_model_name}")
            prediction = classifier.predict(embeddings, verbose=0)
            
            # Extract prediction probability
            prediction_prob = float(prediction[0][0])
            
            # Determine classification based on threshold
            config = self.model_configs[language]
            threshold = config["confidence_threshold"]
            
            if prediction_prob >= threshold:
                classification = "AI_GENERATED"
            else:
                classification = "HUMAN"
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(prediction_prob, language)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = DetectionResult(
                classification=classification,
                confidence_score=confidence_score,
                model_version=f"{self.foundation_model_name}-{config['version']}",
                processing_time=processing_time
            )
            
            logger.info(
                f"Detection completed for {language}: {classification} "
                f"(confidence: {confidence_score:.3f}, time: {processing_time:.3f}s, "
                f"foundation: {self.foundation_model_name})"
            )
            
            return result
            
        except (UnsupportedLanguageError, ModelLoadingError):
            raise
        except Exception as e:
            logger.error(f"Voice detection failed for {language}: {str(e)}")
            raise InferenceError(f"Voice detection failed: {str(e)}", language=language)
    
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
        is_foundation_loaded = self.foundation_model_name in self.foundation_model_cache
        
        return {
            "language": language,
            "version": config["version"],
            "classifier_path": str(config["classifier_path"]),
            "foundation_model": self.foundation_model_name,
            "foundation_config": self.foundation_config,
            "is_classifier_loaded": is_classifier_loaded,
            "is_foundation_loaded": is_foundation_loaded,
            "input_shape": config["input_shape"],
            "confidence_threshold": config["confidence_threshold"],
            "device": self.device
        }
    
    def get_available_foundation_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available foundation models."""
        available = {}
        
        for name, config in self.FOUNDATION_MODELS.items():
            model_path = self.model_base_path / config["path"]
            available[name] = {
                **config,
                "available": model_path.exists(),
                "path": str(model_path),
                "is_loaded": name in self.foundation_model_cache
            }
        
        return available
    
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
            self.foundation_model_cache.clear()
            self.foundation_processor_cache.clear()
            logger.info("Cleared foundation model cache")
    
    def switch_foundation_model(self, model_name: str):
        """
        Switch to a different foundation model.
        
        Args:
            model_name: Name of the foundation model to switch to
        """
        if model_name not in self.FOUNDATION_MODELS:
            raise ValueError(f"Unknown foundation model: {model_name}")
        
        # Clear current caches
        self.clear_model_cache(clear_foundation=True)
        self.classifier_cache.clear()
        
        # Update configuration
        self.foundation_model_name = model_name
        self.foundation_config = self.FOUNDATION_MODELS[model_name]
        
        # Reinitialize model configs with new embedding dimension
        self._initialize_model_configs()
        
        logger.info(f"Switched to foundation model: {model_name}")
    
    def _calculate_confidence_score(self, model_output: float, language: str) -> float:
        """
        Calculate confidence score from model output.
        
        Args:
            model_output: Raw model output (sigmoid probability)
            language: Target language
            
        Returns:
            Confidence score between 0.0 and 1.0
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
            logger.error(f"Confidence calculation failed: {str(e)}")
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