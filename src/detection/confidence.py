"""
Enhanced Confidence Scoring System for AI Voice Detection.

This module implements improved confidence calculation methods that replace
random confidence generation with meaningful uncertainty measures based on
actual model certainty and prediction characteristics.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty in model predictions."""
    ALEATORIC = "aleatoric"  # Data uncertainty (inherent noise)
    EPISTEMIC = "epistemic"  # Model uncertainty (lack of knowledge)
    PREDICTION = "prediction"  # Prediction uncertainty (close to decision boundary)
    FEATURE = "feature"  # Feature uncertainty (poor quality features)
    ENSEMBLE = "ensemble"  # Ensemble disagreement uncertainty


class ConfidenceLevel(Enum):
    """Confidence level categories for human interpretation."""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"  # 0.7 - 0.9
    MODERATE = "moderate"  # 0.5 - 0.7
    LOW = "low"  # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


@dataclass
class UncertaintyIndicator:
    """Indicator for specific type of uncertainty."""
    uncertainty_type: UncertaintyType
    value: float  # 0.0 to 1.0, higher means more uncertain
    description: str
    contributing_factors: List[str]


class ConfidenceMetrics(BaseModel):
    """Comprehensive confidence metrics for a prediction."""
    
    raw_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw confidence score from basic calculation"
    )
    
    adjusted_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence adjusted for uncertainty factors"
    )
    
    final_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Final confidence score after all adjustments"
    )
    
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Human-readable confidence level category"
    )
    
    uncertainty_indicators: List[UncertaintyIndicator] = Field(
        default_factory=list,
        description="List of uncertainty indicators affecting confidence"
    )
    
    prediction_stability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Stability of prediction across multiple evaluations"
    )
    
    feature_quality_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Quality assessment of input features"
    )
    
    model_certainty: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model's intrinsic certainty about the prediction"
    )
    
    calibration_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How well-calibrated the confidence is"
    )


class EnhancedConfidenceCalculator:
    """
    Enhanced confidence calculator that provides meaningful uncertainty measures.
    
    This calculator replaces simple distance-based confidence with a comprehensive
    system that considers multiple sources of uncertainty and provides detailed
    confidence metrics.
    """
    
    def __init__(self, settings=None):
        """
        Initialize the enhanced confidence calculator.
        
        Args:
            settings: Application settings object
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        
        # Confidence calculation parameters
        self.uncertainty_weights = {
            UncertaintyType.PREDICTION: 0.3,
            UncertaintyType.FEATURE: 0.25,
            UncertaintyType.EPISTEMIC: 0.2,
            UncertaintyType.ALEATORIC: 0.15,
            UncertaintyType.ENSEMBLE: 0.1
        }
        
        # Calibration parameters for different confidence ranges
        self.calibration_params = {
            "very_high": {"threshold": 0.9, "adjustment": 0.95},
            "high": {"threshold": 0.7, "adjustment": 0.85},
            "moderate": {"threshold": 0.5, "adjustment": 0.65},
            "low": {"threshold": 0.3, "adjustment": 0.4},
            "very_low": {"threshold": 0.0, "adjustment": 0.2}
        }
        
        # Feature quality thresholds
        self.feature_quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "fair": 0.5,
            "poor": 0.3
        }
        
        logger.info("Enhanced confidence calculator initialized")
    
    def calculate_enhanced_confidence(
        self,
        model_output: float,
        language: str,
        features: Optional[np.ndarray] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        prediction_history: Optional[List[float]] = None
    ) -> ConfidenceMetrics:
        """
        Calculate enhanced confidence metrics with comprehensive uncertainty analysis.
        
        Args:
            model_output: Raw model output probability (0.0 to 1.0)
            language: Target language for language-specific adjustments
            features: Input features used for prediction (optional)
            model_metadata: Metadata about the model used (optional)
            prediction_history: History of predictions for stability analysis (optional)
            
        Returns:
            ConfidenceMetrics: Comprehensive confidence analysis
        """
        try:
            start_time = time.time()
            
            # Step 1: Calculate raw confidence using improved method
            raw_confidence = self._calculate_raw_confidence(model_output, language)
            
            # Step 2: Analyze uncertainty indicators
            uncertainty_indicators = self._analyze_uncertainty_indicators(
                model_output, language, features, model_metadata, prediction_history
            )
            
            # Step 3: Calculate feature quality score
            feature_quality_score = self._assess_feature_quality(features)
            
            # Step 4: Calculate prediction stability
            prediction_stability = self._calculate_prediction_stability(
                model_output, prediction_history
            )
            
            # Step 5: Calculate model certainty
            model_certainty = self._calculate_model_certainty(
                model_output, model_metadata
            )
            
            # Step 6: Apply uncertainty adjustments
            adjusted_confidence = self._apply_uncertainty_adjustments(
                raw_confidence, uncertainty_indicators
            )
            
            # Step 7: Apply calibration
            final_confidence, calibration_score = self._apply_confidence_calibration(
                adjusted_confidence, model_output, language
            )
            
            # Step 8: Determine confidence level
            confidence_level = self._determine_confidence_level(final_confidence)
            
            # Create comprehensive metrics
            metrics = ConfidenceMetrics(
                raw_confidence=raw_confidence,
                adjusted_confidence=adjusted_confidence,
                final_confidence=final_confidence,
                confidence_level=confidence_level,
                uncertainty_indicators=uncertainty_indicators,
                prediction_stability=prediction_stability,
                feature_quality_score=feature_quality_score,
                model_certainty=model_certainty,
                calibration_score=calibration_score
            )
            
            processing_time = time.time() - start_time
            logger.debug(
                f"Enhanced confidence calculated for {language}: "
                f"{final_confidence:.3f} ({confidence_level.value}) "
                f"in {processing_time:.3f}s"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Enhanced confidence calculation failed: {str(e)}")
            # Return fallback confidence metrics
            return self._create_fallback_confidence_metrics(model_output)
    
    def _calculate_raw_confidence(self, model_output: float, language: str) -> float:
        """
        Calculate raw confidence using improved method.
        
        This method improves upon the basic distance-from-0.5 approach by
        considering the sigmoid function's properties and language-specific
        characteristics.
        
        Args:
            model_output: Raw model output probability
            language: Target language
            
        Returns:
            Raw confidence score (0.0 to 1.0)
        """
        try:
            # Improved confidence calculation using sigmoid properties
            # The sigmoid function has steeper gradients near 0.5, so we need
            # to account for this non-linear relationship
            
            # Calculate distance from decision boundary (0.5)
            distance_from_boundary = abs(model_output - 0.5)
            
            # Apply non-linear transformation to account for sigmoid properties
            # Use a power function to emphasize extreme values
            power_factor = 1.5  # Adjustable parameter
            normalized_distance = (distance_from_boundary * 2.0) ** power_factor
            
            # Scale back to 0-1 range
            base_confidence = min(normalized_distance, 1.0)
            
            # Apply language-specific adjustments
            language_adjustment = self._get_language_confidence_adjustment(
                language, model_output
            )
            
            # Combine base confidence with language adjustment
            raw_confidence = min(base_confidence * language_adjustment, 1.0)
            
            # Ensure minimum confidence for very uncertain predictions
            raw_confidence = max(raw_confidence, 0.05)
            
            return float(raw_confidence)
            
        except Exception as e:
            logger.error(f"Raw confidence calculation failed: {str(e)}")
            # Fallback to basic calculation
            return min(abs(model_output - 0.5) * 2.0, 1.0)
    
    def _get_language_confidence_adjustment(self, language: str, model_output: float) -> float:
        """
        Get language-specific confidence adjustment factor.
        
        Different languages may have different model performance characteristics,
        so we adjust confidence accordingly.
        
        Args:
            language: Target language
            model_output: Raw model output
            
        Returns:
            Adjustment factor (typically 0.8 to 1.2)
        """
        try:
            # Language-specific adjustment factors based on model performance
            language_factors = {
                "English": 1.0,  # Baseline
                "Tamil": 0.95,   # Slightly lower due to less training data
                "Hindi": 0.98,   # Good performance
                "Malayalam": 0.92,  # Lower confidence due to complexity
                "Telugu": 0.94   # Moderate adjustment
            }
            
            base_factor = language_factors.get(language, 0.9)
            
            # Additional adjustment based on prediction extremity
            # More extreme predictions get higher confidence for well-performing languages
            extremity = abs(model_output - 0.5) * 2.0
            extremity_bonus = extremity * 0.1 if base_factor >= 0.95 else 0.0
            
            return min(base_factor + extremity_bonus, 1.2)
            
        except Exception as e:
            logger.error(f"Language adjustment calculation failed: {str(e)}")
            return 1.0
    
    def _analyze_uncertainty_indicators(
        self,
        model_output: float,
        language: str,
        features: Optional[np.ndarray],
        model_metadata: Optional[Dict[str, Any]],
        prediction_history: Optional[List[float]]
    ) -> List[UncertaintyIndicator]:
        """
        Analyze various sources of uncertainty in the prediction.
        
        Args:
            model_output: Raw model output
            language: Target language
            features: Input features (optional)
            model_metadata: Model metadata (optional)
            prediction_history: Prediction history (optional)
            
        Returns:
            List of uncertainty indicators
        """
        indicators = []
        
        try:
            # 1. Prediction uncertainty (closeness to decision boundary)
            prediction_uncertainty = self._calculate_prediction_uncertainty(model_output)
            if prediction_uncertainty > 0.3:
                indicators.append(UncertaintyIndicator(
                    uncertainty_type=UncertaintyType.PREDICTION,
                    value=prediction_uncertainty,
                    description=f"Prediction close to decision boundary ({model_output:.3f})",
                    contributing_factors=["close_to_threshold", "ambiguous_classification"]
                ))
            
            # 2. Feature quality uncertainty
            if features is not None:
                feature_uncertainty = self._calculate_feature_uncertainty(features)
                if feature_uncertainty > 0.4:
                    indicators.append(UncertaintyIndicator(
                        uncertainty_type=UncertaintyType.FEATURE,
                        value=feature_uncertainty,
                        description="Poor quality or noisy input features detected",
                        contributing_factors=["low_snr", "feature_inconsistency", "missing_features"]
                    ))
            
            # 3. Model epistemic uncertainty
            if model_metadata is not None:
                epistemic_uncertainty = self._calculate_epistemic_uncertainty(
                    model_metadata, language
                )
                if epistemic_uncertainty > 0.3:
                    indicators.append(UncertaintyIndicator(
                        uncertainty_type=UncertaintyType.EPISTEMIC,
                        value=epistemic_uncertainty,
                        description="Model uncertainty due to limited training data",
                        contributing_factors=["limited_training_data", "model_complexity", "domain_shift"]
                    ))
            
            # 4. Prediction stability uncertainty
            if prediction_history is not None and len(prediction_history) > 1:
                stability_uncertainty = self._calculate_stability_uncertainty(prediction_history)
                if stability_uncertainty > 0.4:
                    indicators.append(UncertaintyIndicator(
                        uncertainty_type=UncertaintyType.ENSEMBLE,
                        value=stability_uncertainty,
                        description="Inconsistent predictions across multiple evaluations",
                        contributing_factors=["prediction_variance", "model_instability"]
                    ))
            
            # 5. Aleatoric uncertainty (inherent data noise)
            aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(
                model_output, features
            )
            if aleatoric_uncertainty > 0.5:
                indicators.append(UncertaintyIndicator(
                    uncertainty_type=UncertaintyType.ALEATORIC,
                    value=aleatoric_uncertainty,
                    description="High inherent noise in audio data",
                    contributing_factors=["audio_noise", "compression_artifacts", "recording_quality"]
                ))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Uncertainty analysis failed: {str(e)}")
            return []
    
    def _calculate_prediction_uncertainty(self, model_output: float) -> float:
        """Calculate uncertainty based on proximity to decision boundary."""
        # Higher uncertainty when closer to 0.5 (decision boundary)
        distance_from_boundary = abs(model_output - 0.5)
        # Convert to uncertainty (inverse of distance)
        uncertainty = 1.0 - (distance_from_boundary * 2.0)
        return max(uncertainty, 0.0)
    
    def _calculate_feature_uncertainty(self, features: np.ndarray) -> float:
        """Calculate uncertainty based on feature quality."""
        try:
            if features is None or len(features) == 0:
                return 1.0  # Maximum uncertainty for missing features
            
            # Calculate feature statistics
            feature_variance = np.var(features)
            feature_mean = np.mean(np.abs(features))
            
            # Detect potential issues
            uncertainty_factors = []
            
            # Very low variance might indicate poor features
            if feature_variance < 0.01:
                uncertainty_factors.append(0.3)
            
            # Very high variance might indicate noise
            if feature_variance > 10.0:
                uncertainty_factors.append(0.4)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                uncertainty_factors.append(0.8)
            
            # Very small mean might indicate weak signal
            if feature_mean < 0.001:
                uncertainty_factors.append(0.2)
            
            # Combine uncertainty factors
            if uncertainty_factors:
                return min(sum(uncertainty_factors), 1.0)
            else:
                # Good features have low uncertainty
                return 0.1
                
        except Exception as e:
            logger.error(f"Feature uncertainty calculation failed: {str(e)}")
            return 0.5  # Moderate uncertainty on error
    
    def _calculate_epistemic_uncertainty(
        self, model_metadata: Dict[str, Any], language: str
    ) -> float:
        """Calculate model epistemic uncertainty."""
        try:
            uncertainty_factors = []
            
            # Check model validation accuracy
            validation_accuracy = model_metadata.get('validation_accuracy', 0.5)
            if validation_accuracy < 0.8:
                uncertainty_factors.append((0.8 - validation_accuracy) * 2.0)
            
            # Check training data size (if available)
            training_samples = model_metadata.get('training_samples', 1000)
            if training_samples < 5000:
                uncertainty_factors.append(0.3)
            
            # Language-specific uncertainty
            language_uncertainty = {
                "English": 0.1,
                "Tamil": 0.2,
                "Hindi": 0.15,
                "Malayalam": 0.25,
                "Telugu": 0.2
            }
            uncertainty_factors.append(language_uncertainty.get(language, 0.3))
            
            return min(sum(uncertainty_factors), 1.0)
            
        except Exception as e:
            logger.error(f"Epistemic uncertainty calculation failed: {str(e)}")
            return 0.2  # Default low epistemic uncertainty
    
    def _calculate_stability_uncertainty(self, prediction_history: List[float]) -> float:
        """Calculate uncertainty based on prediction stability."""
        try:
            if len(prediction_history) < 2:
                return 0.0
            
            # Calculate variance in predictions
            prediction_variance = np.var(prediction_history)
            
            # Higher variance indicates higher uncertainty
            # Scale variance to 0-1 range (assuming max reasonable variance is 0.25)
            uncertainty = min(prediction_variance * 4.0, 1.0)
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"Stability uncertainty calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_aleatoric_uncertainty(
        self, model_output: float, features: Optional[np.ndarray]
    ) -> float:
        """Calculate aleatoric (data) uncertainty."""
        try:
            uncertainty_factors = []
            
            # Base aleatoric uncertainty from model output entropy
            # Higher entropy indicates more inherent uncertainty
            if model_output > 0.0 and model_output < 1.0:
                entropy = -(model_output * np.log(model_output) + 
                          (1 - model_output) * np.log(1 - model_output))
                # Normalize entropy (max entropy for binary is log(2))
                normalized_entropy = entropy / np.log(2)
                uncertainty_factors.append(normalized_entropy)
            
            # Feature-based aleatoric uncertainty
            if features is not None:
                # High feature variance might indicate noisy data
                feature_variance = np.var(features)
                if feature_variance > 5.0:
                    uncertainty_factors.append(0.3)
            
            return min(sum(uncertainty_factors), 1.0) if uncertainty_factors else 0.2
            
        except Exception as e:
            logger.error(f"Aleatoric uncertainty calculation failed: {str(e)}")
            return 0.2  # Default moderate aleatoric uncertainty
    
    def _assess_feature_quality(self, features: Optional[np.ndarray]) -> float:
        """Assess the quality of input features."""
        try:
            if features is None or len(features) == 0:
                return 0.0  # No features available
            
            quality_score = 1.0
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                quality_score *= 0.1  # Very poor quality
            
            # Check feature variance (too low or too high is bad)
            feature_variance = np.var(features)
            if feature_variance < 0.001:
                quality_score *= 0.3  # Low variance indicates poor features
            elif feature_variance > 100.0:
                quality_score *= 0.4  # High variance indicates noise
            
            # Check feature range
            feature_range = np.max(features) - np.min(features)
            if feature_range < 0.001:
                quality_score *= 0.2  # No dynamic range
            
            # Check for reasonable feature magnitudes
            feature_mean = np.mean(np.abs(features))
            if feature_mean < 1e-6:
                quality_score *= 0.1  # Features too small
            elif feature_mean > 1e6:
                quality_score *= 0.3  # Features too large
            
            return max(quality_score, 0.0)
            
        except Exception as e:
            logger.error(f"Feature quality assessment failed: {str(e)}")
            return 0.5  # Moderate quality on error
    
    def _calculate_prediction_stability(
        self, model_output: float, prediction_history: Optional[List[float]]
    ) -> float:
        """Calculate prediction stability score."""
        try:
            if prediction_history is None or len(prediction_history) < 2:
                return 0.8  # Assume reasonable stability with no history
            
            # Add current prediction to history for analysis
            all_predictions = prediction_history + [model_output]
            
            # Calculate coefficient of variation
            mean_prediction = np.mean(all_predictions)
            std_prediction = np.std(all_predictions)
            
            if mean_prediction == 0:
                return 0.5  # Moderate stability for zero mean
            
            cv = std_prediction / abs(mean_prediction)
            
            # Convert coefficient of variation to stability score
            # Lower CV means higher stability
            stability = max(1.0 - cv, 0.0)
            
            return stability
            
        except Exception as e:
            logger.error(f"Prediction stability calculation failed: {str(e)}")
            return 0.5  # Moderate stability on error
    
    def _calculate_model_certainty(
        self, model_output: float, model_metadata: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate model's intrinsic certainty about the prediction."""
        try:
            # Base certainty from model output extremity
            extremity = abs(model_output - 0.5) * 2.0
            base_certainty = extremity
            
            # Adjust based on model metadata
            if model_metadata is not None:
                # Model validation accuracy affects certainty
                validation_accuracy = model_metadata.get('validation_accuracy', 0.8)
                accuracy_factor = validation_accuracy
                
                # Model complexity affects certainty
                model_size = model_metadata.get('model_size_bytes', 1000000)
                # Larger models might be more certain (up to a point)
                size_factor = min(model_size / 10000000, 1.2)  # Cap at 1.2x
                
                # Combine factors
                certainty = base_certainty * accuracy_factor * size_factor
            else:
                certainty = base_certainty
            
            return min(certainty, 1.0)
            
        except Exception as e:
            logger.error(f"Model certainty calculation failed: {str(e)}")
            return 0.5  # Moderate certainty on error
    
    def _apply_uncertainty_adjustments(
        self, raw_confidence: float, uncertainty_indicators: List[UncertaintyIndicator]
    ) -> float:
        """Apply uncertainty adjustments to raw confidence."""
        try:
            if not uncertainty_indicators:
                return raw_confidence
            
            # Calculate weighted uncertainty penalty
            total_penalty = 0.0
            
            for indicator in uncertainty_indicators:
                weight = self.uncertainty_weights.get(
                    indicator.uncertainty_type, 0.1
                )
                penalty = indicator.value * weight
                total_penalty += penalty
            
            # Apply penalty (reduce confidence)
            adjusted_confidence = raw_confidence * (1.0 - min(total_penalty, 0.8))
            
            # Ensure minimum confidence
            adjusted_confidence = max(adjusted_confidence, 0.05)
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Uncertainty adjustment failed: {str(e)}")
            return raw_confidence
    
    def _apply_confidence_calibration(
        self, adjusted_confidence: float, model_output: float, language: str
    ) -> Tuple[float, float]:
        """Apply confidence calibration to improve reliability."""
        try:
            # Determine calibration category
            if adjusted_confidence >= 0.9:
                category = "very_high"
            elif adjusted_confidence >= 0.7:
                category = "high"
            elif adjusted_confidence >= 0.5:
                category = "moderate"
            elif adjusted_confidence >= 0.3:
                category = "low"
            else:
                category = "very_low"
            
            # Apply calibration adjustment
            calibration_params = self.calibration_params[category]
            target_confidence = calibration_params["adjustment"]
            
            # Blend original confidence with calibrated target
            blend_factor = 0.3  # How much to blend towards target
            calibrated_confidence = (
                adjusted_confidence * (1 - blend_factor) +
                target_confidence * blend_factor
            )
            
            # Calculate calibration score (how much adjustment was needed)
            calibration_score = 1.0 - abs(adjusted_confidence - calibrated_confidence)
            
            return calibrated_confidence, calibration_score
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {str(e)}")
            return adjusted_confidence, 0.5
    
    def _determine_confidence_level(self, final_confidence: float) -> ConfidenceLevel:
        """Determine human-readable confidence level."""
        if final_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif final_confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif final_confidence >= 0.5:
            return ConfidenceLevel.MODERATE
        elif final_confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _create_fallback_confidence_metrics(self, model_output: float) -> ConfidenceMetrics:
        """Create fallback confidence metrics when calculation fails."""
        try:
            # Basic confidence calculation as fallback
            raw_confidence = min(abs(model_output - 0.5) * 2.0, 1.0)
            
            return ConfidenceMetrics(
                raw_confidence=raw_confidence,
                adjusted_confidence=raw_confidence * 0.8,  # Reduce for uncertainty
                final_confidence=raw_confidence * 0.7,  # Further reduce for safety
                confidence_level=self._determine_confidence_level(raw_confidence * 0.7),
                uncertainty_indicators=[
                    UncertaintyIndicator(
                        uncertainty_type=UncertaintyType.EPISTEMIC,
                        value=0.5,
                        description="Confidence calculation failed, using fallback",
                        contributing_factors=["calculation_error", "fallback_mode"]
                    )
                ],
                prediction_stability=0.5,
                feature_quality_score=0.5,
                model_certainty=0.5,
                calibration_score=0.3
            )
            
        except Exception as e:
            logger.error(f"Fallback confidence metrics creation failed: {str(e)}")
            # Ultimate fallback
            return ConfidenceMetrics(
                raw_confidence=0.5,
                adjusted_confidence=0.4,
                final_confidence=0.3,
                confidence_level=ConfidenceLevel.LOW,
                uncertainty_indicators=[],
                prediction_stability=0.5,
                feature_quality_score=0.5,
                model_certainty=0.5,
                calibration_score=0.3
            )
    
    def get_confidence_explanation(self, metrics: ConfidenceMetrics) -> str:
        """
        Generate human-readable explanation of confidence score.
        
        Args:
            metrics: Confidence metrics to explain
            
        Returns:
            Human-readable explanation string
        """
        try:
            level_descriptions = {
                ConfidenceLevel.VERY_HIGH: "very high confidence",
                ConfidenceLevel.HIGH: "high confidence",
                ConfidenceLevel.MODERATE: "moderate confidence",
                ConfidenceLevel.LOW: "low confidence",
                ConfidenceLevel.VERY_LOW: "very low confidence"
            }
            
            base_description = level_descriptions[metrics.confidence_level]
            
            # Add uncertainty factors if present
            if metrics.uncertainty_indicators:
                uncertainty_types = [
                    indicator.uncertainty_type.value 
                    for indicator in metrics.uncertainty_indicators
                ]
                uncertainty_desc = ", ".join(uncertainty_types)
                
                explanation = (
                    f"The model has {base_description} ({metrics.final_confidence:.2f}) "
                    f"in this prediction. However, there are some uncertainty factors "
                    f"that affect reliability: {uncertainty_desc}."
                )
            else:
                explanation = (
                    f"The model has {base_description} ({metrics.final_confidence:.2f}) "
                    f"in this prediction with no significant uncertainty factors detected."
                )
            
            # Add feature quality information
            if metrics.feature_quality_score < 0.5:
                explanation += " The input audio quality may be affecting prediction reliability."
            elif metrics.feature_quality_score > 0.8:
                explanation += " The input audio quality is excellent, supporting reliable prediction."
            
            # Add stability information
            if metrics.prediction_stability < 0.5:
                explanation += " The prediction shows some instability across evaluations."
            elif metrics.prediction_stability > 0.8:
                explanation += " The prediction is highly stable and consistent."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Confidence explanation generation failed: {str(e)}")
            return f"Confidence score: {metrics.final_confidence:.2f}"