"""
Threshold Optimization Module

This module implements threshold optimization for AI voice detection classifiers,
calculating optimal classification thresholds per language and model combination
to maximize F1-score and overall accuracy.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class ThresholdConfig(BaseModel):
    """Configuration for optimized classification thresholds."""
    
    language: str
    model_version: str
    threshold: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    last_updated: datetime
    validation_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    optimization_method: str = "f1_score"


class OptimizationResult(BaseModel):
    """Result of threshold optimization process."""
    
    language: str
    original_threshold: float
    optimized_threshold: float
    improvement: float  # F1-score improvement
    metrics: Dict[str, float]
    sample_count: int
    optimization_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ValidationData:
    """Container for validation data used in threshold optimization."""
    
    predictions: np.ndarray  # Model predictions (probabilities)
    labels: np.ndarray      # Ground truth labels (0 or 1)
    language: str
    model_version: str
    
    def __post_init__(self):
        """Validate data consistency."""
        if len(self.predictions) != len(self.labels):
            raise ValueError("Predictions and labels must have the same length")
        
        if not all(0 <= p <= 1 for p in self.predictions):
            raise ValueError("Predictions must be probabilities between 0 and 1")
        
        if not all(l in [0, 1] for l in self.labels):
            raise ValueError("Labels must be binary (0 or 1)")


class ThresholdOptimizer:
    """
    Threshold optimization system for AI voice detection classifiers.
    
    Calculates and maintains optimal classification thresholds per language
    and model combination to maximize F1-score and overall accuracy.
    """
    
    def __init__(self, settings=None, storage_path: Optional[str] = None):
        """
        Initialize the ThresholdOptimizer.
        
        Args:
            settings: Application settings object
            storage_path: Path to store threshold configurations
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.storage_path = Path(storage_path or "models/thresholds")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded threshold configurations
        self.threshold_cache: Dict[str, ThresholdConfig] = {}
        
        # Default thresholds for fallback
        self.default_thresholds = {
            "Tamil": 0.5,
            "English": 0.5,
            "Hindi": 0.5,
            "Malayalam": 0.5,
            "Telugu": 0.5
        }
        
        # Optimization parameters
        self.min_samples_required = 50  # Minimum samples needed for optimization
        self.threshold_search_steps = 100  # Number of threshold values to test
        self.cache_expiry_hours = 24  # Hours before cache expires
        
        logger.info(f"ThresholdOptimizer initialized with storage path: {self.storage_path}")
    
    def calculate_optimal_threshold(self, language: str, validation_data: ValidationData) -> OptimizationResult:
        """
        Calculate optimal threshold for a language using validation data.
        
        Args:
            language: Target language
            validation_data: Validation data containing predictions and labels
            
        Returns:
            OptimizationResult with optimization details
        """
        start_time = time.time()
        
        try:
            logger.info(f"Calculating optimal threshold for {language} with {len(validation_data.predictions)} samples")
            
            # Validate minimum sample requirement
            if len(validation_data.predictions) < self.min_samples_required:
                raise ValueError(f"Insufficient samples: {len(validation_data.predictions)} < {self.min_samples_required}")
            
            # Get current threshold for comparison
            current_threshold = self.get_threshold(language)
            
            # Calculate optimal threshold using F1-score maximization
            optimal_threshold, best_metrics = self.optimize_for_f1_score(
                validation_data.predictions, 
                validation_data.labels
            )
            
            # Calculate improvement
            current_f1 = self._calculate_f1_at_threshold(
                validation_data.predictions, 
                validation_data.labels, 
                current_threshold
            )
            improvement = best_metrics['f1_score'] - current_f1
            
            optimization_time = time.time() - start_time
            
            result = OptimizationResult(
                language=language,
                original_threshold=current_threshold,
                optimized_threshold=optimal_threshold,
                improvement=improvement,
                metrics=best_metrics,
                sample_count=len(validation_data.predictions),
                optimization_time=optimization_time,
                success=True
            )
            
            logger.info(
                f"Threshold optimization completed for {language}: "
                f"{current_threshold:.3f} -> {optimal_threshold:.3f} "
                f"(F1 improvement: {improvement:+.3f})"
            )
            
            return result
            
        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(f"Threshold optimization failed for {language}: {str(e)}")
            
            return OptimizationResult(
                language=language,
                original_threshold=self.get_threshold(language),
                optimized_threshold=self.get_threshold(language),
                improvement=0.0,
                metrics={},
                sample_count=len(validation_data.predictions) if validation_data else 0,
                optimization_time=optimization_time,
                success=False,
                error_message=str(e)
            )
    
    def optimize_for_f1_score(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold that maximizes F1-score.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels (0 or 1)
            
        Returns:
            Tuple of (optimal_threshold, metrics_dict)
        """
        try:
            # Generate threshold candidates
            thresholds = np.linspace(0.01, 0.99, self.threshold_search_steps)
            
            best_f1 = 0.0
            best_threshold = 0.5
            best_metrics = {}
            
            # Test each threshold
            for threshold in thresholds:
                # Convert predictions to binary classifications
                binary_predictions = (predictions >= threshold).astype(int)
                
                # Calculate metrics
                precision = precision_score(labels, binary_predictions, zero_division=0)
                recall = recall_score(labels, binary_predictions, zero_division=0)
                f1 = f1_score(labels, binary_predictions, zero_division=0)
                
                # Update best threshold if F1-score improved
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'accuracy': np.mean(binary_predictions == labels)
                    }
            
            # Add AUC-ROC if possible
            try:
                auc_roc = roc_auc_score(labels, predictions)
                best_metrics['auc_roc'] = auc_roc
            except Exception as auc_error:
                logger.warning(f"Could not calculate AUC-ROC: {auc_error}")
                best_metrics['auc_roc'] = None
            
            logger.debug(f"Optimal threshold: {best_threshold:.3f} with F1-score: {best_f1:.3f}")
            return best_threshold, best_metrics
            
        except Exception as e:
            logger.error(f"F1-score optimization failed: {str(e)}")
            # Return default threshold with basic metrics
            return 0.5, {
                'threshold': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5,
                'accuracy': 0.5,
                'auc_roc': None
            }
    
    def get_threshold(self, language: str, model_version: Optional[str] = None) -> float:
        """
        Get optimized threshold for a language.
        
        Args:
            language: Target language
            model_version: Specific model version (optional)
            
        Returns:
            Optimized threshold value
        """
        try:
            # Try to load from cache first
            cache_key = f"{language}_{model_version or 'default'}"
            
            if cache_key in self.threshold_cache:
                config = self.threshold_cache[cache_key]
                
                # Check if cache is still valid
                if self._is_cache_valid(config):
                    logger.debug(f"Using cached threshold for {language}: {config.threshold}")
                    return config.threshold
                else:
                    # Remove expired cache entry
                    del self.threshold_cache[cache_key]
            
            # Try to load from storage
            config = self._load_threshold_config(language, model_version)
            if config:
                # Cache the loaded configuration
                self.threshold_cache[cache_key] = config
                logger.debug(f"Loaded threshold for {language}: {config.threshold}")
                return config.threshold
            
            # Fall back to default threshold
            default_threshold = self.default_thresholds.get(language, 0.5)
            logger.debug(f"Using default threshold for {language}: {default_threshold}")
            return default_threshold
            
        except Exception as e:
            logger.error(f"Failed to get threshold for {language}: {str(e)}")
            return self.default_thresholds.get(language, 0.5)
    
    def update_thresholds(self, language: str, validation_data: ValidationData, 
                         model_version: Optional[str] = None) -> OptimizationResult:
        """
        Update thresholds for a language using new validation data.
        
        Args:
            language: Target language
            validation_data: New validation data
            model_version: Specific model version (optional)
            
        Returns:
            OptimizationResult with update details
        """
        try:
            logger.info(f"Updating thresholds for {language} with {len(validation_data.predictions)} new samples")
            
            # Calculate optimal threshold
            result = self.calculate_optimal_threshold(language, validation_data)
            
            if result.success and result.improvement > 0.001:  # Only update if meaningful improvement
                # Create threshold configuration
                config = ThresholdConfig(
                    language=language,
                    model_version=model_version or validation_data.model_version or "default",
                    threshold=result.optimized_threshold,
                    precision=result.metrics.get('precision', 0.0),
                    recall=result.metrics.get('recall', 0.0),
                    f1_score=result.metrics.get('f1_score', 0.0),
                    auc_roc=result.metrics.get('auc_roc'),
                    sample_count=result.sample_count,
                    last_updated=datetime.now(),
                    validation_accuracy=result.metrics.get('accuracy', 0.0),
                    optimization_method="f1_score"
                )
                
                # Save configuration
                self._save_threshold_config(config)
                
                # Update cache
                cache_key = f"{language}_{config.model_version}"
                self.threshold_cache[cache_key] = config
                
                logger.info(f"Updated threshold for {language}: {result.optimized_threshold:.3f}")
                logger.debug(f"Cache key used: {cache_key}")
            else:
                logger.info(f"No significant improvement for {language}, keeping current threshold")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update thresholds for {language}: {str(e)}")
            return OptimizationResult(
                language=language,
                original_threshold=self.get_threshold(language),
                optimized_threshold=self.get_threshold(language),
                improvement=0.0,
                metrics={},
                sample_count=0,
                optimization_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def get_all_thresholds(self) -> Dict[str, ThresholdConfig]:
        """
        Get all available threshold configurations.
        
        Returns:
            Dictionary mapping language to threshold configuration
        """
        try:
            all_configs = {}
            
            # Load all stored configurations
            for threshold_file in self.storage_path.glob("*.json"):
                try:
                    with open(threshold_file, 'r') as f:
                        data = json.load(f)
                    
                    config = ThresholdConfig(**data)
                    all_configs[config.language] = config
                    
                except Exception as e:
                    logger.warning(f"Failed to load threshold config from {threshold_file}: {e}")
            
            # Add default configurations for missing languages
            for language in self.default_thresholds:
                if language not in all_configs:
                    all_configs[language] = ThresholdConfig(
                        language=language,
                        model_version="default",
                        threshold=self.default_thresholds[language],
                        precision=0.5,
                        recall=0.5,
                        f1_score=0.5,
                        sample_count=0,
                        last_updated=datetime.now(),
                        optimization_method="default"
                    )
            
            return all_configs
            
        except Exception as e:
            logger.error(f"Failed to get all thresholds: {str(e)}")
            return {}
    
    def reset_threshold(self, language: str, model_version: Optional[str] = None):
        """
        Reset threshold to default value for a language.
        
        Args:
            language: Target language
            model_version: Specific model version (optional)
        """
        try:
            # Remove from cache
            cache_key = f"{language}_{model_version or 'default'}"
            if cache_key in self.threshold_cache:
                del self.threshold_cache[cache_key]
            
            # Remove stored configuration
            filename = self._get_config_filename(language, model_version)
            config_path = self.storage_path / filename
            if config_path.exists():
                config_path.unlink()
                logger.info(f"Reset threshold for {language} to default")
            
        except Exception as e:
            logger.error(f"Failed to reset threshold for {language}: {str(e)}")
    
    def get_optimization_history(self, language: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get optimization history for a language.
        
        Args:
            language: Target language
            days: Number of days of history to retrieve
            
        Returns:
            List of optimization records
        """
        try:
            history_file = self.storage_path / f"{language}_history.json"
            
            if not history_file.exists():
                return []
            
            with open(history_file, 'r') as f:
                all_history = json.load(f)
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_history = [
                record for record in all_history
                if datetime.fromisoformat(record['timestamp']) >= cutoff_date
            ]
            
            return recent_history
            
        except Exception as e:
            logger.error(f"Failed to get optimization history for {language}: {str(e)}")
            return []
    
    def _calculate_f1_at_threshold(self, predictions: np.ndarray, labels: np.ndarray, threshold: float) -> float:
        """Calculate F1-score at a specific threshold."""
        try:
            binary_predictions = (predictions >= threshold).astype(int)
            return f1_score(labels, binary_predictions, zero_division=0)
        except Exception:
            return 0.0
    
    def _is_cache_valid(self, config: ThresholdConfig) -> bool:
        """Check if cached configuration is still valid."""
        try:
            expiry_time = config.last_updated + timedelta(hours=self.cache_expiry_hours)
            return datetime.now() < expiry_time
        except Exception:
            return False
    
    def _load_threshold_config(self, language: str, model_version: Optional[str] = None) -> Optional[ThresholdConfig]:
        """Load threshold configuration from storage."""
        try:
            filename = self._get_config_filename(language, model_version)
            config_path = self.storage_path / filename
            
            if not config_path.exists():
                return None
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            return ThresholdConfig(**data)
            
        except Exception as e:
            logger.warning(f"Failed to load threshold config for {language}: {e}")
            return None
    
    def _save_threshold_config(self, config: ThresholdConfig):
        """Save threshold configuration to storage."""
        try:
            filename = self._get_config_filename(config.language, config.model_version)
            config_path = self.storage_path / filename
            
            # Convert to dictionary for JSON serialization
            config_dict = config.model_dump()
            config_dict['last_updated'] = config.last_updated.isoformat()
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Also save to history
            self._save_to_history(config)
            
            logger.debug(f"Saved threshold config for {config.language}")
            
        except Exception as e:
            logger.error(f"Failed to save threshold config for {config.language}: {e}")
    
    def _save_to_history(self, config: ThresholdConfig):
        """Save configuration to optimization history."""
        try:
            history_file = self.storage_path / f"{config.language}_history.json"
            
            # Load existing history
            history = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new record
            record = {
                'timestamp': config.last_updated.isoformat(),
                'threshold': config.threshold,
                'f1_score': config.f1_score,
                'precision': config.precision,
                'recall': config.recall,
                'sample_count': config.sample_count,
                'model_version': config.model_version
            }
            
            history.append(record)
            
            # Keep only last 100 records
            history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save optimization history: {e}")
    
    def _get_config_filename(self, language: str, model_version: Optional[str] = None) -> str:
        """Generate configuration filename."""
        version_suffix = f"_{model_version}" if model_version and model_version != "default" else ""
        return f"{language.lower()}_threshold{version_suffix}.json"