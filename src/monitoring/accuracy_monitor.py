"""
AccuracyMonitor class for real-time accuracy tracking and degradation detection.

This module implements sliding window accuracy calculation, degradation detection algorithms,
and performance metrics collection and storage for the AI voice detection system.
"""

import logging
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
import statistics
import threading
from pathlib import Path
import json

import numpy as np

from src.detection.engine import DetectionResult

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for performance degradation."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PredictionRecord:
    """Record of a single prediction for monitoring."""
    timestamp: datetime
    language: str
    classification: str
    confidence_score: float
    processing_time: float
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    model_version: str = "unknown"
    threshold_used: Optional[float] = None


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a specific time window."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_count: int
    time_window: timedelta
    start_time: datetime
    end_time: datetime
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    processing_times: List[float] = field(default_factory=list)


@dataclass
class DegradationAlert:
    """Alert for accuracy degradation detection."""
    timestamp: datetime
    language: str
    alert_level: AlertLevel
    current_accuracy: float
    baseline_accuracy: float
    degradation_amount: float
    sample_count: int
    time_window: timedelta
    message: str
    additional_info: Dict[str, Any] = field(default_factory=dict)


class AccuracyMonitor:
    """
    Real-time accuracy monitoring with sliding window calculation and degradation detection.
    
    Features:
    - Sliding window accuracy calculation with configurable window sizes
    - Automatic degradation detection with configurable thresholds
    - Per-language accuracy tracking
    - Confidence score distribution analysis
    - Processing time monitoring
    - Persistent storage of metrics and alerts
    """
    
    def __init__(self, settings=None, window_size: int = 1000, 
                 degradation_threshold: float = 0.05, storage_path: Optional[str] = None):
        """
        Initialize the AccuracyMonitor.
        
        Args:
            settings: Application settings object
            window_size: Size of sliding window for accuracy calculation
            degradation_threshold: Threshold for degradation detection (e.g., 0.05 = 5% drop)
            storage_path: Path for persistent storage of metrics
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        
        # Storage configuration
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(settings.ml.models_base_path) / "monitoring"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe data structures
        self._lock = threading.RLock()
        
        # Sliding window storage: language -> deque of PredictionRecord
        self.prediction_windows: Dict[str, Deque[PredictionRecord]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        
        # Baseline accuracy storage: language -> accuracy
        self.baseline_accuracies: Dict[str, float] = {}
        
        # Alert history: language -> list of DegradationAlert
        self.alert_history: Dict[str, List[DegradationAlert]] = defaultdict(list)
        
        # Performance tracking
        self.total_predictions = 0
        self.start_time = datetime.now()
        
        # Load existing baselines and metrics
        self._load_persistent_data()
        
        logger.info(f"AccuracyMonitor initialized with window_size={window_size}, "
                   f"degradation_threshold={degradation_threshold}")
    
    def track_prediction(self, result: DetectionResult, language: str, 
                        ground_truth: Optional[str] = None) -> None:
        """
        Track a prediction result for accuracy monitoring.
        
        Args:
            result: Detection result from the engine
            language: Language of the prediction
            ground_truth: Ground truth label if available ("AI_GENERATED" or "HUMAN")
        """
        try:
            with self._lock:
                # Create prediction record
                record = PredictionRecord(
                    timestamp=datetime.now(),
                    language=language,
                    classification=result.classification,
                    confidence_score=result.confidence_score,
                    processing_time=result.processing_time,
                    ground_truth=ground_truth,
                    model_version=result.model_version,
                    threshold_used=getattr(result, 'threshold_used', None)
                )
                
                # Calculate correctness if ground truth is available
                if ground_truth:
                    record.is_correct = (result.classification == ground_truth)
                
                # Add to sliding window
                self.prediction_windows[language].append(record)
                self.total_predictions += 1
                
                # Check for degradation if we have ground truth
                if ground_truth and len(self.prediction_windows[language]) >= 10:
                    self._check_degradation(language)
                
                logger.debug(f"Tracked prediction for {language}: {result.classification} "
                           f"(confidence: {result.confidence_score:.3f})")
                
        except Exception as e:
            logger.error(f"Failed to track prediction for {language}: {str(e)}")
    
    def calculate_running_accuracy(self, language: str, window_size: Optional[int] = None) -> float:
        """
        Calculate running accuracy for a specific language using sliding window.
        
        Args:
            language: Target language
            window_size: Window size for calculation (uses default if None)
            
        Returns:
            Running accuracy (0.0 to 1.0) or -1.0 if insufficient data
        """
        try:
            with self._lock:
                if language not in self.prediction_windows:
                    return -1.0
                
                window = self.prediction_windows[language]
                effective_window_size = window_size or self.window_size
                
                # Get recent predictions with ground truth
                recent_predictions = list(window)[-effective_window_size:]
                labeled_predictions = [p for p in recent_predictions if p.ground_truth is not None]
                
                if not labeled_predictions:
                    return -1.0
                
                # Calculate accuracy
                correct_predictions = sum(1 for p in labeled_predictions if p.is_correct)
                accuracy = correct_predictions / len(labeled_predictions)
                
                logger.debug(f"Running accuracy for {language}: {accuracy:.3f} "
                           f"({correct_predictions}/{len(labeled_predictions)} samples)")
                
                return accuracy
                
        except Exception as e:
            logger.error(f"Failed to calculate running accuracy for {language}: {str(e)}")
            return -1.0
    
    def detect_accuracy_degradation(self, language: str) -> bool:
        """
        Detect if accuracy has degraded below acceptable thresholds.
        
        Args:
            language: Target language
            
        Returns:
            True if degradation detected, False otherwise
        """
        try:
            with self._lock:
                current_accuracy = self.calculate_running_accuracy(language)
                
                if current_accuracy < 0:
                    return False  # Insufficient data
                
                baseline_accuracy = self.baseline_accuracies.get(language)
                
                if baseline_accuracy is None:
                    # Set current accuracy as baseline if none exists
                    self.baseline_accuracies[language] = current_accuracy
                    self._save_baselines()
                    return False
                
                # Check for degradation
                degradation = baseline_accuracy - current_accuracy
                is_degraded = degradation > self.degradation_threshold
                
                if is_degraded:
                    logger.warning(f"Accuracy degradation detected for {language}: "
                                 f"{current_accuracy:.3f} vs baseline {baseline_accuracy:.3f} "
                                 f"(degradation: {degradation:.3f})")
                
                return is_degraded
                
        except Exception as e:
            logger.error(f"Failed to detect degradation for {language}: {str(e)}")
            return False
    
    def get_accuracy_metrics(self, language: str, time_window: Optional[timedelta] = None) -> Optional[AccuracyMetrics]:
        """
        Get comprehensive accuracy metrics for a language within a time window.
        
        Args:
            language: Target language
            time_window: Time window for metrics (default: last hour)
            
        Returns:
            AccuracyMetrics object or None if insufficient data
        """
        try:
            with self._lock:
                if language not in self.prediction_windows:
                    return None
                
                window = self.prediction_windows[language]
                if not window:
                    return None
                
                # Default to last hour if no time window specified
                if time_window is None:
                    time_window = timedelta(hours=1)
                
                # Filter predictions within time window
                cutoff_time = datetime.now() - time_window
                recent_predictions = [p for p in window if p.timestamp >= cutoff_time]
                labeled_predictions = [p for p in recent_predictions if p.ground_truth is not None]
                
                if not labeled_predictions:
                    return None
                
                # Calculate metrics
                correct_predictions = [p for p in labeled_predictions if p.is_correct]
                
                # Basic metrics
                accuracy = len(correct_predictions) / len(labeled_predictions)
                
                # Precision and recall for binary classification
                ai_predictions = [p for p in labeled_predictions if p.classification == "AI_GENERATED"]
                human_predictions = [p for p in labeled_predictions if p.classification == "HUMAN"]
                
                ai_correct = [p for p in ai_predictions if p.is_correct]
                human_correct = [p for p in human_predictions if p.is_correct]
                
                ai_ground_truth = [p for p in labeled_predictions if p.ground_truth == "AI_GENERATED"]
                human_ground_truth = [p for p in labeled_predictions if p.ground_truth == "HUMAN"]
                
                # Calculate precision and recall
                precision = len(ai_correct) / len(ai_predictions) if ai_predictions else 0.0
                recall = len(ai_correct) / len(ai_ground_truth) if ai_ground_truth else 0.0
                
                # F1 score
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Confidence distribution
                confidence_bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
                for p in recent_predictions:
                    if p.confidence_score < 0.2:
                        confidence_bins["0.0-0.2"] += 1
                    elif p.confidence_score < 0.4:
                        confidence_bins["0.2-0.4"] += 1
                    elif p.confidence_score < 0.6:
                        confidence_bins["0.4-0.6"] += 1
                    elif p.confidence_score < 0.8:
                        confidence_bins["0.6-0.8"] += 1
                    else:
                        confidence_bins["0.8-1.0"] += 1
                
                # Processing times
                processing_times = [p.processing_time for p in recent_predictions]
                
                # Create metrics object
                start_time = min(p.timestamp for p in recent_predictions)
                end_time = max(p.timestamp for p in recent_predictions)
                
                metrics = AccuracyMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    sample_count=len(labeled_predictions),
                    time_window=time_window,
                    start_time=start_time,
                    end_time=end_time,
                    confidence_distribution=confidence_bins,
                    processing_times=processing_times
                )
                
                logger.debug(f"Generated accuracy metrics for {language}: "
                           f"accuracy={accuracy:.3f}, samples={len(labeled_predictions)}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get accuracy metrics for {language}: {str(e)}")
            return None
    
    def get_all_language_metrics(self, time_window: Optional[timedelta] = None) -> Dict[str, AccuracyMetrics]:
        """
        Get accuracy metrics for all monitored languages.
        
        Args:
            time_window: Time window for metrics
            
        Returns:
            Dictionary mapping language to AccuracyMetrics
        """
        try:
            with self._lock:
                all_metrics = {}
                
                for language in self.prediction_windows.keys():
                    metrics = self.get_accuracy_metrics(language, time_window)
                    if metrics:
                        all_metrics[language] = metrics
                
                return all_metrics
                
        except Exception as e:
            logger.error(f"Failed to get all language metrics: {str(e)}")
            return {}
    
    def set_baseline_accuracy(self, language: str, accuracy: float) -> None:
        """
        Set baseline accuracy for a language.
        
        Args:
            language: Target language
            accuracy: Baseline accuracy (0.0 to 1.0)
        """
        try:
            with self._lock:
                if not (0.0 <= accuracy <= 1.0):
                    raise ValueError(f"Accuracy must be between 0.0 and 1.0, got {accuracy}")
                
                self.baseline_accuracies[language] = accuracy
                self._save_baselines()
                
                logger.info(f"Set baseline accuracy for {language}: {accuracy:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to set baseline accuracy for {language}: {str(e)}")
            raise
    
    def get_baseline_accuracy(self, language: str) -> Optional[float]:
        """
        Get baseline accuracy for a language.
        
        Args:
            language: Target language
            
        Returns:
            Baseline accuracy or None if not set
        """
        with self._lock:
            return self.baseline_accuracies.get(language)
    
    def get_recent_alerts(self, language: Optional[str] = None, 
                         hours: int = 24) -> List[DegradationAlert]:
        """
        Get recent degradation alerts.
        
        Args:
            language: Specific language (None for all languages)
            hours: Number of hours to look back
            
        Returns:
            List of recent DegradationAlert objects
        """
        try:
            with self._lock:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_alerts = []
                
                if language:
                    # Get alerts for specific language
                    alerts = self.alert_history.get(language, [])
                    recent_alerts.extend([a for a in alerts if a.timestamp >= cutoff_time])
                else:
                    # Get alerts for all languages
                    for lang_alerts in self.alert_history.values():
                        recent_alerts.extend([a for a in lang_alerts if a.timestamp >= cutoff_time])
                
                # Sort by timestamp (most recent first)
                recent_alerts.sort(key=lambda x: x.timestamp, reverse=True)
                
                return recent_alerts
                
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {str(e)}")
            return []
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.
        
        Returns:
            Dictionary containing monitoring status and statistics
        """
        try:
            with self._lock:
                summary = {
                    "total_predictions": self.total_predictions,
                    "monitoring_duration": str(datetime.now() - self.start_time),
                    "monitored_languages": list(self.prediction_windows.keys()),
                    "window_size": self.window_size,
                    "degradation_threshold": self.degradation_threshold,
                    "baseline_accuracies": dict(self.baseline_accuracies),
                    "current_accuracies": {},
                    "recent_alerts_count": 0,
                    "storage_path": str(self.storage_path)
                }
                
                # Calculate current accuracies
                for language in self.prediction_windows.keys():
                    current_accuracy = self.calculate_running_accuracy(language)
                    if current_accuracy >= 0:
                        summary["current_accuracies"][language] = current_accuracy
                
                # Count recent alerts (last 24 hours)
                recent_alerts = self.get_recent_alerts(hours=24)
                summary["recent_alerts_count"] = len(recent_alerts)
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {str(e)}")
            return {"error": str(e)}
    
    def clear_data(self, language: Optional[str] = None, keep_baselines: bool = True) -> None:
        """
        Clear monitoring data.
        
        Args:
            language: Specific language to clear (None for all)
            keep_baselines: Whether to keep baseline accuracies
        """
        try:
            with self._lock:
                if language:
                    # Clear specific language
                    if language in self.prediction_windows:
                        self.prediction_windows[language].clear()
                    if language in self.alert_history:
                        self.alert_history[language].clear()
                    if not keep_baselines and language in self.baseline_accuracies:
                        del self.baseline_accuracies[language]
                else:
                    # Clear all data
                    self.prediction_windows.clear()
                    self.alert_history.clear()
                    if not keep_baselines:
                        self.baseline_accuracies.clear()
                    self.total_predictions = 0
                    self.start_time = datetime.now()
                
                if not keep_baselines:
                    self._save_baselines()
                
                logger.info(f"Cleared monitoring data for {language or 'all languages'}")
                
        except Exception as e:
            logger.error(f"Failed to clear monitoring data: {str(e)}")
            raise
    
    def _check_degradation(self, language: str) -> None:
        """
        Internal method to check for accuracy degradation and generate alerts.
        
        Args:
            language: Target language
        """
        try:
            current_accuracy = self.calculate_running_accuracy(language)
            baseline_accuracy = self.baseline_accuracies.get(language)
            
            if current_accuracy < 0 or baseline_accuracy is None:
                return
            
            degradation = baseline_accuracy - current_accuracy
            
            if degradation > self.degradation_threshold:
                # Determine alert level
                if degradation > self.degradation_threshold * 3:
                    alert_level = AlertLevel.CRITICAL
                elif degradation > self.degradation_threshold * 2:
                    alert_level = AlertLevel.WARNING
                else:
                    alert_level = AlertLevel.INFO
                
                # Create alert
                alert = DegradationAlert(
                    timestamp=datetime.now(),
                    language=language,
                    alert_level=alert_level,
                    current_accuracy=current_accuracy,
                    baseline_accuracy=baseline_accuracy,
                    degradation_amount=degradation,
                    sample_count=len([p for p in self.prediction_windows[language] if p.ground_truth]),
                    time_window=timedelta(seconds=self.window_size * 10),  # Approximate
                    message=f"Accuracy degradation detected for {language}: "
                           f"{current_accuracy:.3f} vs baseline {baseline_accuracy:.3f} "
                           f"(degradation: {degradation:.3f})",
                    additional_info={
                        "window_size": self.window_size,
                        "degradation_threshold": self.degradation_threshold
                    }
                )
                
                # Store alert
                self.alert_history[language].append(alert)
                
                # Keep only recent alerts (last 30 days)
                cutoff_time = datetime.now() - timedelta(days=30)
                self.alert_history[language] = [
                    a for a in self.alert_history[language] if a.timestamp >= cutoff_time
                ]
                
                # Log alert
                logger.warning(f"[{alert_level.value.upper()}] {alert.message}")
                
        except Exception as e:
            logger.error(f"Failed to check degradation for {language}: {str(e)}")
    
    def _load_persistent_data(self) -> None:
        """Load persistent monitoring data from storage."""
        try:
            baselines_file = self.storage_path / "baselines.json"
            
            if baselines_file.exists():
                with open(baselines_file, 'r') as f:
                    self.baseline_accuracies = json.load(f)
                logger.info(f"Loaded {len(self.baseline_accuracies)} baseline accuracies")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent data: {str(e)}")
    
    def _save_baselines(self) -> None:
        """Save baseline accuracies to persistent storage."""
        try:
            baselines_file = self.storage_path / "baselines.json"
            
            with open(baselines_file, 'w') as f:
                json.dump(self.baseline_accuracies, f, indent=2)
            
            logger.debug(f"Saved {len(self.baseline_accuracies)} baseline accuracies")
            
        except Exception as e:
            logger.error(f"Failed to save baselines: {str(e)}")