"""
PerformanceReport generation system for comprehensive reporting with per-language metrics.

This module implements comprehensive reporting with per-language metrics, confidence distribution
analysis, and processing time and resource utilization tracking.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics
import threading

import numpy as np

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .accuracy_monitor import AccuracyMonitor, AccuracyMetrics, DegradationAlert

logger = logging.getLogger(__name__)


@dataclass
class LanguagePerformance:
    """Performance metrics for a specific language."""
    language: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_count: int
    avg_confidence: float
    avg_processing_time: float
    confidence_distribution: Dict[str, int]
    error_rate: float
    threshold_used: Optional[float] = None
    is_threshold_optimized: bool = False


@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_bytes: int
    memory_usage_percent: float
    disk_usage_bytes: int
    disk_free_bytes: int
    active_threads: int
    model_cache_size_mb: float = 0.0
    gpu_usage_percent: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report for the AI voice detection system."""
    
    # Report metadata
    report_id: str
    generation_timestamp: datetime
    time_period_start: datetime
    time_period_end: datetime
    time_period_duration: timedelta
    
    # Overall system metrics
    total_predictions: int
    overall_accuracy: float
    overall_precision: float
    overall_recall: float
    overall_f1_score: float
    
    # Per-language performance
    per_language_performance: Dict[str, LanguagePerformance]
    
    # Confidence and processing metrics
    overall_confidence_distribution: Dict[str, int]
    avg_processing_time: float
    processing_time_percentiles: Dict[str, float]  # P50, P95, P99
    
    # Error and degradation tracking
    error_rates: Dict[str, float]
    degradation_alerts: List[DegradationAlert]
    
    # Resource utilization
    resource_metrics: List[SystemResourceMetrics]
    avg_cpu_usage: float
    avg_memory_usage: float
    peak_memory_usage: float
    
    # Model and threshold information
    model_versions: Dict[str, str]
    threshold_optimization_status: Dict[str, bool]
    
    # Additional insights
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceReportGenerator:
    """
    Generator for comprehensive performance reports with per-language metrics,
    confidence distribution analysis, and resource utilization tracking.
    """
    
    def __init__(self, accuracy_monitor: AccuracyMonitor, settings=None):
        """
        Initialize the PerformanceReportGenerator.
        
        Args:
            accuracy_monitor: AccuracyMonitor instance for data collection
            settings: Application settings object
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.accuracy_monitor = accuracy_monitor
        
        # Resource monitoring
        self.resource_history: List[SystemResourceMetrics] = []
        self._resource_lock = threading.RLock()
        
        # Start resource monitoring thread
        self._monitoring_active = True
        self._resource_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._resource_thread.start()
        
        logger.info("PerformanceReportGenerator initialized with resource monitoring")
    
    def generate_performance_report(self, time_period: timedelta, 
                                  include_insights: bool = True) -> PerformanceReport:
        """
        Generate comprehensive performance report for the specified time period.
        
        Args:
            time_period: Time period for the report (e.g., last 24 hours)
            include_insights: Whether to include automated insights and recommendations
            
        Returns:
            PerformanceReport with comprehensive metrics and analysis
        """
        try:
            end_time = datetime.now()
            start_time = end_time - time_period
            
            logger.info(f"Generating performance report for period: {start_time} to {end_time}")
            
            # Generate unique report ID
            report_id = f"perf_report_{int(time.time())}"
            
            # Collect per-language metrics
            language_metrics = self.accuracy_monitor.get_all_language_metrics(time_period)
            per_language_performance = self._generate_language_performance(language_metrics)
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(per_language_performance)
            
            # Get confidence distribution
            confidence_distribution = self._calculate_overall_confidence_distribution(language_metrics)
            
            # Get processing time metrics
            processing_metrics = self._calculate_processing_time_metrics(language_metrics)
            
            # Get error rates
            error_rates = self._calculate_error_rates(language_metrics)
            
            # Get degradation alerts
            degradation_alerts = self.accuracy_monitor.get_recent_alerts(
                hours=int(time_period.total_seconds() / 3600)
            )
            
            # Get resource metrics
            resource_metrics = self._get_resource_metrics_for_period(start_time, end_time)
            resource_summary = self._calculate_resource_summary(resource_metrics)
            
            # Get model and threshold information
            model_info = self._get_model_information()
            
            # Create performance report
            report = PerformanceReport(
                report_id=report_id,
                generation_timestamp=datetime.now(),
                time_period_start=start_time,
                time_period_end=end_time,
                time_period_duration=time_period,
                
                total_predictions=overall_metrics['total_predictions'],
                overall_accuracy=overall_metrics['accuracy'],
                overall_precision=overall_metrics['precision'],
                overall_recall=overall_metrics['recall'],
                overall_f1_score=overall_metrics['f1_score'],
                
                per_language_performance=per_language_performance,
                
                overall_confidence_distribution=confidence_distribution,
                avg_processing_time=processing_metrics['avg_time'],
                processing_time_percentiles=processing_metrics['percentiles'],
                
                error_rates=error_rates,
                degradation_alerts=degradation_alerts,
                
                resource_metrics=resource_metrics,
                avg_cpu_usage=resource_summary['avg_cpu'],
                avg_memory_usage=resource_summary['avg_memory'],
                peak_memory_usage=resource_summary['peak_memory'],
                
                model_versions=model_info['versions'],
                threshold_optimization_status=model_info['threshold_status']
            )
            
            # Generate insights and recommendations
            if include_insights:
                report.insights = self._generate_insights(report)
                report.recommendations = self._generate_recommendations(report)
            
            logger.info(f"Generated performance report {report_id} with "
                       f"{len(per_language_performance)} languages, "
                       f"{overall_metrics['total_predictions']} predictions")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {str(e)}")
            raise
    
    def save_report(self, report: PerformanceReport, output_path: Optional[str] = None) -> str:
        """
        Save performance report to file.
        
        Args:
            report: PerformanceReport to save
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to saved report file
        """
        try:
            if output_path is None:
                reports_dir = self.accuracy_monitor.storage_path / "reports"
                reports_dir.mkdir(exist_ok=True)
                output_path = reports_dir / f"{report.report_id}.json"
            else:
                output_path = Path(output_path)
            
            # Convert report to dictionary for JSON serialization
            report_dict = self._report_to_dict(report)
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Saved performance report to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {str(e)}")
            raise
    
    def load_report(self, report_path: str) -> PerformanceReport:
        """
        Load performance report from file.
        
        Args:
            report_path: Path to report file
            
        Returns:
            PerformanceReport object
        """
        try:
            with open(report_path, 'r') as f:
                report_dict = json.load(f)
            
            report = self._dict_to_report(report_dict)
            
            logger.info(f"Loaded performance report from {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to load performance report: {str(e)}")
            raise
    
    def get_recent_reports(self, days: int = 7) -> List[str]:
        """
        Get list of recent report files.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of report file paths
        """
        try:
            reports_dir = self.accuracy_monitor.storage_path / "reports"
            
            if not reports_dir.exists():
                return []
            
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_reports = []
            
            for report_file in reports_dir.glob("*.json"):
                if report_file.stat().st_mtime > cutoff_time.timestamp():
                    recent_reports.append(str(report_file))
            
            # Sort by modification time (newest first)
            recent_reports.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            
            return recent_reports
            
        except Exception as e:
            logger.error(f"Failed to get recent reports: {str(e)}")
            return []
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring thread."""
        self._monitoring_active = False
        if self._resource_thread.is_alive():
            self._resource_thread.join(timeout=5.0)
        logger.info("Stopped performance monitoring")
    
    def _generate_language_performance(self, language_metrics: Dict[str, AccuracyMetrics]) -> Dict[str, LanguagePerformance]:
        """Generate per-language performance metrics."""
        try:
            performance = {}
            
            for language, metrics in language_metrics.items():
                # Calculate average confidence and processing time
                with self.accuracy_monitor._lock:
                    window = self.accuracy_monitor.prediction_windows.get(language, [])
                    recent_predictions = [p for p in window 
                                        if p.timestamp >= metrics.start_time]
                    
                    if recent_predictions:
                        avg_confidence = statistics.mean(p.confidence_score for p in recent_predictions)
                        avg_processing_time = statistics.mean(p.processing_time for p in recent_predictions)
                        
                        # Calculate error rate (predictions without ground truth or incorrect)
                        total_predictions = len(recent_predictions)
                        labeled_predictions = [p for p in recent_predictions if p.ground_truth is not None]
                        incorrect_predictions = [p for p in labeled_predictions if not p.is_correct]
                        error_rate = len(incorrect_predictions) / len(labeled_predictions) if labeled_predictions else 0.0
                        
                        # Get threshold information
                        threshold_used = None
                        is_optimized = False
                        if recent_predictions:
                            threshold_used = recent_predictions[-1].threshold_used
                            # Check if threshold is optimized (not default 0.5)
                            is_optimized = threshold_used is not None and threshold_used != 0.5
                    else:
                        avg_confidence = 0.0
                        avg_processing_time = 0.0
                        error_rate = 0.0
                        threshold_used = None
                        is_optimized = False
                
                performance[language] = LanguagePerformance(
                    language=language,
                    accuracy=metrics.accuracy,
                    precision=metrics.precision,
                    recall=metrics.recall,
                    f1_score=metrics.f1_score,
                    sample_count=metrics.sample_count,
                    avg_confidence=avg_confidence,
                    avg_processing_time=avg_processing_time,
                    confidence_distribution=metrics.confidence_distribution,
                    error_rate=error_rate,
                    threshold_used=threshold_used,
                    is_threshold_optimized=is_optimized
                )
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to generate language performance: {str(e)}")
            return {}
    
    def _calculate_overall_metrics(self, per_language_performance: Dict[str, LanguagePerformance]) -> Dict[str, Any]:
        """Calculate overall system metrics from per-language performance."""
        try:
            if not per_language_performance:
                return {
                    'total_predictions': 0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
            
            # Weight metrics by sample count
            total_samples = sum(perf.sample_count for perf in per_language_performance.values())
            
            if total_samples == 0:
                return {
                    'total_predictions': 0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
            
            # Calculate weighted averages
            weighted_accuracy = sum(perf.accuracy * perf.sample_count 
                                  for perf in per_language_performance.values()) / total_samples
            
            weighted_precision = sum(perf.precision * perf.sample_count 
                                   for perf in per_language_performance.values()) / total_samples
            
            weighted_recall = sum(perf.recall * perf.sample_count 
                                for perf in per_language_performance.values()) / total_samples
            
            weighted_f1 = sum(perf.f1_score * perf.sample_count 
                            for perf in per_language_performance.values()) / total_samples
            
            return {
                'total_predictions': total_samples,
                'accuracy': weighted_accuracy,
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate overall metrics: {str(e)}")
            return {
                'total_predictions': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def _calculate_overall_confidence_distribution(self, language_metrics: Dict[str, AccuracyMetrics]) -> Dict[str, int]:
        """Calculate overall confidence distribution across all languages."""
        try:
            overall_distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
            
            for metrics in language_metrics.values():
                for bin_name, count in metrics.confidence_distribution.items():
                    overall_distribution[bin_name] += count
            
            return overall_distribution
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence distribution: {str(e)}")
            return {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    
    def _calculate_processing_time_metrics(self, language_metrics: Dict[str, AccuracyMetrics]) -> Dict[str, Any]:
        """Calculate processing time metrics across all languages."""
        try:
            all_processing_times = []
            
            for metrics in language_metrics.values():
                all_processing_times.extend(metrics.processing_times)
            
            if not all_processing_times:
                return {
                    'avg_time': 0.0,
                    'percentiles': {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
                }
            
            avg_time = statistics.mean(all_processing_times)
            
            # Calculate percentiles
            sorted_times = sorted(all_processing_times)
            n = len(sorted_times)
            
            percentiles = {
                'P50': sorted_times[int(n * 0.5)] if n > 0 else 0.0,
                'P95': sorted_times[int(n * 0.95)] if n > 0 else 0.0,
                'P99': sorted_times[int(n * 0.99)] if n > 0 else 0.0
            }
            
            return {
                'avg_time': avg_time,
                'percentiles': percentiles
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate processing time metrics: {str(e)}")
            return {
                'avg_time': 0.0,
                'percentiles': {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
            }
    
    def _calculate_error_rates(self, language_metrics: Dict[str, AccuracyMetrics]) -> Dict[str, float]:
        """Calculate error rates per language."""
        try:
            error_rates = {}
            
            for language, metrics in language_metrics.items():
                error_rate = 1.0 - metrics.accuracy if metrics.accuracy >= 0 else 0.0
                error_rates[language] = error_rate
            
            return error_rates
            
        except Exception as e:
            logger.error(f"Failed to calculate error rates: {str(e)}")
            return {}
    
    def _get_resource_metrics_for_period(self, start_time: datetime, end_time: datetime) -> List[SystemResourceMetrics]:
        """Get resource metrics for the specified time period."""
        try:
            with self._resource_lock:
                period_metrics = [
                    metric for metric in self.resource_history
                    if start_time <= metric.timestamp <= end_time
                ]
            
            return period_metrics
            
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {str(e)}")
            return []
    
    def _calculate_resource_summary(self, resource_metrics: List[SystemResourceMetrics]) -> Dict[str, float]:
        """Calculate resource utilization summary."""
        try:
            if not resource_metrics:
                return {
                    'avg_cpu': 0.0,
                    'avg_memory': 0.0,
                    'peak_memory': 0.0
                }
            
            cpu_values = [m.cpu_usage_percent for m in resource_metrics]
            memory_values = [m.memory_usage_percent for m in resource_metrics]
            
            return {
                'avg_cpu': statistics.mean(cpu_values),
                'avg_memory': statistics.mean(memory_values),
                'peak_memory': max(memory_values)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate resource summary: {str(e)}")
            return {
                'avg_cpu': 0.0,
                'avg_memory': 0.0,
                'peak_memory': 0.0
            }
    
    def _get_model_information(self) -> Dict[str, Any]:
        """Get model version and threshold optimization information."""
        try:
            # This would typically interface with the detection engine
            # For now, return placeholder information
            return {
                'versions': {
                    'English': 'xls-r-300m-1.0.0',
                    'Tamil': 'xls-r-300m-1.0.0',
                    'Hindi': 'xls-r-300m-1.0.0',
                    'Malayalam': 'xls-r-300m-1.0.0',
                    'Telugu': 'xls-r-300m-1.0.0'
                },
                'threshold_status': {
                    'English': True,
                    'Tamil': False,
                    'Hindi': True,
                    'Malayalam': False,
                    'Telugu': True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get model information: {str(e)}")
            return {'versions': {}, 'threshold_status': {}}
    
    def _generate_insights(self, report: PerformanceReport) -> List[str]:
        """Generate automated insights from the performance report."""
        try:
            insights = []
            
            # Overall performance insights
            if report.overall_accuracy > 0.9:
                insights.append(f"Excellent overall accuracy of {report.overall_accuracy:.1%} maintained across all languages.")
            elif report.overall_accuracy < 0.8:
                insights.append(f"Overall accuracy of {report.overall_accuracy:.1%} is below recommended threshold of 80%.")
            
            # Per-language insights
            best_language = max(report.per_language_performance.items(), 
                              key=lambda x: x[1].accuracy, default=(None, None))
            worst_language = min(report.per_language_performance.items(), 
                               key=lambda x: x[1].accuracy, default=(None, None))
            
            if best_language[0] and worst_language[0]:
                accuracy_gap = best_language[1].accuracy - worst_language[1].accuracy
                if accuracy_gap > 0.1:
                    insights.append(f"Significant accuracy gap between {best_language[0]} "
                                  f"({best_language[1].accuracy:.1%}) and {worst_language[0]} "
                                  f"({worst_language[1].accuracy:.1%}).")
            
            # Processing time insights
            if report.avg_processing_time > 5.0:
                insights.append(f"Average processing time of {report.avg_processing_time:.2f}s "
                              f"may impact user experience.")
            
            # Confidence distribution insights
            low_confidence_count = report.overall_confidence_distribution.get("0.0-0.2", 0) + \
                                 report.overall_confidence_distribution.get("0.2-0.4", 0)
            total_predictions = sum(report.overall_confidence_distribution.values())
            
            if total_predictions > 0 and low_confidence_count / total_predictions > 0.2:
                insights.append(f"High proportion ({low_confidence_count/total_predictions:.1%}) "
                              f"of low-confidence predictions may indicate model uncertainty.")
            
            # Alert insights
            if report.degradation_alerts:
                critical_alerts = [a for a in report.degradation_alerts if a.alert_level.value == "critical"]
                if critical_alerts:
                    insights.append(f"{len(critical_alerts)} critical accuracy degradation alerts "
                                  f"require immediate attention.")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return []
    
    def _generate_recommendations(self, report: PerformanceReport) -> List[str]:
        """Generate automated recommendations from the performance report."""
        try:
            recommendations = []
            
            # Accuracy recommendations
            for language, perf in report.per_language_performance.items():
                if perf.accuracy < 0.8:
                    recommendations.append(f"Consider retraining or optimizing the {language} classifier "
                                         f"(current accuracy: {perf.accuracy:.1%}).")
                
                if not perf.is_threshold_optimized:
                    recommendations.append(f"Optimize classification threshold for {language} "
                                         f"to improve precision/recall balance.")
            
            # Processing time recommendations
            if report.avg_processing_time > 3.0:
                recommendations.append("Consider optimizing model inference or using lighter models "
                                     "to reduce processing time.")
            
            # Resource utilization recommendations
            if report.peak_memory_usage > 80.0:
                recommendations.append("High memory usage detected. Consider implementing model "
                                     "caching strategies or memory optimization.")
            
            if report.avg_cpu_usage > 70.0:
                recommendations.append("High CPU usage detected. Consider load balancing or "
                                     "horizontal scaling for better performance.")
            
            # Confidence recommendations
            low_confidence_languages = [
                lang for lang, perf in report.per_language_performance.items()
                if perf.avg_confidence < 0.7
            ]
            
            if low_confidence_languages:
                recommendations.append(f"Review model confidence for {', '.join(low_confidence_languages)} "
                                     f"- consider additional training data or model improvements.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return []
    
    def _monitor_resources(self) -> None:
        """Background thread for monitoring system resources."""
        try:
            while self._monitoring_active:
                try:
                    # Collect resource metrics
                    timestamp = datetime.now()
                    
                    if PSUTIL_AVAILABLE:
                        import psutil
                        
                        # CPU and memory
                        cpu_percent = psutil.cpu_percent(interval=1.0)
                        memory = psutil.virtual_memory()
                        disk = psutil.disk_usage('/')
                        
                        # Thread count
                        active_threads = threading.active_count()
                        
                        metrics = SystemResourceMetrics(
                            timestamp=timestamp,
                            cpu_usage_percent=cpu_percent,
                            memory_usage_bytes=memory.used,
                            memory_usage_percent=memory.percent,
                            disk_usage_bytes=disk.used,
                            disk_free_bytes=disk.free,
                            active_threads=active_threads
                        )
                    else:
                        # Default metrics when psutil not available
                        metrics = SystemResourceMetrics(
                            timestamp=timestamp,
                            cpu_usage_percent=0.0,
                            memory_usage_bytes=0,
                            memory_usage_percent=0.0,
                            disk_usage_bytes=0,
                            disk_free_bytes=1024*1024*1024,  # 1GB default
                            active_threads=threading.active_count()
                        )
                    
                    # Store metrics
                    with self._resource_lock:
                        self.resource_history.append(metrics)
                        
                        # Keep only last 24 hours of data
                        cutoff_time = datetime.now() - timedelta(hours=24)
                        self.resource_history = [
                            m for m in self.resource_history if m.timestamp >= cutoff_time
                        ]
                    
                    # Sleep for monitoring interval
                    time.sleep(60)  # Monitor every minute
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {str(e)}")
                    time.sleep(60)  # Continue monitoring despite errors
                    
        except Exception as e:
            logger.error(f"Resource monitoring thread failed: {str(e)}")
    
    def _report_to_dict(self, report: PerformanceReport) -> Dict[str, Any]:
        """Convert PerformanceReport to dictionary for JSON serialization."""
        try:
            return {
                'report_id': report.report_id,
                'generation_timestamp': report.generation_timestamp.isoformat(),
                'time_period_start': report.time_period_start.isoformat(),
                'time_period_end': report.time_period_end.isoformat(),
                'time_period_duration': str(report.time_period_duration),
                
                'total_predictions': report.total_predictions,
                'overall_accuracy': report.overall_accuracy,
                'overall_precision': report.overall_precision,
                'overall_recall': report.overall_recall,
                'overall_f1_score': report.overall_f1_score,
                
                'per_language_performance': {
                    lang: {
                        'language': perf.language,
                        'accuracy': perf.accuracy,
                        'precision': perf.precision,
                        'recall': perf.recall,
                        'f1_score': perf.f1_score,
                        'sample_count': perf.sample_count,
                        'avg_confidence': perf.avg_confidence,
                        'avg_processing_time': perf.avg_processing_time,
                        'confidence_distribution': perf.confidence_distribution,
                        'error_rate': perf.error_rate,
                        'threshold_used': perf.threshold_used,
                        'is_threshold_optimized': perf.is_threshold_optimized
                    }
                    for lang, perf in report.per_language_performance.items()
                },
                
                'overall_confidence_distribution': report.overall_confidence_distribution,
                'avg_processing_time': report.avg_processing_time,
                'processing_time_percentiles': report.processing_time_percentiles,
                
                'error_rates': report.error_rates,
                'degradation_alerts': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'language': alert.language,
                        'alert_level': alert.alert_level.value,
                        'current_accuracy': alert.current_accuracy,
                        'baseline_accuracy': alert.baseline_accuracy,
                        'degradation_amount': alert.degradation_amount,
                        'sample_count': alert.sample_count,
                        'time_window': str(alert.time_window),
                        'message': alert.message,
                        'additional_info': alert.additional_info
                    }
                    for alert in report.degradation_alerts
                ],
                
                'resource_metrics': [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'cpu_usage_percent': metric.cpu_usage_percent,
                        'memory_usage_bytes': metric.memory_usage_bytes,
                        'memory_usage_percent': metric.memory_usage_percent,
                        'disk_usage_bytes': metric.disk_usage_bytes,
                        'disk_free_bytes': metric.disk_free_bytes,
                        'active_threads': metric.active_threads,
                        'model_cache_size_mb': metric.model_cache_size_mb,
                        'gpu_usage_percent': metric.gpu_usage_percent,
                        'gpu_memory_usage_mb': metric.gpu_memory_usage_mb
                    }
                    for metric in report.resource_metrics
                ],
                
                'avg_cpu_usage': report.avg_cpu_usage,
                'avg_memory_usage': report.avg_memory_usage,
                'peak_memory_usage': report.peak_memory_usage,
                
                'model_versions': report.model_versions,
                'threshold_optimization_status': report.threshold_optimization_status,
                
                'insights': report.insights,
                'recommendations': report.recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to convert report to dict: {str(e)}")
            raise
    
    def _dict_to_report(self, report_dict: Dict[str, Any]) -> PerformanceReport:
        """Convert dictionary to PerformanceReport object."""
        try:
            # This is a simplified version - in practice you'd want full deserialization
            # For now, just create a basic report structure
            return PerformanceReport(
                report_id=report_dict['report_id'],
                generation_timestamp=datetime.fromisoformat(report_dict['generation_timestamp']),
                time_period_start=datetime.fromisoformat(report_dict['time_period_start']),
                time_period_end=datetime.fromisoformat(report_dict['time_period_end']),
                time_period_duration=timedelta(seconds=0),  # Simplified
                
                total_predictions=report_dict['total_predictions'],
                overall_accuracy=report_dict['overall_accuracy'],
                overall_precision=report_dict['overall_precision'],
                overall_recall=report_dict['overall_recall'],
                overall_f1_score=report_dict['overall_f1_score'],
                
                per_language_performance={},  # Simplified
                overall_confidence_distribution=report_dict['overall_confidence_distribution'],
                avg_processing_time=report_dict['avg_processing_time'],
                processing_time_percentiles=report_dict['processing_time_percentiles'],
                
                error_rates=report_dict['error_rates'],
                degradation_alerts=[],  # Simplified
                
                resource_metrics=[],  # Simplified
                avg_cpu_usage=report_dict['avg_cpu_usage'],
                avg_memory_usage=report_dict['avg_memory_usage'],
                peak_memory_usage=report_dict['peak_memory_usage'],
                
                model_versions=report_dict['model_versions'],
                threshold_optimization_status=report_dict['threshold_optimization_status'],
                
                insights=report_dict.get('insights', []),
                recommendations=report_dict.get('recommendations', [])
            )
            
        except Exception as e:
            logger.error(f"Failed to convert dict to report: {str(e)}")
            raise