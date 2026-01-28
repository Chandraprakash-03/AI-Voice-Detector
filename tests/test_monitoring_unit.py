"""
Unit tests for monitoring system components.

Tests the AccuracyMonitor and PerformanceReportGenerator classes
with specific examples and edge cases.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.monitoring.accuracy_monitor import (
    AccuracyMonitor, PredictionRecord, AccuracyMetrics, 
    DegradationAlert, AlertLevel
)
from src.monitoring.performance_report import (
    PerformanceReportGenerator, PerformanceReport, 
    LanguagePerformance, SystemResourceMetrics
)
from src.detection.engine import DetectionResult


class TestAccuracyMonitor:
    """Test cases for AccuracyMonitor class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings object."""
        settings = Mock()
        settings.ml.models_base_path = "/tmp/test_models"
        return settings
    
    @pytest.fixture
    def accuracy_monitor(self, mock_settings, temp_storage):
        """Create AccuracyMonitor instance for testing."""
        return AccuracyMonitor(
            settings=mock_settings,
            window_size=10,
            degradation_threshold=0.1,
            storage_path=temp_storage
        )
    
    def test_initialization(self, accuracy_monitor):
        """Test AccuracyMonitor initialization."""
        assert accuracy_monitor.window_size == 10
        assert accuracy_monitor.degradation_threshold == 0.1
        assert accuracy_monitor.total_predictions == 0
        assert len(accuracy_monitor.prediction_windows) == 0
        assert len(accuracy_monitor.baseline_accuracies) == 0
    
    def test_track_prediction_without_ground_truth(self, accuracy_monitor):
        """Test tracking prediction without ground truth."""
        result = DetectionResult(
            classification="HUMAN",
            confidence_score=0.85,
            model_version="test-v1.0",
            processing_time=1.5
        )
        
        accuracy_monitor.track_prediction(result, "English")
        
        assert accuracy_monitor.total_predictions == 1
        assert len(accuracy_monitor.prediction_windows["English"]) == 1
        
        record = accuracy_monitor.prediction_windows["English"][0]
        assert record.language == "English"
        assert record.classification == "HUMAN"
        assert record.confidence_score == 0.85
        assert record.ground_truth is None
        assert record.is_correct is None
    
    def test_track_prediction_with_ground_truth(self, accuracy_monitor):
        """Test tracking prediction with ground truth."""
        result = DetectionResult(
            classification="HUMAN",
            confidence_score=0.85,
            model_version="test-v1.0",
            processing_time=1.5
        )
        
        accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
        
        record = accuracy_monitor.prediction_windows["English"][0]
        assert record.ground_truth == "HUMAN"
        assert record.is_correct is True
        
        # Test incorrect prediction
        result2 = DetectionResult(
            classification="AI_GENERATED",
            confidence_score=0.75,
            model_version="test-v1.0",
            processing_time=1.2
        )
        
        accuracy_monitor.track_prediction(result2, "English", ground_truth="HUMAN")
        
        record2 = accuracy_monitor.prediction_windows["English"][1]
        assert record2.ground_truth == "HUMAN"
        assert record2.is_correct is False
    
    def test_calculate_running_accuracy(self, accuracy_monitor):
        """Test running accuracy calculation."""
        # Test with no data
        accuracy = accuracy_monitor.calculate_running_accuracy("English")
        assert accuracy == -1.0
        
        # Add predictions with ground truth
        predictions = [
            ("HUMAN", "HUMAN", True),
            ("AI_GENERATED", "AI_GENERATED", True),
            ("HUMAN", "AI_GENERATED", False),
            ("AI_GENERATED", "HUMAN", False),
            ("HUMAN", "HUMAN", True)
        ]
        
        for pred, truth, _ in predictions:
            result = DetectionResult(
                classification=pred,
                confidence_score=0.8,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth=truth)
        
        # Calculate accuracy: 3 correct out of 5 = 0.6
        accuracy = accuracy_monitor.calculate_running_accuracy("English")
        assert accuracy == 0.6
    
    def test_sliding_window_behavior(self, accuracy_monitor):
        """Test sliding window behavior with window size limit."""
        # Add more predictions than window size
        for i in range(15):  # Window size is 10
            result = DetectionResult(
                classification="HUMAN",
                confidence_score=0.8,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
        
        # Should only keep last 10 predictions
        assert len(accuracy_monitor.prediction_windows["English"]) == 10
        assert accuracy_monitor.total_predictions == 15
    
    def test_baseline_accuracy_management(self, accuracy_monitor):
        """Test baseline accuracy setting and retrieval."""
        # Test setting baseline
        accuracy_monitor.set_baseline_accuracy("English", 0.9)
        assert accuracy_monitor.get_baseline_accuracy("English") == 0.9
        
        # Test invalid accuracy
        with pytest.raises(ValueError):
            accuracy_monitor.set_baseline_accuracy("English", 1.5)
        
        with pytest.raises(ValueError):
            accuracy_monitor.set_baseline_accuracy("English", -0.1)
    
    def test_degradation_detection(self, accuracy_monitor):
        """Test accuracy degradation detection."""
        # Set baseline
        accuracy_monitor.set_baseline_accuracy("English", 0.9)
        
        # Add predictions that result in lower accuracy
        for i in range(10):
            # 7 correct, 3 incorrect = 0.7 accuracy (0.2 degradation > 0.1 threshold)
            is_correct = i < 7
            pred = "HUMAN" if is_correct else "AI_GENERATED"
            truth = "HUMAN"
            
            result = DetectionResult(
                classification=pred,
                confidence_score=0.8,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth=truth)
        
        # Should detect degradation
        is_degraded = accuracy_monitor.detect_accuracy_degradation("English")
        assert is_degraded is True
        
        # Should have generated an alert
        assert len(accuracy_monitor.alert_history["English"]) > 0
        alert = accuracy_monitor.alert_history["English"][0]
        assert alert.language == "English"
        assert alert.current_accuracy == 0.7
        assert alert.baseline_accuracy == 0.9
        assert abs(alert.degradation_amount - 0.2) < 0.001  # Use approximate comparison for floating point
    
    def test_accuracy_metrics_generation(self, accuracy_monitor):
        """Test comprehensive accuracy metrics generation."""
        # Add diverse predictions
        predictions = [
            ("HUMAN", "HUMAN", 0.9),      # Correct, high confidence
            ("AI_GENERATED", "AI_GENERATED", 0.8),  # Correct, high confidence
            ("HUMAN", "AI_GENERATED", 0.6),  # Incorrect, medium confidence
            ("AI_GENERATED", "HUMAN", 0.7),  # Incorrect, medium confidence
            ("HUMAN", "HUMAN", 0.95),     # Correct, very high confidence
        ]
        
        for pred, truth, conf in predictions:
            result = DetectionResult(
                classification=pred,
                confidence_score=conf,
                model_version="test-v1.0",
                processing_time=1.0 + conf  # Vary processing time
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth=truth)
        
        # Get metrics
        metrics = accuracy_monitor.get_accuracy_metrics("English")
        
        assert metrics is not None
        assert metrics.accuracy == 0.6  # 3 correct out of 5
        assert metrics.sample_count == 5
        assert len(metrics.processing_times) == 5
        assert len(metrics.confidence_distribution) == 5  # All bins
    
    def test_monitoring_summary(self, accuracy_monitor):
        """Test monitoring summary generation."""
        # Add some data
        result = DetectionResult(
            classification="HUMAN",
            confidence_score=0.8,
            model_version="test-v1.0",
            processing_time=1.0
        )
        accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
        accuracy_monitor.set_baseline_accuracy("English", 0.9)
        
        summary = accuracy_monitor.get_monitoring_summary()
        
        assert summary["total_predictions"] == 1
        assert "English" in summary["monitored_languages"]
        assert summary["window_size"] == 10
        assert summary["degradation_threshold"] == 0.1
        assert summary["baseline_accuracies"]["English"] == 0.9
        assert "English" in summary["current_accuracies"]
    
    def test_data_clearing(self, accuracy_monitor):
        """Test data clearing functionality."""
        # Add data for multiple languages
        for lang in ["English", "Tamil"]:
            result = DetectionResult(
                classification="HUMAN",
                confidence_score=0.8,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, lang, ground_truth="HUMAN")
            accuracy_monitor.set_baseline_accuracy(lang, 0.9)
        
        # Clear specific language
        accuracy_monitor.clear_data("English", keep_baselines=True)
        
        assert len(accuracy_monitor.prediction_windows["English"]) == 0
        assert len(accuracy_monitor.prediction_windows["Tamil"]) == 1
        assert accuracy_monitor.get_baseline_accuracy("English") == 0.9  # Kept
        
        # Clear all data without keeping baselines
        accuracy_monitor.clear_data(keep_baselines=False)
        
        assert len(accuracy_monitor.prediction_windows) == 0
        assert len(accuracy_monitor.baseline_accuracies) == 0
        assert accuracy_monitor.total_predictions == 0


class TestPerformanceReportGenerator:
    """Test cases for PerformanceReportGenerator class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings object."""
        settings = Mock()
        settings.ml.models_base_path = "/tmp/test_models"
        return settings
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def accuracy_monitor(self, mock_settings, temp_storage):
        """Create AccuracyMonitor instance for testing."""
        return AccuracyMonitor(
            settings=mock_settings,
            window_size=100,
            storage_path=temp_storage
        )
    
    @pytest.fixture
    def report_generator(self, accuracy_monitor, mock_settings):
        """Create PerformanceReportGenerator instance for testing."""
        with patch('src.monitoring.performance_report.PSUTIL_AVAILABLE', False):
            generator = PerformanceReportGenerator(accuracy_monitor, mock_settings)
            # Stop the monitoring thread for testing
            generator.stop_monitoring()
            return generator
    
    def test_initialization(self, report_generator):
        """Test PerformanceReportGenerator initialization."""
        assert report_generator.accuracy_monitor is not None
        assert report_generator.settings is not None
        assert isinstance(report_generator.resource_history, list)
    
    def test_performance_report_generation_empty_data(self, report_generator):
        """Test performance report generation with no data."""
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        assert report.report_id.startswith("perf_report_")
        assert report.total_predictions == 0
        assert report.overall_accuracy == 0.0
        assert len(report.per_language_performance) == 0
        assert report.avg_processing_time == 0.0
    
    def test_performance_report_generation_with_data(self, report_generator):
        """Test performance report generation with sample data."""
        # Add sample data to accuracy monitor
        accuracy_monitor = report_generator.accuracy_monitor
        
        # Add predictions for multiple languages
        languages = ["English", "Tamil"]
        for lang in languages:
            for i in range(10):
                is_correct = i < 8  # 80% accuracy
                pred = "HUMAN" if is_correct else "AI_GENERATED"
                truth = "HUMAN"
                
                result = DetectionResult(
                    classification=pred,
                    confidence_score=0.8 + (i * 0.01),  # Vary confidence
                    model_version=f"test-{lang}-v1.0",
                    processing_time=1.0 + (i * 0.1)  # Vary processing time
                )
                accuracy_monitor.track_prediction(result, lang, ground_truth=truth)
        
        # Generate report
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        assert report.total_predictions == 20
        assert report.overall_accuracy == 0.8  # 80% overall
        assert len(report.per_language_performance) == 2
        
        # Check per-language performance
        for lang in languages:
            lang_perf = report.per_language_performance[lang]
            assert lang_perf.language == lang
            assert lang_perf.accuracy == 0.8
            assert lang_perf.sample_count == 10
            assert lang_perf.avg_confidence > 0.8
            assert lang_perf.avg_processing_time > 1.0
    
    def test_confidence_distribution_calculation(self, report_generator):
        """Test confidence distribution calculation."""
        accuracy_monitor = report_generator.accuracy_monitor
        
        # Add predictions with specific confidence ranges
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]  # One in each bin
        
        for conf in confidences:
            result = DetectionResult(
                classification="HUMAN",
                confidence_score=conf,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
        
        # Generate report
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        # Check confidence distribution
        dist = report.overall_confidence_distribution
        assert dist["0.0-0.2"] == 1  # 0.1
        assert dist["0.2-0.4"] == 1  # 0.3
        assert dist["0.4-0.6"] == 1  # 0.5
        assert dist["0.6-0.8"] == 1  # 0.7
        assert dist["0.8-1.0"] == 1  # 0.9
    
    def test_processing_time_percentiles(self, report_generator):
        """Test processing time percentile calculation."""
        accuracy_monitor = report_generator.accuracy_monitor
        
        # Add predictions with known processing times
        processing_times = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0]
        
        for pt in processing_times:
            result = DetectionResult(
                classification="HUMAN",
                confidence_score=0.8,
                model_version="test-v1.0",
                processing_time=pt
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
        
        # Generate report
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        # Check percentiles
        percentiles = report.processing_time_percentiles
        assert percentiles["P50"] == 3.0  # Median
        assert percentiles["P95"] >= 5.0   # 95th percentile
        assert percentiles["P99"] >= 10.0  # 99th percentile
    
    def test_insights_generation(self, report_generator):
        """Test automated insights generation."""
        accuracy_monitor = report_generator.accuracy_monitor
        
        # Add data that should trigger insights
        # High accuracy for English, low accuracy for Tamil
        english_correct = 9
        tamil_correct = 6
        
        for i in range(10):
            # English: 90% accuracy
            pred = "HUMAN" if i < english_correct else "AI_GENERATED"
            result = DetectionResult(
                classification=pred,
                confidence_score=0.9,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
            
            # Tamil: 60% accuracy
            pred = "HUMAN" if i < tamil_correct else "AI_GENERATED"
            result = DetectionResult(
                classification=pred,
                confidence_score=0.6,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "Tamil", ground_truth="HUMAN")
        
        # Generate report with insights
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period, include_insights=True)
        
        # Should have insights about accuracy gap
        assert len(report.insights) > 0
        assert any("accuracy gap" in insight.lower() for insight in report.insights)
        
        # Should have recommendations
        assert len(report.recommendations) > 0
    
    def test_report_serialization(self, report_generator, temp_storage):
        """Test report saving and loading."""
        # Generate a simple report
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        # Save report
        output_path = Path(temp_storage) / "test_report.json"
        saved_path = report_generator.save_report(report, str(output_path))
        
        assert Path(saved_path).exists()
        
        # Verify JSON structure
        with open(saved_path, 'r') as f:
            report_data = json.load(f)
        
        assert report_data["report_id"] == report.report_id
        assert report_data["total_predictions"] == report.total_predictions
        assert "generation_timestamp" in report_data
    
    def test_resource_metrics_collection(self, report_generator):
        """Test resource metrics collection."""
        # Manually add some resource metrics
        metrics = SystemResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=50.0,
            memory_usage_bytes=1024*1024*1024,  # 1GB
            memory_usage_percent=60.0,
            disk_usage_bytes=10*1024*1024*1024,  # 10GB
            disk_free_bytes=5*1024*1024*1024,   # 5GB
            active_threads=10
        )
        
        report_generator.resource_history.append(metrics)
        
        # Generate report
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        # Should include resource metrics
        assert len(report.resource_metrics) == 1
        assert report.avg_cpu_usage == 50.0
        assert report.avg_memory_usage == 60.0
        assert report.peak_memory_usage == 60.0
    
    def test_degradation_alerts_inclusion(self, report_generator):
        """Test inclusion of degradation alerts in reports."""
        accuracy_monitor = report_generator.accuracy_monitor
        
        # Set baseline and create degradation
        accuracy_monitor.set_baseline_accuracy("English", 0.9)
        
        # Add predictions that cause degradation
        for i in range(10):
            is_correct = i < 6  # 60% accuracy (0.3 degradation)
            pred = "HUMAN" if is_correct else "AI_GENERATED"
            
            result = DetectionResult(
                classification=pred,
                confidence_score=0.8,
                model_version="test-v1.0",
                processing_time=1.0
            )
            accuracy_monitor.track_prediction(result, "English", ground_truth="HUMAN")
        
        # Generate report
        time_period = timedelta(hours=1)
        report = report_generator.generate_performance_report(time_period)
        
        # Should include degradation alerts
        assert len(report.degradation_alerts) > 0
        alert = report.degradation_alerts[0]
        assert alert.language == "English"
        assert alert.current_accuracy == 0.6
        assert alert.baseline_accuracy == 0.9


if __name__ == "__main__":
    pytest.main([__file__])