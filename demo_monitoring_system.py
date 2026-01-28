#!/usr/bin/env python3
"""
Demo script for the Performance Monitoring System.

This script demonstrates the AccuracyMonitor and PerformanceReportGenerator
functionality with sample data and shows how to integrate monitoring
with the AI voice detection system.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.monitoring.accuracy_monitor import AccuracyMonitor
from src.monitoring.performance_report import PerformanceReportGenerator
from src.detection.engine import DetectionResult


def create_sample_detection_result(classification: str, confidence_score: float, 
                                 processing_time: float, model_version: str = "demo-v1.0") -> DetectionResult:
    """Create a sample DetectionResult for demonstration."""
    return DetectionResult(
        classification=classification,
        confidence_score=confidence_score,
        model_version=model_version,
        processing_time=processing_time
    )


def simulate_detection_session(accuracy_monitor: AccuracyMonitor, 
                             language: str, num_predictions: int = 50):
    """Simulate a detection session with varying accuracy and confidence."""
    print(f"\n🔄 Simulating {num_predictions} predictions for {language}...")
    
    # Simulate predictions with realistic patterns
    for i in range(num_predictions):
        # Vary accuracy over time (start high, degrade slightly)
        base_accuracy = 0.9 - (i / num_predictions) * 0.1  # 90% -> 80%
        is_correct = (i % 10) < (base_accuracy * 10)
        
        # Generate prediction and ground truth
        if is_correct:
            classification = "HUMAN"
            ground_truth = "HUMAN"
            confidence = 0.8 + (i % 3) * 0.05  # 0.8-0.9
        else:
            classification = "AI_GENERATED"
            ground_truth = "HUMAN"  # Incorrect prediction
            confidence = 0.6 + (i % 3) * 0.05  # 0.6-0.7 (lower for incorrect)
        
        # Vary processing time
        processing_time = 1.0 + (i % 5) * 0.2  # 1.0-1.8 seconds
        
        # Create detection result
        result = create_sample_detection_result(
            classification=classification,
            confidence_score=confidence,
            processing_time=processing_time,
            model_version=f"{language.lower()}-classifier-v1.0"
        )
        
        # Track prediction
        accuracy_monitor.track_prediction(result, language, ground_truth=ground_truth)
        
        # Small delay to simulate real processing
        time.sleep(0.01)
    
    print(f"✅ Completed {num_predictions} predictions for {language}")


def demonstrate_monitoring_system():
    """Demonstrate the complete monitoring system functionality."""
    print("🚀 AI Voice Detection Monitoring System Demo")
    print("=" * 50)
    
    # Create temporary storage for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary storage: {temp_dir}")
        
        # Initialize monitoring system
        print("\n🔧 Initializing monitoring system...")
        accuracy_monitor = AccuracyMonitor(
            window_size=100,
            degradation_threshold=0.05,  # 5% degradation threshold
            storage_path=temp_dir
        )
        
        report_generator = PerformanceReportGenerator(accuracy_monitor)
        
        # Set baseline accuracies
        print("\n📊 Setting baseline accuracies...")
        languages = ["English", "Tamil", "Hindi"]
        baselines = {"English": 0.92, "Tamil": 0.88, "Hindi": 0.90}
        
        for lang, baseline in baselines.items():
            accuracy_monitor.set_baseline_accuracy(lang, baseline)
            print(f"  {lang}: {baseline:.1%}")
        
        # Simulate detection sessions for multiple languages
        print("\n🎯 Simulating detection sessions...")
        for language in languages:
            simulate_detection_session(accuracy_monitor, language, num_predictions=30)
        
        # Display real-time monitoring results
        print("\n📈 Real-time Monitoring Results:")
        print("-" * 40)
        
        for language in languages:
            current_accuracy = accuracy_monitor.calculate_running_accuracy(language)
            baseline_accuracy = accuracy_monitor.get_baseline_accuracy(language)
            is_degraded = accuracy_monitor.detect_accuracy_degradation(language)
            
            status = "🔴 DEGRADED" if is_degraded else "🟢 HEALTHY"
            
            print(f"{language:>10}: {current_accuracy:.1%} (baseline: {baseline_accuracy:.1%}) {status}")
        
        # Get comprehensive accuracy metrics
        print("\n📊 Detailed Accuracy Metrics:")
        print("-" * 40)
        
        for language in languages:
            metrics = accuracy_monitor.get_accuracy_metrics(language)
            if metrics:
                print(f"\n{language}:")
                print(f"  Accuracy: {metrics.accuracy:.1%}")
                print(f"  Precision: {metrics.precision:.1%}")
                print(f"  Recall: {metrics.recall:.1%}")
                print(f"  F1 Score: {metrics.f1_score:.1%}")
                print(f"  Sample Count: {metrics.sample_count}")
                print(f"  Avg Processing Time: {sum(metrics.processing_times)/len(metrics.processing_times):.2f}s")
        
        # Check for degradation alerts
        print("\n🚨 Degradation Alerts:")
        print("-" * 40)
        
        recent_alerts = accuracy_monitor.get_recent_alerts(hours=1)
        if recent_alerts:
            for alert in recent_alerts:
                severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}
                icon = severity_icon.get(alert.alert_level.value, "❓")
                print(f"{icon} {alert.message}")
        else:
            print("✅ No recent alerts")
        
        # Generate comprehensive performance report
        print("\n📋 Generating Performance Report...")
        print("-" * 40)
        
        time_period = timedelta(hours=1)  # Last hour
        report = report_generator.generate_performance_report(
            time_period=time_period,
            include_insights=True
        )
        
        print(f"Report ID: {report.report_id}")
        print(f"Time Period: {report.time_period_duration}")
        print(f"Total Predictions: {report.total_predictions}")
        print(f"Overall Accuracy: {report.overall_accuracy:.1%}")
        print(f"Overall Precision: {report.overall_precision:.1%}")
        print(f"Overall Recall: {report.overall_recall:.1%}")
        print(f"Overall F1 Score: {report.overall_f1_score:.1%}")
        print(f"Avg Processing Time: {report.avg_processing_time:.2f}s")
        
        # Display per-language performance
        print(f"\n📊 Per-Language Performance:")
        for lang, perf in report.per_language_performance.items():
            print(f"  {lang}:")
            print(f"    Accuracy: {perf.accuracy:.1%}")
            print(f"    Avg Confidence: {perf.avg_confidence:.2f}")
            print(f"    Error Rate: {perf.error_rate:.1%}")
            print(f"    Samples: {perf.sample_count}")
        
        # Display confidence distribution
        print(f"\n🎯 Confidence Distribution:")
        total_predictions = sum(report.overall_confidence_distribution.values())
        for bin_range, count in report.overall_confidence_distribution.items():
            percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
            bar = "█" * int(percentage / 5)  # Scale bar
            print(f"  {bin_range}: {count:>3} ({percentage:>5.1f}%) {bar}")
        
        # Display processing time percentiles
        print(f"\n⏱️  Processing Time Percentiles:")
        for percentile, time_val in report.processing_time_percentiles.items():
            print(f"  {percentile}: {time_val:.2f}s")
        
        # Display automated insights
        if report.insights:
            print(f"\n💡 Automated Insights:")
            for i, insight in enumerate(report.insights, 1):
                print(f"  {i}. {insight}")
        
        # Display recommendations
        if report.recommendations:
            print(f"\n🔧 Recommendations:")
            for i, recommendation in enumerate(report.recommendations, 1):
                print(f"  {i}. {recommendation}")
        
        # Save report to file
        print(f"\n💾 Saving Performance Report...")
        report_path = report_generator.save_report(report)
        print(f"Report saved to: {report_path}")
        
        # Display monitoring summary
        print(f"\n📋 Monitoring System Summary:")
        print("-" * 40)
        
        summary = accuracy_monitor.get_monitoring_summary()
        print(f"Total Predictions Tracked: {summary['total_predictions']}")
        print(f"Monitoring Duration: {summary['monitoring_duration']}")
        print(f"Languages Monitored: {', '.join(summary['monitored_languages'])}")
        print(f"Window Size: {summary['window_size']}")
        print(f"Degradation Threshold: {summary['degradation_threshold']:.1%}")
        print(f"Recent Alerts: {summary['recent_alerts_count']}")
        
        # Stop monitoring
        report_generator.stop_monitoring()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"🔍 Check the temporary directory for saved reports and data.")


if __name__ == "__main__":
    try:
        demonstrate_monitoring_system()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()