#!/usr/bin/env python3
"""
System Health Check Script for AI Voice Detection System

This script provides comprehensive health checking capabilities for monitoring
system status, model validation, and performance metrics.
"""

import sys
import os
import json
import logging
import argparse
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.detection.engine import DetectionEngine
from src.monitoring.accuracy_monitor import AccuracyMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Comprehensive health checker for the AI voice detection system.
    
    Provides both local system checks and API endpoint health verification.
    """
    
    def __init__(self, api_base_url: Optional[str] = None):
        """
        Initialize HealthChecker.
        
        Args:
            api_base_url: Base URL for API health checks (optional)
        """
        self.settings = get_settings()
        self.api_base_url = api_base_url or f"http://{self.settings.api.host}:{self.settings.api.port}"
        
        # Health check results
        self.health_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
            "metrics": {},
            "alerts": [],
            "recommendations": []
        }
        
        logger.info(f"HealthChecker initialized with API base URL: {self.api_base_url}")
    
    def check_all(self, include_api: bool = True, include_local: bool = True) -> bool:
        """
        Run all health checks.
        
        Args:
            include_api: Whether to include API endpoint checks
            include_local: Whether to include local system checks
            
        Returns:
            True if all checks pass, False otherwise
        """
        logger.info("Starting comprehensive health check...")
        
        try:
            if include_local:
                self._check_local_system()
                self._check_models()
                self._check_monitoring()
                self._check_configuration()
            
            if include_api:
                self._check_api_endpoints()
            
            # Determine overall status
            self._determine_overall_status()
            
            # Generate recommendations
            self._generate_recommendations()
            
            logger.info(f"Health check completed: {self.health_results['overall_status']}")
            return self.health_results['overall_status'] in ["healthy", "warning"]
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.health_results['overall_status'] = "critical"
            self.health_results['alerts'].append(f"Health check process failed: {str(e)}")
            return False
    
    def _check_local_system(self):
        """Check local system health."""
        logger.info("Checking local system health...")
        
        check = {
            "status": "unknown",
            "details": {},
            "issues": []
        }
        
        try:
            # Initialize DetectionEngine and get health status
            detection_engine = DetectionEngine(self.settings)
            system_health = detection_engine.get_system_health_status()
            
            check["details"] = system_health
            
            # Analyze health score
            health_score = system_health.get("overall_health", {}).get("score", 0.0)
            
            if health_score >= 0.9:
                check["status"] = "excellent"
            elif health_score >= 0.8:
                check["status"] = "healthy"
            elif health_score >= 0.6:
                check["status"] = "warning"
                check["issues"].append(f"System health score is moderate: {health_score:.2f}")
            else:
                check["status"] = "critical"
                check["issues"].append(f"System health score is low: {health_score:.2f}")
            
            # Check individual components
            foundation_model = system_health.get("foundation_model", {})
            if not foundation_model.get("valid", False):
                check["issues"].append("Foundation model validation failed")
            
            classifiers = system_health.get("classifiers", {})
            invalid_classifiers = [lang for lang, status in classifiers.items() 
                                 if not status.get("classifier_valid", False)]
            if invalid_classifiers:
                check["issues"].append(f"Invalid classifiers: {', '.join(invalid_classifiers)}")
            
            # Store metrics
            self.health_results["metrics"]["system_health_score"] = health_score
            self.health_results["metrics"]["valid_classifiers"] = len(classifiers) - len(invalid_classifiers)
            self.health_results["metrics"]["total_classifiers"] = len(classifiers)
            
        except Exception as e:
            check["status"] = "critical"
            check["issues"].append(f"Local system check failed: {str(e)}")
        
        self.health_results["checks"]["local_system"] = check
    
    def _check_models(self):
        """Check model health and validation status."""
        logger.info("Checking model health...")
        
        check = {
            "status": "unknown",
            "details": {},
            "issues": []
        }
        
        try:
            # Initialize DetectionEngine
            detection_engine = DetectionEngine(self.settings)
            
            # Get model registry information
            registry_info = detection_engine.get_model_registry_info()
            
            if "error" in registry_info:
                check["issues"].append(f"Model registry error: {registry_info['error']}")
                check["status"] = "warning"
            else:
                check["details"]["registry"] = registry_info
                
                total_models = registry_info.get("total_models", 0)
                if total_models == 0:
                    check["issues"].append("No models registered in registry")
                    check["status"] = "critical"
                else:
                    self.health_results["metrics"]["registered_models"] = total_models
            
            # Check threshold configurations
            threshold_info = detection_engine.get_all_threshold_info()
            optimized_thresholds = sum(1 for info in threshold_info.values() 
                                     if info.get("is_optimized", False))
            
            check["details"]["thresholds"] = {
                "total": len(threshold_info),
                "optimized": optimized_thresholds
            }
            
            if optimized_thresholds == 0:
                check["issues"].append("No optimized thresholds configured")
            
            self.health_results["metrics"]["optimized_thresholds"] = optimized_thresholds
            
            # Determine status if not already set
            if check["status"] == "unknown":
                if len(check["issues"]) == 0:
                    check["status"] = "healthy"
                elif len(check["issues"]) <= 2:
                    check["status"] = "warning"
                else:
                    check["status"] = "critical"
            
        except Exception as e:
            check["status"] = "critical"
            check["issues"].append(f"Model check failed: {str(e)}")
        
        self.health_results["checks"]["models"] = check
    
    def _check_monitoring(self):
        """Check monitoring system health."""
        logger.info("Checking monitoring system...")
        
        check = {
            "status": "unknown",
            "details": {},
            "issues": []
        }
        
        try:
            # Initialize AccuracyMonitor
            accuracy_monitor = AccuracyMonitor(self.settings)
            monitoring_summary = accuracy_monitor.get_monitoring_summary()
            
            check["details"]["summary"] = monitoring_summary
            
            # Check monitoring activity
            total_predictions = monitoring_summary.get("total_predictions", 0)
            recent_alerts = monitoring_summary.get("recent_alerts_count", 0)
            
            if total_predictions == 0:
                check["issues"].append("No predictions tracked in monitoring system")
                check["status"] = "warning"
            else:
                self.health_results["metrics"]["total_predictions"] = total_predictions
            
            if recent_alerts > 0:
                check["issues"].append(f"{recent_alerts} recent alerts detected")
                
                # Get recent alerts for details
                recent_alert_details = accuracy_monitor.get_recent_alerts(hours=24)
                check["details"]["recent_alerts"] = [
                    {
                        "timestamp": alert.timestamp.isoformat(),
                        "language": alert.language,
                        "level": alert.alert_level.value,
                        "message": alert.message
                    }
                    for alert in recent_alert_details[:5]  # Show up to 5 recent alerts
                ]
            
            # Check accuracy levels
            current_accuracies = monitoring_summary.get("current_accuracies", {})
            baseline_accuracies = monitoring_summary.get("baseline_accuracies", {})
            
            low_accuracy_languages = []
            for language, accuracy in current_accuracies.items():
                if accuracy < 0.8:  # Below 80% accuracy
                    low_accuracy_languages.append(f"{language}: {accuracy:.2f}")
            
            if low_accuracy_languages:
                check["issues"].append(f"Low accuracy detected: {', '.join(low_accuracy_languages)}")
            
            self.health_results["metrics"]["monitored_languages"] = len(current_accuracies)
            self.health_results["metrics"]["recent_alerts"] = recent_alerts
            
            # Determine status if not already set
            if check["status"] == "unknown":
                if recent_alerts > 5:
                    check["status"] = "critical"
                elif recent_alerts > 0 or low_accuracy_languages:
                    check["status"] = "warning"
                else:
                    check["status"] = "healthy"
            
        except Exception as e:
            check["status"] = "critical"
            check["issues"].append(f"Monitoring check failed: {str(e)}")
        
        self.health_results["checks"]["monitoring"] = check
    
    def _check_configuration(self):
        """Check configuration health."""
        logger.info("Checking configuration...")
        
        check = {
            "status": "unknown",
            "details": {},
            "issues": []
        }
        
        try:
            from src.config_manager import ConfigurationManager
            
            config_manager = ConfigurationManager()
            config_summary = config_manager.get_configuration_summary()
            
            check["details"]["summary"] = config_summary
            
            # Check threshold configurations
            thresholds_info = config_summary.get("thresholds", {})
            if thresholds_info.get("count", 0) == 0:
                check["issues"].append("No threshold configurations found")
            
            # Check model paths
            model_paths_info = config_summary.get("model_paths", {})
            missing_paths = model_paths_info.get("missing_paths", 0)
            if missing_paths > 0:
                check["issues"].append(f"{missing_paths} model paths are missing")
            
            # Check deployment configuration
            deployment_info = config_summary.get("deployment", {})
            validation_errors = deployment_info.get("validation_errors", 0)
            if validation_errors > 0:
                check["issues"].append(f"{validation_errors} deployment configuration errors")
            
            # Check system configuration
            system_config_info = config_summary.get("system_config", {})
            if not system_config_info.get("monitoring_enabled", False):
                check["issues"].append("Monitoring not enabled in system configuration")
            
            if not system_config_info.get("validation_enabled", False):
                check["issues"].append("Validation not enabled in system configuration")
            
            # Determine status
            if validation_errors > 0 or missing_paths > 0:
                check["status"] = "critical"
            elif len(check["issues"]) > 0:
                check["status"] = "warning"
            else:
                check["status"] = "healthy"
            
        except Exception as e:
            check["status"] = "critical"
            check["issues"].append(f"Configuration check failed: {str(e)}")
        
        self.health_results["checks"]["configuration"] = check
    
    def _check_api_endpoints(self):
        """Check API endpoint health."""
        logger.info("Checking API endpoints...")
        
        check = {
            "status": "unknown",
            "details": {},
            "issues": []
        }
        
        try:
            # Test basic health endpoint
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=10)
                if response.status_code == 200:
                    check["details"]["basic_health"] = response.json()
                else:
                    check["issues"].append(f"Basic health endpoint returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                check["issues"].append(f"Basic health endpoint unreachable: {str(e)}")
            
            # Test detailed health endpoint
            try:
                response = requests.get(f"{self.api_base_url}/health/detailed", timeout=15)
                if response.status_code == 200:
                    detailed_health = response.json()
                    check["details"]["detailed_health"] = detailed_health
                    
                    # Analyze detailed health
                    health_score = detailed_health.get("health_score", 0.0)
                    if health_score < 0.6:
                        check["issues"].append(f"API reports low health score: {health_score:.2f}")
                    
                    self.health_results["metrics"]["api_health_score"] = health_score
                else:
                    check["issues"].append(f"Detailed health endpoint returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                check["issues"].append(f"Detailed health endpoint unreachable: {str(e)}")
            
            # Test monitoring endpoint
            try:
                response = requests.get(f"{self.api_base_url}/health/monitoring", timeout=10)
                if response.status_code == 200:
                    monitoring_health = response.json()
                    check["details"]["monitoring_health"] = monitoring_health
                    
                    if not monitoring_health.get("monitoring_active", False):
                        check["issues"].append("API monitoring is not active")
                else:
                    check["issues"].append(f"Monitoring health endpoint returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                check["issues"].append(f"Monitoring health endpoint unreachable: {str(e)}")
            
            # Test models endpoint
            try:
                response = requests.get(f"{self.api_base_url}/health/models", timeout=10)
                if response.status_code == 200:
                    models_health = response.json()
                    check["details"]["models_health"] = models_health
                    
                    if models_health.get("overall_model_health") != "healthy":
                        check["issues"].append(f"API reports model health: {models_health.get('overall_model_health')}")
                else:
                    check["issues"].append(f"Models health endpoint returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                check["issues"].append(f"Models health endpoint unreachable: {str(e)}")
            
            # Determine status
            if len(check["issues"]) == 0:
                check["status"] = "healthy"
            elif len(check["issues"]) <= 2:
                check["status"] = "warning"
            else:
                check["status"] = "critical"
            
        except Exception as e:
            check["status"] = "critical"
            check["issues"].append(f"API endpoints check failed: {str(e)}")
        
        self.health_results["checks"]["api_endpoints"] = check
    
    def _determine_overall_status(self):
        """Determine overall health status."""
        checks = self.health_results["checks"]
        
        # Count status types
        critical_count = sum(1 for check in checks.values() if check["status"] == "critical")
        warning_count = sum(1 for check in checks.values() if check["status"] == "warning")
        healthy_count = sum(1 for check in checks.values() if check["status"] in ["healthy", "excellent"])
        
        # Collect all issues
        all_issues = []
        for check in checks.values():
            all_issues.extend(check.get("issues", []))
        
        self.health_results["alerts"] = all_issues
        
        # Determine overall status
        if critical_count > 0:
            self.health_results["overall_status"] = "critical"
        elif warning_count > 0:
            self.health_results["overall_status"] = "warning"
        elif healthy_count > 0:
            self.health_results["overall_status"] = "healthy"
        else:
            self.health_results["overall_status"] = "unknown"
        
        self.health_results["summary"] = {
            "total_checks": len(checks),
            "critical_checks": critical_count,
            "warning_checks": warning_count,
            "healthy_checks": healthy_count,
            "total_issues": len(all_issues)
        }
    
    def _generate_recommendations(self):
        """Generate health recommendations."""
        recommendations = []
        
        # Based on overall status
        if self.health_results["overall_status"] == "critical":
            recommendations.append("Immediate attention required - critical issues detected")
        elif self.health_results["overall_status"] == "warning":
            recommendations.append("Review and address warning issues")
        
        # Specific recommendations based on checks
        checks = self.health_results["checks"]
        
        # Local system recommendations
        local_system = checks.get("local_system", {})
        if local_system.get("status") == "critical":
            recommendations.append("Check system resources and model availability")
        
        # Model recommendations
        models = checks.get("models", {})
        if models.get("status") in ["critical", "warning"]:
            recommendations.append("Validate and update model configurations")
        
        # Monitoring recommendations
        monitoring = checks.get("monitoring", {})
        if monitoring.get("status") in ["critical", "warning"]:
            recommendations.append("Review monitoring alerts and accuracy metrics")
        
        # API recommendations
        api_endpoints = checks.get("api_endpoints", {})
        if api_endpoints.get("status") in ["critical", "warning"]:
            recommendations.append("Check API service status and network connectivity")
        
        # General recommendations
        metrics = self.health_results.get("metrics", {})
        
        if metrics.get("optimized_thresholds", 0) == 0:
            recommendations.append("Configure optimized thresholds for better accuracy")
        
        if metrics.get("total_predictions", 0) == 0:
            recommendations.append("Start processing requests to enable monitoring")
        
        if metrics.get("recent_alerts", 0) > 0:
            recommendations.append("Investigate and resolve recent accuracy alerts")
        
        self.health_results["recommendations"] = recommendations
    
    def save_report(self, output_file: str):
        """Save health check report to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.health_results, f, indent=2)
            logger.info(f"Health check report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save health check report: {str(e)}")
    
    def print_summary(self):
        """Print health check summary to console."""
        print("\n" + "="*80)
        print("SYSTEM HEALTH CHECK SUMMARY")
        print("="*80)
        
        print(f"Timestamp: {self.health_results['timestamp']}")
        print(f"Overall Status: {self.health_results['overall_status'].upper()}")
        
        # Print status icon
        status_icons = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓"
        }
        icon = status_icons.get(self.health_results['overall_status'], "❓")
        print(f"Status Icon: {icon}")
        
        summary = self.health_results.get("summary", {})
        print(f"\nCheck Results:")
        print(f"  Total Checks: {summary.get('total_checks', 0)}")
        print(f"  Healthy: {summary.get('healthy_checks', 0)}")
        print(f"  Warning: {summary.get('warning_checks', 0)}")
        print(f"  Critical: {summary.get('critical_checks', 0)}")
        print(f"  Total Issues: {summary.get('total_issues', 0)}")
        
        # Print key metrics
        metrics = self.health_results.get("metrics", {})
        if metrics:
            print(f"\nKey Metrics:")
            for key, value in metrics.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Print issues/alerts
        if self.health_results["alerts"]:
            print(f"\nISSUES ({len(self.health_results['alerts'])}):")
            for i, alert in enumerate(self.health_results["alerts"], 1):
                print(f"  {i}. {alert}")
        
        # Print recommendations
        if self.health_results["recommendations"]:
            print(f"\nRECOMMENDATIONS ({len(self.health_results['recommendations'])}):")
            for i, rec in enumerate(self.health_results["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Main function for health check script."""
    parser = argparse.ArgumentParser(description="Check AI Voice Detection System health")
    parser.add_argument(
        "--api-url",
        help="Base URL for API health checks (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only perform local system checks (skip API endpoints)"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only perform API endpoint checks (skip local system)"
    )
    parser.add_argument(
        "--output",
        help="Output file for health check report (JSON format)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine what checks to run
    include_api = not args.local_only
    include_local = not args.api_only
    
    # Run health check
    health_checker = HealthChecker(args.api_url)
    success = health_checker.check_all(include_api=include_api, include_local=include_local)
    
    # Print summary
    health_checker.print_summary()
    
    # Save report if requested
    if args.output:
        health_checker.save_report(args.output)
    
    # Exit with appropriate code
    if success:
        print(f"\n✅ System health check PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ System health check FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()