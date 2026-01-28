#!/usr/bin/env python3
"""
Deployment Validation Script for AI Voice Detection System

This script validates the deployment environment, model availability,
configuration correctness, and system health before deployment.
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.config_manager import ConfigurationManager, DeploymentEnvironment
from src.validation.validator import ModelValidator
from src.detection.engine import DetectionEngine
from src.monitoring.accuracy_monitor import AccuracyMonitor
from src.registry.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentValidator:
    """
    Comprehensive deployment validator for the AI voice detection system.
    
    Validates configuration, models, system health, and deployment readiness.
    """
    
    def __init__(self, environment: str = "production"):
        """
        Initialize DeploymentValidator.
        
        Args:
            environment: Target deployment environment
        """
        self.environment = DeploymentEnvironment(environment)
        self.settings = get_settings()
        self.config_manager = ConfigurationManager()
        
        # Validation results
        self.validation_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "overall_status": "unknown",
            "validations": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        logger.info(f"DeploymentValidator initialized for {environment} environment")
    
    def validate_all(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if all validations pass, False otherwise
        """
        logger.info("Starting comprehensive deployment validation...")
        
        try:
            # Run all validation checks
            self._validate_configuration()
            self._validate_environment_variables()
            self._validate_file_system()
            self._validate_models()
            self._validate_system_components()
            self._validate_network_configuration()
            self._validate_security_settings()
            self._validate_performance_requirements()
            
            # Determine overall status
            self._determine_overall_status()
            
            # Generate recommendations
            self._generate_recommendations()
            
            logger.info(f"Deployment validation completed: {self.validation_results['overall_status']}")
            return self.validation_results['overall_status'] == "ready"
            
        except Exception as e:
            logger.error(f"Deployment validation failed: {str(e)}")
            self.validation_results['overall_status'] = "failed"
            self.validation_results['errors'].append(f"Validation process failed: {str(e)}")
            return False
    
    def _validate_configuration(self):
        """Validate configuration files and settings."""
        logger.info("Validating configuration...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate configuration manager
            config_summary = self.config_manager.get_configuration_summary()
            validation["details"]["config_summary"] = config_summary
            
            # Check threshold configurations
            thresholds = self.config_manager.get_all_thresholds()
            if not thresholds:
                validation["warnings"].append("No optimized thresholds configured")
            else:
                validation["details"]["thresholds_count"] = len(thresholds)
            
            # Validate model paths
            missing_paths = self.config_manager.validate_model_paths()
            if missing_paths:
                validation["errors"].extend([f"Missing model path: {path}" for path in missing_paths])
            
            # Validate deployment configuration
            deployment_errors = self.config_manager.validate_deployment_config()
            if deployment_errors:
                validation["errors"].extend(deployment_errors)
            
            # Check environment-specific settings
            if self.environment == DeploymentEnvironment.PRODUCTION:
                if self.settings.debug:
                    validation["errors"].append("Debug mode enabled in production")
                if not self.settings.security.ssl_enabled:
                    validation["warnings"].append("SSL not enabled in production")
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"Configuration validation error: {str(e)}")
        
        self.validation_results["validations"]["configuration"] = validation
    
    def _validate_environment_variables(self):
        """Validate required environment variables."""
        logger.info("Validating environment variables...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Required environment variables
            required_vars = [
                "ENVIRONMENT",
                "API_HOST",
                "API_PORT",
                "MODELS_BASE_PATH"
            ]
            
            # Production-specific requirements
            if self.environment == DeploymentEnvironment.PRODUCTION:
                required_vars.extend([
                    "API_KEYS",
                    "LOG_LEVEL"
                ])
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                validation["errors"].extend([f"Missing environment variable: {var}" for var in missing_vars])
            
            # Check API keys
            api_keys = self.settings.auth.api_keys
            if not api_keys or (len(api_keys) == 1 and api_keys[0] == "RW-b3ZMf29EcBQLObtVffHiqine2b89qzlq3Hgg2pBUoqyIElXhg0DKleUAeZsXNIdK"):
                validation["warnings"].append("Using default API key - should be changed in production")
            
            validation["details"]["api_keys_count"] = len(api_keys)
            validation["details"]["environment"] = os.getenv("ENVIRONMENT", "unknown")
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"Environment variables validation error: {str(e)}")
        
        self.validation_results["validations"]["environment_variables"] = validation
    
    def _validate_file_system(self):
        """Validate file system permissions and directory structure."""
        logger.info("Validating file system...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required directories
            required_dirs = [
                self.settings.ml.models_base_path,
                str(Path(self.settings.ml.models_base_path) / "classifiers"),
                str(Path(self.settings.ml.models_base_path) / "foundation"),
                str(Path(self.settings.ml.models_base_path) / "registry"),
                ".kiro/config"
            ]
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                if not path.exists():
                    validation["errors"].append(f"Required directory missing: {dir_path}")
                elif not path.is_dir():
                    validation["errors"].append(f"Path is not a directory: {dir_path}")
                elif not os.access(path, os.R_OK | os.W_OK):
                    validation["errors"].append(f"Insufficient permissions for directory: {dir_path}")
            
            # Check disk space
            models_path = Path(self.settings.ml.models_base_path)
            if models_path.exists():
                import shutil
                total, used, free = shutil.disk_usage(models_path)
                free_gb = free / (1024**3)
                validation["details"]["free_disk_space_gb"] = round(free_gb, 2)
                
                if free_gb < 1.0:  # Less than 1GB free
                    validation["errors"].append(f"Low disk space: {free_gb:.2f}GB free")
                elif free_gb < 5.0:  # Less than 5GB free
                    validation["warnings"].append(f"Limited disk space: {free_gb:.2f}GB free")
            
            # Check log file permissions if configured
            if self.settings.logging.log_file:
                log_path = Path(self.settings.logging.log_file)
                log_dir = log_path.parent
                if not log_dir.exists():
                    try:
                        log_dir.mkdir(parents=True)
                    except Exception as e:
                        validation["errors"].append(f"Cannot create log directory: {str(e)}")
                elif not os.access(log_dir, os.W_OK):
                    validation["errors"].append(f"Cannot write to log directory: {log_dir}")
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"File system validation error: {str(e)}")
        
        self.validation_results["validations"]["file_system"] = validation
    
    def _validate_models(self):
        """Validate model files and their integrity."""
        logger.info("Validating models...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Initialize model validator
            model_validator = ModelValidator(self.settings)
            
            # Validate foundation models
            foundation_results = {}
            foundation_models = ["xls-r-300m", "hubert-base", "wav2vec2-base"]
            
            for model_name in foundation_models:
                model_path = Path(self.settings.ml.models_base_path) / "foundation" / model_name
                if model_path.exists():
                    result = model_validator.validate_foundation_model(str(model_path))
                    foundation_results[model_name] = {
                        "valid": result.is_valid,
                        "confidence": result.confidence_score,
                        "error": result.error_message
                    }
                    if not result.is_valid:
                        validation["errors"].append(f"Foundation model {model_name} validation failed: {result.error_message}")
                else:
                    foundation_results[model_name] = {"valid": False, "error": "Model not found"}
                    validation["warnings"].append(f"Foundation model {model_name} not found")
            
            validation["details"]["foundation_models"] = foundation_results
            
            # Validate classifiers
            classifier_results = {}
            languages = ["English", "Tamil", "Hindi", "Malayalam", "Telugu"]
            
            for language in languages:
                classifier_path = Path(self.settings.ml.models_base_path) / "classifiers" / f"{language.lower()}_classifier.h5"
                if classifier_path.exists():
                    result = model_validator.validate_classifier(str(classifier_path), language)
                    classifier_results[language] = {
                        "valid": result.is_valid,
                        "confidence": result.confidence_score,
                        "error": result.error_message,
                        "file_size_mb": round(classifier_path.stat().st_size / (1024*1024), 2)
                    }
                    if not result.is_valid:
                        validation["errors"].append(f"Classifier {language} validation failed: {result.error_message}")
                else:
                    classifier_results[language] = {"valid": False, "error": "Model not found"}
                    validation["errors"].append(f"Classifier for {language} not found")
            
            validation["details"]["classifiers"] = classifier_results
            
            # Check if at least one foundation model and some classifiers are valid
            valid_foundation = any(result["valid"] for result in foundation_results.values())
            valid_classifiers = sum(1 for result in classifier_results.values() if result["valid"])
            
            if not valid_foundation:
                validation["errors"].append("No valid foundation models found")
            
            if valid_classifiers == 0:
                validation["errors"].append("No valid classifiers found")
            elif valid_classifiers < len(languages):
                validation["warnings"].append(f"Only {valid_classifiers}/{len(languages)} classifiers are valid")
            
            validation["details"]["summary"] = {
                "valid_foundation_models": sum(1 for result in foundation_results.values() if result["valid"]),
                "valid_classifiers": valid_classifiers,
                "total_classifiers": len(languages)
            }
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"Model validation error: {str(e)}")
        
        self.validation_results["validations"]["models"] = validation
    
    def _validate_system_components(self):
        """Validate system components initialization."""
        logger.info("Validating system components...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test DetectionEngine initialization
            try:
                detection_engine = DetectionEngine(self.settings)
                health_status = detection_engine.get_system_health_status()
                
                validation["details"]["detection_engine"] = {
                    "initialized": True,
                    "health_score": health_status.get("overall_health", {}).get("score", 0.0),
                    "foundation_model": health_status.get("foundation_model", {}).get("name", "unknown")
                }
                
                # Check health score
                health_score = health_status.get("overall_health", {}).get("score", 0.0)
                if health_score < 0.6:
                    validation["errors"].append(f"Low system health score: {health_score:.2f}")
                elif health_score < 0.8:
                    validation["warnings"].append(f"Moderate system health score: {health_score:.2f}")
                
            except Exception as e:
                validation["errors"].append(f"DetectionEngine initialization failed: {str(e)}")
                validation["details"]["detection_engine"] = {"initialized": False, "error": str(e)}
            
            # Test AccuracyMonitor initialization
            try:
                accuracy_monitor = AccuracyMonitor(self.settings)
                monitoring_summary = accuracy_monitor.get_monitoring_summary()
                
                validation["details"]["accuracy_monitor"] = {
                    "initialized": True,
                    "window_size": monitoring_summary.get("window_size", 0),
                    "degradation_threshold": monitoring_summary.get("degradation_threshold", 0.0)
                }
                
            except Exception as e:
                validation["warnings"].append(f"AccuracyMonitor initialization failed: {str(e)}")
                validation["details"]["accuracy_monitor"] = {"initialized": False, "error": str(e)}
            
            # Test ModelRegistry initialization
            try:
                registry_path = Path(self.settings.ml.models_base_path) / "registry"
                model_registry = ModelRegistry(str(registry_path))
                registry_stats = model_registry.get_registry_stats()
                
                validation["details"]["model_registry"] = {
                    "initialized": True,
                    "total_models": registry_stats.get("total_models", 0),
                    "models_by_type": registry_stats.get("models_by_type", {})
                }
                
            except Exception as e:
                validation["warnings"].append(f"ModelRegistry initialization failed: {str(e)}")
                validation["details"]["model_registry"] = {"initialized": False, "error": str(e)}
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"System components validation error: {str(e)}")
        
        self.validation_results["validations"]["system_components"] = validation
    
    def _validate_network_configuration(self):
        """Validate network and API configuration."""
        logger.info("Validating network configuration...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check port availability
            import socket
            
            host = self.settings.api.host
            port = self.settings.api.port
            
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex((host, port))
                    if result == 0:
                        validation["warnings"].append(f"Port {port} is already in use")
                    else:
                        validation["details"]["port_available"] = True
            except Exception as e:
                validation["warnings"].append(f"Could not check port availability: {str(e)}")
            
            # Validate CORS settings
            cors_origins = self.settings.api.cors_origins
            if "*" in cors_origins and self.environment == DeploymentEnvironment.PRODUCTION:
                validation["warnings"].append("CORS allows all origins in production - security risk")
            
            validation["details"]["cors_origins"] = cors_origins
            validation["details"]["api_endpoint"] = f"{host}:{port}"
            
            # Check SSL configuration
            if self.settings.security.ssl_enabled:
                ssl_cert = self.settings.security.ssl_cert_file
                ssl_key = self.settings.security.ssl_key_file
                
                if not ssl_cert or not Path(ssl_cert).exists():
                    validation["errors"].append(f"SSL certificate not found: {ssl_cert}")
                
                if not ssl_key or not Path(ssl_key).exists():
                    validation["errors"].append(f"SSL key not found: {ssl_key}")
                
                validation["details"]["ssl_enabled"] = True
            else:
                validation["details"]["ssl_enabled"] = False
                if self.environment == DeploymentEnvironment.PRODUCTION:
                    validation["warnings"].append("SSL not enabled in production")
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"Network configuration validation error: {str(e)}")
        
        self.validation_results["validations"]["network_configuration"] = validation
    
    def _validate_security_settings(self):
        """Validate security configuration."""
        logger.info("Validating security settings...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check API key security
            api_keys = self.settings.auth.api_keys
            
            for i, key in enumerate(api_keys):
                if len(key) < 32:
                    validation["warnings"].append(f"API key {i+1} is too short (< 32 characters)")
                
                # Check for default/weak keys
                if key == "RW-b3ZMf29EcBQLObtVffHiqine2b89qzlq3Hgg2pBUoqyIElXhg0DKleUAeZsXNIdK":
                    validation["errors"].append("Using default API key - must be changed")
            
            validation["details"]["api_keys_count"] = len(api_keys)
            
            # Check request limits
            max_request_size = self.settings.security.max_request_size
            if max_request_size > 100 * 1024 * 1024:  # 100MB
                validation["warnings"].append(f"Large max request size: {max_request_size / (1024*1024):.1f}MB")
            
            validation["details"]["max_request_size_mb"] = round(max_request_size / (1024*1024), 1)
            validation["details"]["request_timeout"] = self.settings.security.request_timeout
            
            # Check security headers
            if not self.settings.security.security_headers_enabled:
                validation["warnings"].append("Security headers disabled")
            
            validation["details"]["security_headers_enabled"] = self.settings.security.security_headers_enabled
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"Security settings validation error: {str(e)}")
        
        self.validation_results["validations"]["security_settings"] = validation
    
    def _validate_performance_requirements(self):
        """Validate performance requirements and system resources."""
        logger.info("Validating performance requirements...")
        
        validation = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check system resources
            try:
                import psutil
                
                # Memory check
                memory = psutil.virtual_memory()
                memory_gb = memory.total / (1024**3)
                validation["details"]["total_memory_gb"] = round(memory_gb, 2)
                validation["details"]["available_memory_gb"] = round(memory.available / (1024**3), 2)
                
                if memory_gb < 2.0:
                    validation["errors"].append(f"Insufficient memory: {memory_gb:.1f}GB (minimum 2GB required)")
                elif memory_gb < 4.0:
                    validation["warnings"].append(f"Limited memory: {memory_gb:.1f}GB (4GB+ recommended)")
                
                # CPU check
                cpu_count = psutil.cpu_count()
                validation["details"]["cpu_count"] = cpu_count
                
                if cpu_count < 2:
                    validation["warnings"].append(f"Limited CPU cores: {cpu_count} (2+ recommended)")
                
                # Disk space check (already done in file system validation)
                
            except ImportError:
                validation["warnings"].append("psutil not available - cannot check system resources")
            
            # Check worker configuration
            workers = self.settings.api.workers
            validation["details"]["configured_workers"] = workers
            
            if workers > 4:
                validation["warnings"].append(f"High worker count: {workers} (may consume significant resources)")
            
            # Check timeout settings
            timeout = self.settings.security.request_timeout
            validation["details"]["request_timeout"] = timeout
            
            if timeout > 300:  # 5 minutes
                validation["warnings"].append(f"Long request timeout: {timeout}s")
            elif timeout < 30:  # 30 seconds
                validation["warnings"].append(f"Short request timeout: {timeout}s (may cause timeouts for large files)")
            
            validation["status"] = "passed" if not validation["errors"] else "failed"
            
        except Exception as e:
            validation["status"] = "failed"
            validation["errors"].append(f"Performance requirements validation error: {str(e)}")
        
        self.validation_results["validations"]["performance_requirements"] = validation
    
    def _determine_overall_status(self):
        """Determine overall deployment readiness status."""
        total_validations = len(self.validation_results["validations"])
        passed_validations = sum(1 for v in self.validation_results["validations"].values() 
                               if v["status"] == "passed")
        failed_validations = sum(1 for v in self.validation_results["validations"].values() 
                               if v["status"] == "failed")
        
        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        
        for validation in self.validation_results["validations"].values():
            all_errors.extend(validation.get("errors", []))
            all_warnings.extend(validation.get("warnings", []))
        
        self.validation_results["errors"] = all_errors
        self.validation_results["warnings"] = all_warnings
        
        # Determine status
        if failed_validations == 0:
            if len(all_warnings) == 0:
                self.validation_results["overall_status"] = "ready"
            else:
                self.validation_results["overall_status"] = "ready_with_warnings"
        elif failed_validations <= 2 and len(all_errors) <= 3:
            self.validation_results["overall_status"] = "needs_attention"
        else:
            self.validation_results["overall_status"] = "not_ready"
        
        self.validation_results["summary"] = {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings)
        }
    
    def _generate_recommendations(self):
        """Generate deployment recommendations based on validation results."""
        recommendations = []
        
        # Based on overall status
        if self.validation_results["overall_status"] == "not_ready":
            recommendations.append("Address all critical errors before deployment")
        elif self.validation_results["overall_status"] == "needs_attention":
            recommendations.append("Review and address validation errors")
        
        # Specific recommendations based on validation results
        validations = self.validation_results["validations"]
        
        # Configuration recommendations
        if validations.get("configuration", {}).get("status") == "failed":
            recommendations.append("Fix configuration issues before deployment")
        
        # Model recommendations
        models_validation = validations.get("models", {})
        if models_validation.get("status") == "failed":
            recommendations.append("Ensure all required models are available and valid")
        elif models_validation.get("details", {}).get("summary", {}).get("valid_classifiers", 0) < 5:
            recommendations.append("Consider adding more language classifiers for better coverage")
        
        # Security recommendations
        if validations.get("security_settings", {}).get("status") == "failed":
            recommendations.append("Address security configuration issues")
        
        if self.environment == DeploymentEnvironment.PRODUCTION:
            recommendations.extend([
                "Ensure SSL is properly configured for production",
                "Use strong, unique API keys",
                "Configure appropriate CORS origins",
                "Enable security headers",
                "Set up monitoring and alerting",
                "Configure log rotation and retention"
            ])
        
        self.validation_results["recommendations"] = recommendations
    
    def save_report(self, output_file: str):
        """Save validation report to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            logger.info(f"Validation report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")
    
    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "="*80)
        print("DEPLOYMENT VALIDATION SUMMARY")
        print("="*80)
        
        print(f"Environment: {self.validation_results['environment']}")
        print(f"Timestamp: {self.validation_results['timestamp']}")
        print(f"Overall Status: {self.validation_results['overall_status'].upper()}")
        
        summary = self.validation_results.get("summary", {})
        print(f"\nValidation Results:")
        print(f"  Total Validations: {summary.get('total_validations', 0)}")
        print(f"  Passed: {summary.get('passed_validations', 0)}")
        print(f"  Failed: {summary.get('failed_validations', 0)}")
        print(f"  Errors: {summary.get('total_errors', 0)}")
        print(f"  Warnings: {summary.get('total_warnings', 0)}")
        
        # Print errors
        if self.validation_results["errors"]:
            print(f"\nERRORS ({len(self.validation_results['errors'])}):")
            for i, error in enumerate(self.validation_results["errors"], 1):
                print(f"  {i}. {error}")
        
        # Print warnings
        if self.validation_results["warnings"]:
            print(f"\nWARNINGS ({len(self.validation_results['warnings'])}):")
            for i, warning in enumerate(self.validation_results["warnings"], 1):
                print(f"  {i}. {warning}")
        
        # Print recommendations
        if self.validation_results["recommendations"]:
            print(f"\nRECOMMENDATIONS ({len(self.validation_results['recommendations'])}):")
            for i, rec in enumerate(self.validation_results["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Main function for deployment validation script."""
    parser = argparse.ArgumentParser(description="Validate AI Voice Detection System deployment")
    parser.add_argument(
        "--environment", 
        choices=["development", "staging", "production", "testing"],
        default="production",
        help="Target deployment environment"
    )
    parser.add_argument(
        "--output", 
        help="Output file for validation report (JSON format)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = DeploymentValidator(args.environment)
    success = validator.validate_all()
    
    # Print summary
    validator.print_summary()
    
    # Save report if requested
    if args.output:
        validator.save_report(args.output)
    
    # Exit with appropriate code
    if success:
        print("\n✅ Deployment validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Deployment validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()