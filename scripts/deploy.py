#!/usr/bin/env python3
"""
Main Deployment Script for AI Voice Detection System

This script orchestrates the complete deployment process including
configuration, validation, and health checking.
"""

import sys
import os
import json
import logging
import argparse
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """
    Main deployment orchestrator for the AI voice detection system.
    
    Coordinates configuration, validation, and deployment processes.
    """
    
    def __init__(self, environment: str = "production"):
        """
        Initialize DeploymentOrchestrator.
        
        Args:
            environment: Target deployment environment
        """
        self.environment = environment
        self.scripts_dir = Path(__file__).parent
        self.project_root = self.scripts_dir.parent
        
        # Deployment results
        self.deployment_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "status": "unknown",
            "steps": {},
            "errors": [],
            "warnings": []
        }
        
        logger.info(f"DeploymentOrchestrator initialized for {environment} environment")
    
    def deploy(self, **kwargs) -> bool:
        """
        Execute complete deployment process.
        
        Args:
            **kwargs: Deployment configuration options
            
        Returns:
            True if deployment successful, False otherwise
        """
        logger.info("Starting deployment process...")
        
        try:
            # Step 1: Configure deployment
            if not self._configure_deployment(**kwargs):
                return False
            
            # Step 2: Validate deployment
            if not self._validate_deployment():
                return False
            
            # Step 3: Setup system
            if not self._setup_system(**kwargs):
                return False
            
            # Step 4: Final health check
            if not self._final_health_check():
                return False
            
            # Step 5: Generate deployment report
            self._generate_deployment_report()
            
            self.deployment_results["status"] = "success"
            logger.info("Deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            self.deployment_results["status"] = "failed"
            self.deployment_results["errors"].append(f"Deployment process failed: {str(e)}")
            return False
    
    def _configure_deployment(self, **kwargs) -> bool:
        """Configure deployment settings."""
        logger.info("Step 1: Configuring deployment...")
        
        step_result = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Build configuration command
            config_cmd = [
                sys.executable,
                str(self.scripts_dir / "configure_deployment.py"),
                "--environment", self.environment,
                "--export-dir", kwargs.get("export_dir", "deployment_config")
            ]
            
            # Add optional parameters
            if kwargs.get("models_path"):
                config_cmd.extend(["--models-path", kwargs["models_path"]])
            if kwargs.get("api_host"):
                config_cmd.extend(["--api-host", kwargs["api_host"]])
            if kwargs.get("api_port"):
                config_cmd.extend(["--api-port", str(kwargs["api_port"])])
            if kwargs.get("workers"):
                config_cmd.extend(["--workers", str(kwargs["workers"])])
            if kwargs.get("setup_defaults"):
                config_cmd.append("--setup-defaults")
            if kwargs.get("verbose"):
                config_cmd.append("--verbose")
            
            # Execute configuration
            result = subprocess.run(
                config_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            step_result["details"]["command"] = " ".join(config_cmd)
            step_result["details"]["return_code"] = result.returncode
            step_result["details"]["stdout"] = result.stdout
            step_result["details"]["stderr"] = result.stderr
            
            if result.returncode == 0:
                step_result["status"] = "success"
                logger.info("Deployment configuration completed successfully")
            else:
                step_result["status"] = "failed"
                step_result["errors"].append(f"Configuration failed with return code {result.returncode}")
                if result.stderr:
                    step_result["errors"].append(f"Configuration error: {result.stderr}")
            
        except Exception as e:
            step_result["status"] = "failed"
            step_result["errors"].append(f"Configuration step failed: {str(e)}")
        
        self.deployment_results["steps"]["configure"] = step_result
        return step_result["status"] == "success"
    
    def _validate_deployment(self) -> bool:
        """Validate deployment readiness."""
        logger.info("Step 2: Validating deployment...")
        
        step_result = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Build validation command
            validate_cmd = [
                sys.executable,
                str(self.scripts_dir / "validate_deployment.py"),
                "--environment", self.environment,
                "--output", "deployment_validation.json"
            ]
            
            if self.deployment_results.get("verbose"):
                validate_cmd.append("--verbose")
            
            # Execute validation
            result = subprocess.run(
                validate_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            step_result["details"]["command"] = " ".join(validate_cmd)
            step_result["details"]["return_code"] = result.returncode
            step_result["details"]["stdout"] = result.stdout
            step_result["details"]["stderr"] = result.stderr
            
            # Load validation results if available
            validation_file = self.project_root / "deployment_validation.json"
            if validation_file.exists():
                try:
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                        step_result["details"]["validation_results"] = validation_data
                        
                        # Extract key metrics
                        overall_status = validation_data.get("overall_status", "unknown")
                        errors = validation_data.get("errors", [])
                        warnings = validation_data.get("warnings", [])
                        
                        step_result["details"]["overall_status"] = overall_status
                        step_result["errors"].extend(errors)
                        step_result["warnings"].extend(warnings)
                        
                except Exception as e:
                    step_result["warnings"].append(f"Could not parse validation results: {str(e)}")
            
            if result.returncode == 0:
                step_result["status"] = "success"
                logger.info("Deployment validation passed")
            else:
                step_result["status"] = "failed"
                step_result["errors"].append(f"Validation failed with return code {result.returncode}")
                if result.stderr:
                    step_result["errors"].append(f"Validation error: {result.stderr}")
            
        except Exception as e:
            step_result["status"] = "failed"
            step_result["errors"].append(f"Validation step failed: {str(e)}")
        
        self.deployment_results["steps"]["validate"] = step_result
        return step_result["status"] == "success"
    
    def _setup_system(self, **kwargs) -> bool:
        """Setup system components."""
        logger.info("Step 3: Setting up system...")
        
        step_result = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create required directories
            required_dirs = [
                "logs",
                "models",
                "models/classifiers",
                "models/foundation",
                "models/explanation",
                "models/registry",
                ".kiro",
                ".kiro/config"
            ]
            
            created_dirs = []
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
            
            step_result["details"]["created_directories"] = created_dirs
            
            # Copy configuration files if they exist
            config_dir = self.project_root / kwargs.get("export_dir", "deployment_config")
            if config_dir.exists():
                env_file = config_dir / f".env.{self.environment}"
                if env_file.exists():
                    target_env = self.project_root / f".env.{self.environment}"
                    import shutil
                    shutil.copy2(env_file, target_env)
                    step_result["details"]["copied_env_file"] = str(target_env)
            
            # Set permissions (if on Unix-like system)
            if os.name != 'nt':  # Not Windows
                try:
                    import stat
                    # Make scripts executable
                    for script in ["validate_deployment.py", "health_check.py", "configure_deployment.py"]:
                        script_path = self.scripts_dir / script
                        if script_path.exists():
                            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
                    step_result["details"]["set_permissions"] = True
                except Exception as e:
                    step_result["warnings"].append(f"Could not set script permissions: {str(e)}")
            
            step_result["status"] = "success"
            logger.info("System setup completed")
            
        except Exception as e:
            step_result["status"] = "failed"
            step_result["errors"].append(f"System setup failed: {str(e)}")
        
        self.deployment_results["steps"]["setup"] = step_result
        return step_result["status"] == "success"
    
    def _final_health_check(self) -> bool:
        """Perform final health check."""
        logger.info("Step 4: Performing final health check...")
        
        step_result = {
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Build health check command
            health_cmd = [
                sys.executable,
                str(self.scripts_dir / "health_check.py"),
                "--local-only",  # Only local checks for deployment validation
                "--output", "deployment_health.json"
            ]
            
            if self.deployment_results.get("verbose"):
                health_cmd.append("--verbose")
            
            # Execute health check
            result = subprocess.run(
                health_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            step_result["details"]["command"] = " ".join(health_cmd)
            step_result["details"]["return_code"] = result.returncode
            step_result["details"]["stdout"] = result.stdout
            step_result["details"]["stderr"] = result.stderr
            
            # Load health check results if available
            health_file = self.project_root / "deployment_health.json"
            if health_file.exists():
                try:
                    with open(health_file, 'r') as f:
                        health_data = json.load(f)
                        step_result["details"]["health_results"] = health_data
                        
                        # Extract key metrics
                        overall_status = health_data.get("overall_status", "unknown")
                        alerts = health_data.get("alerts", [])
                        recommendations = health_data.get("recommendations", [])
                        
                        step_result["details"]["overall_status"] = overall_status
                        step_result["warnings"].extend(alerts)
                        step_result["details"]["recommendations"] = recommendations
                        
                except Exception as e:
                    step_result["warnings"].append(f"Could not parse health check results: {str(e)}")
            
            if result.returncode == 0:
                step_result["status"] = "success"
                logger.info("Final health check passed")
            else:
                step_result["status"] = "warning"  # Health check failure is warning, not error
                step_result["warnings"].append(f"Health check returned code {result.returncode}")
                if result.stderr:
                    step_result["warnings"].append(f"Health check warning: {result.stderr}")
            
        except Exception as e:
            step_result["status"] = "warning"
            step_result["warnings"].append(f"Health check step failed: {str(e)}")
        
        self.deployment_results["steps"]["health_check"] = step_result
        return step_result["status"] in ["success", "warning"]
    
    def _generate_deployment_report(self):
        """Generate comprehensive deployment report."""
        logger.info("Generating deployment report...")
        
        try:
            # Collect all errors and warnings
            all_errors = []
            all_warnings = []
            
            for step_name, step_data in self.deployment_results["steps"].items():
                all_errors.extend(step_data.get("errors", []))
                all_warnings.extend(step_data.get("warnings", []))
            
            self.deployment_results["errors"] = all_errors
            self.deployment_results["warnings"] = all_warnings
            
            # Generate summary
            total_steps = len(self.deployment_results["steps"])
            successful_steps = sum(1 for step in self.deployment_results["steps"].values() 
                                 if step["status"] == "success")
            failed_steps = sum(1 for step in self.deployment_results["steps"].values() 
                             if step["status"] == "failed")
            
            self.deployment_results["summary"] = {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "total_errors": len(all_errors),
                "total_warnings": len(all_warnings)
            }
            
            # Save deployment report
            report_file = self.project_root / "deployment_report.json"
            with open(report_file, 'w') as f:
                json.dump(self.deployment_results, f, indent=2)
            
            logger.info(f"Deployment report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate deployment report: {str(e)}")
    
    def print_summary(self):
        """Print deployment summary to console."""
        print("\n" + "="*80)
        print("DEPLOYMENT SUMMARY")
        print("="*80)
        
        print(f"Environment: {self.deployment_results['environment']}")
        print(f"Timestamp: {self.deployment_results['timestamp']}")
        print(f"Status: {self.deployment_results['status'].upper()}")
        
        summary = self.deployment_results.get("summary", {})
        print(f"\nStep Results:")
        print(f"  Total Steps: {summary.get('total_steps', 0)}")
        print(f"  Successful: {summary.get('successful_steps', 0)}")
        print(f"  Failed: {summary.get('failed_steps', 0)}")
        print(f"  Errors: {summary.get('total_errors', 0)}")
        print(f"  Warnings: {summary.get('total_warnings', 0)}")
        
        # Print step details
        print(f"\nStep Details:")
        for step_name, step_data in self.deployment_results["steps"].items():
            status_icon = "✅" if step_data["status"] == "success" else "⚠️" if step_data["status"] == "warning" else "❌"
            print(f"  {status_icon} {step_name.title()}: {step_data['status'].upper()}")
        
        # Print errors
        if self.deployment_results["errors"]:
            print(f"\nERRORS ({len(self.deployment_results['errors'])}):")
            for i, error in enumerate(self.deployment_results["errors"], 1):
                print(f"  {i}. {error}")
        
        # Print warnings
        if self.deployment_results["warnings"]:
            print(f"\nWARNINGS ({len(self.deployment_results['warnings'])}):")
            for i, warning in enumerate(self.deployment_results["warnings"], 1):
                print(f"  {i}. {warning}")
        
        # Print next steps
        print(f"\nNEXT STEPS:")
        if self.deployment_results["status"] == "success":
            print("  1. Review deployment report and configuration files")
            print("  2. Test the API endpoints")
            print("  3. Set up monitoring and alerting")
            print("  4. Configure log rotation and backup procedures")
        else:
            print("  1. Review and fix deployment errors")
            print("  2. Re-run deployment process")
            print("  3. Check system requirements and dependencies")
        
        print("\n" + "="*80)


def main():
    """Main function for deployment script."""
    parser = argparse.ArgumentParser(description="Deploy AI Voice Detection System")
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production", "testing"],
        default="production",
        help="Target deployment environment"
    )
    parser.add_argument(
        "--models-path",
        help="Base path for model files"
    )
    parser.add_argument(
        "--api-host",
        help="API host address"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        help="API port number"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of API workers"
    )
    parser.add_argument(
        "--export-dir",
        default="deployment_config",
        help="Directory to export configuration files"
    )
    parser.add_argument(
        "--setup-defaults",
        action="store_true",
        help="Setup default thresholds and model paths"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare deployment options
    deploy_options = {
        "models_path": args.models_path,
        "api_host": args.api_host,
        "api_port": args.api_port,
        "workers": args.workers,
        "export_dir": args.export_dir,
        "setup_defaults": args.setup_defaults,
        "verbose": args.verbose
    }
    
    # Remove None values
    deploy_options = {k: v for k, v in deploy_options.items() if v is not None}
    
    # Execute deployment
    orchestrator = DeploymentOrchestrator(args.environment)
    success = orchestrator.deploy(**deploy_options)
    
    # Print summary
    orchestrator.print_summary()
    
    # Exit with appropriate code
    if success:
        print(f"\n🚀 Deployment completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()