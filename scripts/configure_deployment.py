#!/usr/bin/env python3
"""
Deployment Configuration Script for AI Voice Detection System

This script helps configure the system for different deployment environments
with appropriate settings, thresholds, and model paths.
"""

import sys
import os
import json
import logging
import argparse
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.config_manager import ConfigurationManager, DeploymentEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentConfigurator:
    """
    Deployment configuration manager for the AI voice detection system.
    
    Handles environment-specific configuration, model setup, and deployment preparation.
    """
    
    def __init__(self, environment: str = "production"):
        """
        Initialize DeploymentConfigurator.
        
        Args:
            environment: Target deployment environment
        """
        self.environment = DeploymentEnvironment(environment)
        self.settings = get_settings()
        self.config_manager = ConfigurationManager()
        
        logger.info(f"DeploymentConfigurator initialized for {environment} environment")
    
    def configure_for_environment(self, **overrides) -> bool:
        """
        Configure system for specific deployment environment.
        
        Args:
            **overrides: Configuration overrides
            
        Returns:
            True if configuration successful, False otherwise
        """
        logger.info(f"Configuring system for {self.environment.value} environment...")
        
        try:
            # Environment-specific configurations
            if self.environment == DeploymentEnvironment.PRODUCTION:
                self._configure_production(**overrides)
            elif self.environment == DeploymentEnvironment.STAGING:
                self._configure_staging(**overrides)
            elif self.environment == DeploymentEnvironment.DEVELOPMENT:
                self._configure_development(**overrides)
            elif self.environment == DeploymentEnvironment.TESTING:
                self._configure_testing(**overrides)
            
            # Validate configuration
            validation_errors = self.config_manager.validate_deployment_config()
            if validation_errors:
                logger.error(f"Configuration validation failed: {validation_errors}")
                return False
            
            logger.info(f"Successfully configured system for {self.environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration failed: {str(e)}")
            return False
    
    def _configure_production(self, **overrides):
        """Configure for production environment."""
        logger.info("Applying production configuration...")
        
        # Production deployment settings
        production_config = {
            "environment": "production",
            "api_host": overrides.get("api_host", "0.0.0.0"),
            "api_port": overrides.get("api_port", 8000),
            "workers": overrides.get("workers", 4),
            "log_level": overrides.get("log_level", "INFO"),
            "enable_monitoring": True,
            "enable_validation": True,
            "enable_fallback": True,
            "max_request_size": overrides.get("max_request_size", 50 * 1024 * 1024),  # 50MB
            "request_timeout": overrides.get("request_timeout", 300),  # 5 minutes
            "cors_origins": overrides.get("cors_origins", ["https://yourdomain.com"]),
            "ssl_enabled": overrides.get("ssl_enabled", True),
            "ssl_cert_path": overrides.get("ssl_cert_path", "/etc/ssl/certs/server.crt"),
            "ssl_key_path": overrides.get("ssl_key_path", "/etc/ssl/private/server.key")
        }
        
        self.config_manager.update_deployment_config(**production_config)
        
        # Production system configuration
        self.config_manager.update_system_config("monitoring", "window_size", 2000)
        self.config_manager.update_system_config("monitoring", "degradation_threshold", 0.03)
        self.config_manager.update_system_config("monitoring", "retention_days", 90)
        
        self.config_manager.update_system_config("validation", "startup_validation", True)
        self.config_manager.update_system_config("validation", "periodic_validation", True)
        self.config_manager.update_system_config("validation", "validation_interval_hours", 12)
        
        self.config_manager.update_system_config("fallback", "confidence_threshold", 0.8)
        self.config_manager.update_system_config("fallback", "memory_limit_mb", 4096)
        self.config_manager.update_system_config("fallback", "max_processing_time_seconds", 60)
        
        self.config_manager.update_system_config("optimization", "auto_threshold_optimization", True)
        self.config_manager.update_system_config("optimization", "optimization_interval_hours", 168)  # Weekly
        
        logger.info("Production configuration applied")
    
    def _configure_staging(self, **overrides):
        """Configure for staging environment."""
        logger.info("Applying staging configuration...")
        
        # Staging deployment settings
        staging_config = {
            "environment": "staging",
            "api_host": overrides.get("api_host", "0.0.0.0"),
            "api_port": overrides.get("api_port", 8001),
            "workers": overrides.get("workers", 2),
            "log_level": overrides.get("log_level", "DEBUG"),
            "enable_monitoring": True,
            "enable_validation": True,
            "enable_fallback": True,
            "max_request_size": overrides.get("max_request_size", 100 * 1024 * 1024),  # 100MB
            "request_timeout": overrides.get("request_timeout", 600),  # 10 minutes
            "cors_origins": overrides.get("cors_origins", ["*"]),  # Allow all for testing
            "ssl_enabled": overrides.get("ssl_enabled", False)
        }
        
        self.config_manager.update_deployment_config(**staging_config)
        
        # Staging system configuration
        self.config_manager.update_system_config("monitoring", "window_size", 1000)
        self.config_manager.update_system_config("monitoring", "degradation_threshold", 0.05)
        self.config_manager.update_system_config("monitoring", "retention_days", 30)
        
        self.config_manager.update_system_config("validation", "startup_validation", True)
        self.config_manager.update_system_config("validation", "periodic_validation", True)
        self.config_manager.update_system_config("validation", "validation_interval_hours", 24)
        
        self.config_manager.update_system_config("fallback", "confidence_threshold", 0.7)
        self.config_manager.update_system_config("fallback", "memory_limit_mb", 2048)
        
        logger.info("Staging configuration applied")
    
    def _configure_development(self, **overrides):
        """Configure for development environment."""
        logger.info("Applying development configuration...")
        
        # Development deployment settings
        dev_config = {
            "environment": "development",
            "api_host": overrides.get("api_host", "127.0.0.1"),
            "api_port": overrides.get("api_port", 8000),
            "workers": overrides.get("workers", 1),
            "log_level": overrides.get("log_level", "DEBUG"),
            "enable_monitoring": True,
            "enable_validation": True,
            "enable_fallback": True,
            "max_request_size": overrides.get("max_request_size", 200 * 1024 * 1024),  # 200MB
            "request_timeout": overrides.get("request_timeout", 900),  # 15 minutes
            "cors_origins": overrides.get("cors_origins", ["*"]),
            "ssl_enabled": False
        }
        
        self.config_manager.update_deployment_config(**dev_config)
        
        # Development system configuration (more lenient)
        self.config_manager.update_system_config("monitoring", "window_size", 500)
        self.config_manager.update_system_config("monitoring", "degradation_threshold", 0.1)
        self.config_manager.update_system_config("monitoring", "retention_days", 7)
        
        self.config_manager.update_system_config("validation", "startup_validation", True)
        self.config_manager.update_system_config("validation", "periodic_validation", False)
        
        self.config_manager.update_system_config("fallback", "confidence_threshold", 0.6)
        self.config_manager.update_system_config("fallback", "memory_limit_mb", 1024)
        
        self.config_manager.update_system_config("optimization", "auto_threshold_optimization", False)
        
        logger.info("Development configuration applied")
    
    def _configure_testing(self, **overrides):
        """Configure for testing environment."""
        logger.info("Applying testing configuration...")
        
        # Testing deployment settings
        testing_config = {
            "environment": "testing",
            "api_host": overrides.get("api_host", "127.0.0.1"),
            "api_port": overrides.get("api_port", 8002),
            "workers": overrides.get("workers", 1),
            "log_level": overrides.get("log_level", "DEBUG"),
            "enable_monitoring": True,
            "enable_validation": True,
            "enable_fallback": True,
            "max_request_size": overrides.get("max_request_size", 50 * 1024 * 1024),
            "request_timeout": overrides.get("request_timeout", 300),
            "cors_origins": overrides.get("cors_origins", ["*"]),
            "ssl_enabled": False
        }
        
        self.config_manager.update_deployment_config(**testing_config)
        
        # Testing system configuration
        self.config_manager.update_system_config("monitoring", "window_size", 100)
        self.config_manager.update_system_config("monitoring", "degradation_threshold", 0.2)
        self.config_manager.update_system_config("monitoring", "retention_days", 1)
        
        self.config_manager.update_system_config("validation", "startup_validation", True)
        self.config_manager.update_system_config("validation", "periodic_validation", False)
        
        self.config_manager.update_system_config("fallback", "confidence_threshold", 0.5)
        self.config_manager.update_system_config("optimization", "auto_threshold_optimization", False)
        
        logger.info("Testing configuration applied")
    
    def setup_model_paths(self, models_base_path: Optional[str] = None) -> bool:
        """
        Setup and validate model paths.
        
        Args:
            models_base_path: Base path for models (optional)
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up model paths...")
        
        try:
            if models_base_path:
                # Update base path
                self.config_manager.model_paths.base_path = models_base_path
                self.config_manager._save_model_paths()
            
            # Create required directories
            base_path = Path(self.config_manager.model_paths.base_path)
            
            required_dirs = [
                base_path / "foundation",
                base_path / "classifiers", 
                base_path / "explanation",
                base_path / "preprocessing",
                base_path / "registry",
                base_path / "thresholds"
            ]
            
            for dir_path in required_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            
            # Validate paths
            missing_paths = self.config_manager.validate_model_paths()
            if missing_paths:
                logger.warning(f"Missing model paths: {missing_paths}")
                logger.info("You may need to download or copy model files to these locations")
            
            return True
            
        except Exception as e:
            logger.error(f"Model paths setup failed: {str(e)}")
            return False
    
    def setup_default_thresholds(self) -> bool:
        """
        Setup default threshold configurations.
        
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up default thresholds...")
        
        try:
            # Default thresholds for each language
            default_thresholds = {
                "English": {"threshold": 0.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8},
                "Tamil": {"threshold": 0.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8},
                "Hindi": {"threshold": 0.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8},
                "Malayalam": {"threshold": 0.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8},
                "Telugu": {"threshold": 0.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
            }
            
            for language, metrics in default_thresholds.items():
                self.config_manager.set_threshold(
                    language=language,
                    threshold=metrics["threshold"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1_score"],
                    model_version="1.0.0"
                )
            
            logger.info(f"Setup default thresholds for {len(default_thresholds)} languages")
            return True
            
        except Exception as e:
            logger.error(f"Default thresholds setup failed: {str(e)}")
            return False
    
    def create_environment_file(self, output_path: str = ".env") -> bool:
        """
        Create environment file with current configuration.
        
        Args:
            output_path: Path for environment file
            
        Returns:
            True if creation successful, False otherwise
        """
        logger.info(f"Creating environment file: {output_path}")
        
        try:
            deployment_config = self.config_manager.deployment
            if not deployment_config:
                logger.error("No deployment configuration found")
                return False
            
            env_content = f"""# AI Voice Detection System Environment Configuration
# Generated on {datetime.now().isoformat()}
# Environment: {deployment_config.environment.value}

# Environment
ENVIRONMENT={deployment_config.environment.value}
DEBUG={"true" if self.environment == DeploymentEnvironment.DEVELOPMENT else "false"}

# API Configuration
API_HOST={deployment_config.api_host}
API_PORT={deployment_config.api_port}
API_WORKERS={deployment_config.workers}

# Model Configuration
MODELS_BASE_PATH={self.config_manager.model_paths.base_path if self.config_manager.model_paths else "models"}
FOUNDATION_MODEL=auto

# Logging
LOG_LEVEL={deployment_config.log_level}
LOG_FILE=logs/app.log

# Security
MAX_REQUEST_SIZE={deployment_config.max_request_size}
REQUEST_TIMEOUT={deployment_config.request_timeout}
SSL_ENABLED={"true" if deployment_config.ssl_enabled else "false"}
"""
            
            if deployment_config.ssl_enabled:
                env_content += f"""SSL_CERT_FILE={deployment_config.ssl_cert_path or "/etc/ssl/certs/server.crt"}
SSL_KEY_FILE={deployment_config.ssl_key_path or "/etc/ssl/private/server.key"}
"""
            
            env_content += f"""
# CORS
CORS_ORIGINS={",".join(deployment_config.cors_origins)}

# Authentication (CHANGE THESE IN PRODUCTION!)
API_KEYS=your-secure-api-key-here

# Monitoring
MONITORING_ENABLED={"true" if deployment_config.enable_monitoring else "false"}
VALIDATION_ENABLED={"true" if deployment_config.enable_validation else "false"}
FALLBACK_ENABLED={"true" if deployment_config.enable_fallback else "false"}
"""
            
            with open(output_path, 'w') as f:
                f.write(env_content)
            
            logger.info(f"Environment file created: {output_path}")
            
            if deployment_config.environment == DeploymentEnvironment.PRODUCTION:
                logger.warning("IMPORTANT: Update API_KEYS and SSL certificate paths in the environment file!")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment file creation failed: {str(e)}")
            return False
    
    def create_docker_compose(self, output_path: str = "docker-compose.yml") -> bool:
        """
        Create Docker Compose file for deployment.
        
        Args:
            output_path: Path for Docker Compose file
            
        Returns:
            True if creation successful, False otherwise
        """
        logger.info(f"Creating Docker Compose file: {output_path}")
        
        try:
            deployment_config = self.config_manager.deployment
            if not deployment_config:
                logger.error("No deployment configuration found")
                return False
            
            compose_content = f"""version: '3.8'

services:
  ai-voice-detection:
    build: .
    ports:
      - "{deployment_config.api_port}:{deployment_config.api_port}"
    environment:
      - ENVIRONMENT={deployment_config.environment.value}
      - API_HOST={deployment_config.api_host}
      - API_PORT={deployment_config.api_port}
      - API_WORKERS={deployment_config.workers}
      - LOG_LEVEL={deployment_config.log_level}
      - MODELS_BASE_PATH=/app/models
      - SSL_ENABLED={"true" if deployment_config.ssl_enabled else "false"}
      - MONITORING_ENABLED={"true" if deployment_config.enable_monitoring else "false"}
      - VALIDATION_ENABLED={"true" if deployment_config.enable_validation else "false"}
      - FALLBACK_ENABLED={"true" if deployment_config.enable_fallback else "false"}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./.kiro:/app/.kiro
"""
            
            if deployment_config.ssl_enabled:
                compose_content += """      - ./ssl:/app/ssl
"""
            
            compose_content += f"""    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{deployment_config.api_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
"""
            
            # Add monitoring service for production
            if deployment_config.environment == DeploymentEnvironment.PRODUCTION:
                compose_content += """
  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
"""
            
            with open(output_path, 'w') as f:
                f.write(compose_content)
            
            logger.info(f"Docker Compose file created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Docker Compose file creation failed: {str(e)}")
            return False
    
    def create_systemd_service(self, output_path: str = "ai-voice-detection.service") -> bool:
        """
        Create systemd service file for deployment.
        
        Args:
            output_path: Path for service file
            
        Returns:
            True if creation successful, False otherwise
        """
        logger.info(f"Creating systemd service file: {output_path}")
        
        try:
            deployment_config = self.config_manager.deployment
            if not deployment_config:
                logger.error("No deployment configuration found")
                return False
            
            service_content = f"""[Unit]
Description=AI Voice Detection API Service
After=network.target
Wants=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/ai-voice-detection
Environment=ENVIRONMENT={deployment_config.environment.value}
Environment=API_HOST={deployment_config.api_host}
Environment=API_PORT={deployment_config.api_port}
Environment=API_WORKERS={deployment_config.workers}
Environment=LOG_LEVEL={deployment_config.log_level}
Environment=MODELS_BASE_PATH=/opt/ai-voice-detection/models
ExecStart=/opt/ai-voice-detection/venv/bin/python -m uvicorn src.main:app --host {deployment_config.api_host} --port {deployment_config.api_port} --workers {deployment_config.workers}
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ai-voice-detection

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/ai-voice-detection/logs /opt/ai-voice-detection/.kiro

[Install]
WantedBy=multi-user.target
"""
            
            with open(output_path, 'w') as f:
                f.write(service_content)
            
            logger.info(f"Systemd service file created: {output_path}")
            logger.info("To install: sudo cp ai-voice-detection.service /etc/systemd/system/")
            logger.info("To enable: sudo systemctl enable ai-voice-detection")
            logger.info("To start: sudo systemctl start ai-voice-detection")
            
            return True
            
        except Exception as e:
            logger.error(f"Systemd service file creation failed: {str(e)}")
            return False
    
    def export_configuration(self, output_dir: str = "deployment_config") -> bool:
        """
        Export complete configuration for deployment.
        
        Args:
            output_dir: Directory to export configuration
            
        Returns:
            True if export successful, False otherwise
        """
        logger.info(f"Exporting configuration to: {output_dir}")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export configuration manager data
            config_file = output_path / "configuration.json"
            self.config_manager.export_configuration(str(config_file))
            
            # Create environment file
            env_file = output_path / f".env.{self.environment.value}"
            self.create_environment_file(str(env_file))
            
            # Create Docker Compose file
            compose_file = output_path / "docker-compose.yml"
            self.create_docker_compose(str(compose_file))
            
            # Create systemd service file
            service_file = output_path / "ai-voice-detection.service"
            self.create_systemd_service(str(service_file))
            
            # Create deployment README
            readme_file = output_path / "README.md"
            self._create_deployment_readme(str(readme_file))
            
            logger.info(f"Configuration exported successfully to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration export failed: {str(e)}")
            return False
    
    def _create_deployment_readme(self, output_path: str):
        """Create deployment README file."""
        readme_content = f"""# AI Voice Detection System - Deployment Configuration

Generated on: {datetime.now().isoformat()}
Environment: {self.environment.value}

## Files Included

- `configuration.json`: Complete system configuration
- `.env.{self.environment.value}`: Environment variables file
- `docker-compose.yml`: Docker Compose configuration
- `ai-voice-detection.service`: Systemd service file

## Deployment Steps

### Docker Deployment

1. Copy the configuration files to your deployment server
2. Update API keys and SSL certificates in the environment file
3. Ensure model files are available in the `models/` directory
4. Run: `docker-compose up -d`

### Manual Deployment

1. Copy files to `/opt/ai-voice-detection/`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Copy systemd service file: `sudo cp ai-voice-detection.service /etc/systemd/system/`
4. Enable service: `sudo systemctl enable ai-voice-detection`
5. Start service: `sudo systemctl start ai-voice-detection`

## Configuration Notes

### Environment: {self.environment.value}

"""
        
        if self.environment == DeploymentEnvironment.PRODUCTION:
            readme_content += """
**PRODUCTION SECURITY CHECKLIST:**

- [ ] Update API keys in environment file
- [ ] Configure SSL certificates
- [ ] Set appropriate CORS origins
- [ ] Enable security headers
- [ ] Configure log rotation
- [ ] Set up monitoring and alerting
- [ ] Configure firewall rules
- [ ] Set up backup procedures

"""
        
        readme_content += f"""
### Health Checks

- Basic health: `curl http://localhost:{self.config_manager.deployment.api_port}/health`
- Detailed health: `curl http://localhost:{self.config_manager.deployment.api_port}/health/detailed`
- Model health: `curl http://localhost:{self.config_manager.deployment.api_port}/health/models`

### Monitoring

The system includes comprehensive monitoring capabilities:

- Real-time accuracy tracking
- Performance metrics collection
- Automatic degradation detection
- Model validation and registry

### Support

For issues or questions, check the system logs and health endpoints.
"""
        
        with open(output_path, 'w') as f:
            f.write(readme_content)


def main():
    """Main function for deployment configuration script."""
    parser = argparse.ArgumentParser(description="Configure AI Voice Detection System for deployment")
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
        "--ssl-cert",
        help="SSL certificate path"
    )
    parser.add_argument(
        "--ssl-key",
        help="SSL key path"
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
    
    # Prepare configuration overrides
    overrides = {}
    if args.api_host:
        overrides["api_host"] = args.api_host
    if args.api_port:
        overrides["api_port"] = args.api_port
    if args.workers:
        overrides["workers"] = args.workers
    if args.ssl_cert:
        overrides["ssl_cert_path"] = args.ssl_cert
    if args.ssl_key:
        overrides["ssl_key_path"] = args.ssl_key
    
    # Configure deployment
    configurator = DeploymentConfigurator(args.environment)
    
    success = True
    
    # Configure for environment
    if not configurator.configure_for_environment(**overrides):
        success = False
    
    # Setup model paths
    if not configurator.setup_model_paths(args.models_path):
        success = False
    
    # Setup defaults if requested
    if args.setup_defaults:
        if not configurator.setup_default_thresholds():
            success = False
    
    # Export configuration
    if not configurator.export_configuration(args.export_dir):
        success = False
    
    if success:
        print(f"\n✅ Deployment configuration completed successfully")
        print(f"Configuration exported to: {args.export_dir}")
        print(f"Environment: {args.environment}")
        sys.exit(0)
    else:
        print(f"\n❌ Deployment configuration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()