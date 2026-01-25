"""
Configuration Management Module

Centralized configuration management for the AI-Generated Voice Detection API.
Supports environment-specific configurations and deployment settings.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from enum import Enum
from pydantic_settings import SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class APIConfig(BaseSettings):
    """API-specific configuration settings."""
    
    # Basic API settings
    title: str = Field(default="AI-Generated Voice Detection API", env="API_TITLE")
    description: str = Field(
        default="A secure REST API that analyzes voice recordings to determine whether they are AI-generated or human-spoken",
        env="API_DESCRIPTION"
    )
    version: str = Field(default="1.0.0", env="API_VERSION")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["GET", "POST"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    @field_validator('cors_origins', 'cors_methods', 'cors_headers', mode='before')
    @classmethod
    def parse_list_from_string(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class AuthConfig(BaseSettings):
    """Authentication configuration settings."""
    
    # API Keys
    api_keys: List[str] = Field(default=["RW-b3ZMf29EcBQLObtVffHiqine2b89qzlq3Hgg2pBUoqyIElXhg0DKleUAeZsXNIdK"], env="API_KEYS")
    
    # Rate limiting (future enhancement)
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    @field_validator('api_keys', mode='before')
    @classmethod
    def parse_api_keys(cls, v):
        """Parse comma-separated API keys."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v


class MLConfig(BaseSettings):
    """Machine Learning model configuration."""
    
    # Model paths
    models_base_path: str = Field(default="models", env="MODELS_BASE_PATH")
    model_cache_size: int = Field(default=5, env="MODEL_CACHE_SIZE")
    
    # Foundation model configuration
    foundation_model: str = Field(default="auto", env="FOUNDATION_MODEL")  # auto, xls-r-300m, hubert-base, wav2vec2-base
    foundation_models_path: str = Field(default="models/foundation", env="FOUNDATION_MODELS_PATH")
    
    # Explanation model configuration
    explanation_model: str = Field(default="auto", env="EXPLANATION_MODEL")  # auto, distilgpt2, flan-t5-small, gpt2
    explanation_models_path: str = Field(default="models/explanation", env="EXPLANATION_MODELS_PATH")
    explanation_enabled: bool = Field(default=True, env="EXPLANATION_ENABLED")
    
    # Legacy HuBERT configuration (for backward compatibility)
    hubert_model_path: str = Field(default="models/foundation/hubert-base", env="HUBERT_MODEL_PATH")
    hubert_config_path: str = Field(default="models/foundation/hubert-base/config.json", env="HUBERT_CONFIG_PATH")
    hubert_cache_dir: str = Field(default="models/foundation/hubert-base/cache", env="HUBERT_CACHE_DIR")
    
    # Classifier paths (language-specific)
    classifiers_path: str = Field(default="models/classifiers", env="CLASSIFIERS_PATH")
    tamil_model_path: str = Field(default="models/classifiers/tamil_classifier.h5", env="TAMIL_MODEL_PATH")
    english_model_path: str = Field(default="models/classifiers/english_classifier.h5", env="ENGLISH_MODEL_PATH")
    hindi_model_path: str = Field(default="models/classifiers/hindi_classifier.h5", env="HINDI_MODEL_PATH")
    malayalam_model_path: str = Field(default="models/classifiers/malayalam_classifier.h5", env="MALAYALAM_MODEL_PATH")
    telugu_model_path: str = Field(default="models/classifiers/telugu_classifier.h5", env="TELUGU_MODEL_PATH")
    
    # Model inference settings
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    max_audio_duration: int = Field(default=300, env="MAX_AUDIO_DURATION")  # seconds
    min_audio_duration: float = Field(default=0.5, env="MIN_AUDIO_DURATION")  # seconds
    
    # Feature extraction settings
    sample_rate: int = Field(default=22050, env="SAMPLE_RATE")
    n_mfcc: int = Field(default=13, env="N_MFCC")
    hop_length: int = Field(default=512, env="HOP_LENGTH")
    n_fft: int = Field(default=2048, env="N_FFT")


class DatabaseConfig(BaseSettings):
    """Database configuration (for future enhancements)."""
    
    # Database connection
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis cache (for model caching)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")  # seconds


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_max_bytes: int = Field(default=10485760, env="LOG_MAX_BYTES")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class SecurityConfig(BaseSettings):
    """Security configuration settings."""
    
    # HTTPS settings
    ssl_enabled: bool = Field(default=False, env="SSL_ENABLED")
    ssl_cert_file: Optional[str] = Field(default=None, env="SSL_CERT_FILE")
    ssl_key_file: Optional[str] = Field(default=None, env="SSL_KEY_FILE")
    
    # Security headers
    security_headers_enabled: bool = Field(default=True, env="SECURITY_HEADERS_ENABLED")
    
    # Request limits
    max_request_size: int = Field(default=50 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 50MB
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # seconds


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Configuration sections
    api: APIConfig = APIConfig()
    auth: AuthConfig = AuthConfig()
    ml: MLConfig = MLConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()
    
    model_config = SettingsConfigDict(
        env_file=[
            ".env.development",
            ".env.production",
            ".env.testing",
            ".env",
        ],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # <-- THIS is what fixes your error
    )
    
    @field_validator('debug', mode='before')
    @classmethod
    def parse_debug(cls, v):
        """Parse debug flag from string."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v
    
    def get_model_path(self, language: str) -> str:
        """Get classifier path for a specific language."""
        language_lower = language.lower()
        model_paths = {
            "tamil": self.ml.tamil_model_path,
            "english": self.ml.english_model_path,
            "hindi": self.ml.hindi_model_path,
            "malayalam": self.ml.malayalam_model_path,
            "telugu": self.ml.telugu_model_path,
        }
        
        if language_lower not in model_paths:
            raise ValueError(f"Unsupported language: {language}")
        
        return model_paths[language_lower]
    
    def get_foundation_model_path(self, model_name: str) -> str:
        """Get foundation model path."""
        return f"{self.ml.foundation_models_path}/{model_name}"
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING


def get_settings() -> Settings:
    """
    Get application settings instance.
    
    This function creates and returns a Settings instance, loading configuration
    from environment variables and .env files.
    
    Returns:
        Settings: Configured application settings
    """
    return Settings()


def setup_logging(settings: Settings) -> None:
    """
    Configure application logging based on settings.
    
    Args:
        settings: Application settings containing logging configuration
    """
    import logging.handlers
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.logging.log_level.value))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(settings.logging.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if configured)
    if settings.logging.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logging.log_file,
            maxBytes=settings.logging.log_max_bytes,
            backupCount=settings.logging.log_backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


# Global settings instance
settings = get_settings()