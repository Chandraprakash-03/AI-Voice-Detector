#!/bin/bash

# Startup script for AI-Generated Voice Detection API

set -e

echo "Starting AI-Generated Voice Detection API..."

# Create necessary directories
mkdir -p logs models

# Set default environment if not specified
export ENVIRONMENT=${ENVIRONMENT:-production}

echo "Environment: $ENVIRONMENT"

# Load environment-specific configuration
if [ -f ".env.$ENVIRONMENT" ]; then
    echo "Loading environment configuration from .env.$ENVIRONMENT"
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
fi

# Validate required directories and files
echo "Validating configuration..."

# Check if models directory exists and has model files
if [ ! -d "models" ]; then
    echo "Warning: Models directory not found. Creating empty directory."
    mkdir -p models
fi

# Check for SSL certificates in production
if [ "$ENVIRONMENT" = "production" ] && [ "$SSL_ENABLED" = "true" ]; then
    if [ ! -f "$SSL_CERT_FILE" ] || [ ! -f "$SSL_KEY_FILE" ]; then
        echo "Error: SSL certificates not found. Please provide valid SSL_CERT_FILE and SSL_KEY_FILE."
        exit 1
    fi
fi

# Validate API keys
if [ -z "$API_KEYS" ]; then
    echo "Warning: No API keys configured. Using default development key."
    export API_KEYS="dev-api-key-12345"
fi

# Set up logging
if [ ! -z "$LOG_FILE" ]; then
    LOG_DIR=$(dirname "$LOG_FILE")
    mkdir -p "$LOG_DIR"
    echo "Logging to: $LOG_FILE"
fi

# Health check function
health_check() {
    echo "Performing startup health check..."
    python -c "
import sys
import os
sys.path.append('.')
try:
    from src.config import get_settings
    settings = get_settings()
    print(f'Configuration loaded successfully for environment: {settings.environment}')
    print(f'API will run on {settings.api.host}:{settings.api.port}')
    print(f'Debug mode: {settings.debug}')
    print(f'Log level: {settings.logging.log_level}')
except Exception as e:
    print(f'Configuration error: {e}')
    sys.exit(1)
"
}

# Run health check
health_check

# Determine the command to run based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Starting production server with $API_WORKERS workers..."
    exec uvicorn src.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --workers "$API_WORKERS" \
        --access-log \
        --log-level info
elif [ "$ENVIRONMENT" = "development" ]; then
    echo "Starting development server with hot reload..."
    exec uvicorn src.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        --log-level debug
else
    echo "Starting server in $ENVIRONMENT mode..."
    exec uvicorn src.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --log-level info
fi