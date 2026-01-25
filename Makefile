# Makefile for AI-Generated Voice Detection API

.PHONY: help install install-dev test test-coverage lint format clean build run run-dev docker-build docker-run docker-dev deploy health-check

# Default target
help:
	@echo "AI-Generated Voice Detection API - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  run          - Run the API server"
	@echo "  run-dev      - Run the API server in development mode"
	@echo "  test         - Run all tests"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-dev   - Run development Docker environment"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy       - Deploy to production"
	@echo "  health-check - Check API health"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean up temporary files"
	@echo "  logs         - View application logs"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

# Development
run:
	python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

run-dev:
	ENVIRONMENT=development python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src tests
	black --check src tests

format:
	black src tests

# Docker
docker-build:
	docker build -t voice-detection-api .

docker-run:
	docker run -p 8000:8000 --env-file .env.production voice-detection-api

docker-dev:
	docker-compose -f docker-compose.dev.yml up --build

docker-prod:
	docker-compose up --build -d

# Deployment
deploy:
	@echo "Deploying to production..."
	docker-compose down
	docker-compose pull
	docker-compose up --build -d
	@echo "Deployment complete. Checking health..."
	sleep 10
	make health-check

# Health check
health-check:
	@echo "Checking API health..."
	@curl -f http://localhost:8000/health || echo "Health check failed"
	@curl -f http://localhost:8000/health/detailed || echo "Detailed health check failed"

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

logs:
	@if [ -f "logs/app-prod.log" ]; then tail -f logs/app-prod.log; else echo "No production logs found"; fi

# Environment setup
setup-dev:
	@echo "Setting up development environment..."
	cp .env.example .env
	mkdir -p logs models
	make install-dev
	@echo "Development environment ready!"

setup-prod:
	@echo "Setting up production environment..."
	cp .env.production .env
	mkdir -p logs models certs
	@echo "Production environment ready!"
	@echo "Don't forget to:"
	@echo "1. Update API keys in .env"
	@echo "2. Add SSL certificates to certs/ directory"
	@echo "3. Add ML models to models/ directory"

# Model management
download-models:
	@echo "Downloading ML models..."
	mkdir -p models
	@echo "Note: Add your model download logic here"

validate-models:
	@echo "Validating ML models..."
	python -c "
import os
from src.config import get_settings
settings = get_settings()
languages = ['tamil', 'english', 'hindi', 'malayalam', 'telugu']
missing = []
for lang in languages:
    path = settings.get_model_path(lang.title())
    if not os.path.exists(path):
        missing.append(f'{lang}: {path}')
if missing:
    print('Missing models:')
    for m in missing:
        print(f'  - {m}')
    exit(1)
else:
    print('All models found!')
"

# Run all checks (lint, format, test)
check: lint format test