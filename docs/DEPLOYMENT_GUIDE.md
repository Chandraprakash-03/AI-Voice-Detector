# Deployment Guide

This guide covers various deployment options for the AI-Generated Voice Detection API, from development to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended for production)
- **Storage**: 1GB+ free space (additional space for ML models)
- **Network**: Internet access for downloading dependencies and models

### Required Software

- **Python 3.8+** with pip
- **Docker** (optional, for containerized deployment)
- **Docker Compose** (optional, for multi-container setup)
- **Git** (for cloning the repository)

## Environment Configuration

### Environment Variables

The application uses environment-specific configuration files:

- `.env.development` - Development settings
- `.env.production` - Production settings
- `.env.testing` - Testing settings

### Key Configuration Variables

| Variable           | Description              | Default           | Required |
| ------------------ | ------------------------ | ----------------- | -------- |
| `ENVIRONMENT`      | Environment type         | development       | No       |
| `API_KEYS`         | Comma-separated API keys | dev-api-key-12345 | Yes      |
| `API_HOST`         | Server host              | 0.0.0.0           | No       |
| `API_PORT`         | Server port              | 8000              | No       |
| `LOG_LEVEL`        | Logging level            | INFO              | No       |
| `MODELS_BASE_PATH` | ML models directory      | models            | No       |
| `SSL_ENABLED`      | Enable HTTPS             | false             | No       |
| `SSL_CERT_FILE`    | SSL certificate path     | None              | No\*     |
| `SSL_KEY_FILE`     | SSL private key path     | None              | No\*     |

\*Required when `SSL_ENABLED=true`

## Local Development

### Quick Start

1. **Clone and setup**

   ```bash
   git clone <repository-url>
   cd ai-voice-detection-api
   make setup-dev
   ```

2. **Install dependencies**

   ```bash
   make install-dev
   ```

3. **Run development server**
   ```bash
   make run-dev
   ```

The API will be available at `http://localhost:8000` with hot reload enabled.

### Manual Setup

1. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.development .env
   mkdir -p logs models
   ```

4. **Run the application**
   ```bash
   ENVIRONMENT=development uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```

## Docker Deployment

### Development with Docker

1. **Build and run development container**

   ```bash
   make docker-dev
   ```

   Or manually:

   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

2. **Access the application**
   - API: `http://localhost:8000`
   - Documentation: `http://localhost:8000/docs`

### Production with Docker

1. **Build production image**

   ```bash
   make docker-build
   ```

2. **Run with Docker Compose**

   ```bash
   make docker-prod
   ```

   Or manually:

   ```bash
   docker-compose up --build -d
   ```

### Custom Docker Configuration

**Environment-specific compose files:**

```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml up

# With custom environment
docker-compose --env-file .env.custom up
```

## Production Deployment

### Prerequisites for Production

1. **SSL Certificates**

   ```bash
   mkdir -p certs
   # Add your SSL certificate files:
   # certs/cert.pem
   # certs/key.pem
   ```

2. **ML Models**

   ```bash
   mkdir -p models
   # Add your trained models:
   # models/tamil_voice_detector.h5
   # models/english_voice_detector.h5
   # models/hindi_voice_detector.h5
   # models/malayalam_voice_detector.h5
   # models/telugu_voice_detector.h5
   ```

3. **Production Configuration**
   ```bash
   cp .env.production .env
   # Edit .env with your production settings
   ```

### Production Setup Steps

1. **Prepare environment**

   ```bash
   make setup-prod
   ```

2. **Configure production settings**

   ```bash
   # Edit .env file
   nano .env
   ```

   Update these critical settings:

   ```bash
   ENVIRONMENT=production
   DEBUG=false
   API_KEYS=your-secure-api-key-1,your-secure-api-key-2
   SSL_ENABLED=true
   SSL_CERT_FILE=/app/certs/cert.pem
   SSL_KEY_FILE=/app/certs/key.pem
   LOG_LEVEL=INFO
   ```

3. **Deploy**
   ```bash
   make deploy
   ```

### Manual Production Deployment

1. **Install production dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run with production server**

   ```bash
   gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

   Or with uvicorn:

   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS with Docker

1. **Build and push Docker image**

   ```bash
   # Build image
   docker build -t voice-detection-api .

   # Tag for ECR
   docker tag voice-detection-api:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/voice-detection-api:latest

   # Push to ECR
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/voice-detection-api:latest
   ```

2. **Create ECS task definition**
   ```json
   {
   	"family": "voice-detection-api",
   	"networkMode": "awsvpc",
   	"requiresCompatibilities": ["FARGATE"],
   	"cpu": "1024",
   	"memory": "2048",
   	"executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
   	"containerDefinitions": [
   		{
   			"name": "voice-detection-api",
   			"image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/voice-detection-api:latest",
   			"portMappings": [
   				{
   					"containerPort": 8000,
   					"protocol": "tcp"
   				}
   			],
   			"environment": [
   				{
   					"name": "ENVIRONMENT",
   					"value": "production"
   				}
   			],
   			"secrets": [
   				{
   					"name": "API_KEYS",
   					"valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:voice-api-keys"
   				}
   			]
   		}
   	]
   }
   ```

#### Using AWS Lambda (Serverless)

1. **Install serverless dependencies**

   ```bash
   pip install mangum
   ```

2. **Create Lambda handler**

   ```python
   # lambda_handler.py
   from mangum import Mangum
   from src.main import app

   handler = Mangum(app)
   ```

3. **Deploy with AWS SAM or Serverless Framework**

### Google Cloud Platform

#### Using Cloud Run

1. **Build and deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/voice-detection-api
   gcloud run deploy --image gcr.io/PROJECT_ID/voice-detection-api --platform managed
   ```

### Azure Deployment

#### Using Azure Container Instances

1. **Build and push to ACR**

   ```bash
   az acr build --registry myregistry --image voice-detection-api .
   ```

2. **Deploy to ACI**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name voice-detection-api \
     --image myregistry.azurecr.io/voice-detection-api:latest \
     --ports 8000
   ```

## Monitoring and Maintenance

### Health Checks

The API provides health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Logging

**Log locations:**

- Development: Console output
- Production: `logs/app-prod.log`
- Docker: Container logs (`docker logs <container-name>`)

**View logs:**

```bash
# Local logs
make logs

# Docker logs
docker-compose logs -f api

# Specific container
docker logs -f voice-detection-api
```

### Monitoring Setup

1. **Application metrics**
   - Response times
   - Error rates
   - Request volume
   - Model inference times

2. **System metrics**
   - CPU usage
   - Memory usage
   - Disk space
   - Network I/O

3. **Recommended tools**
   - **Prometheus + Grafana** for metrics
   - **ELK Stack** for log analysis
   - **Sentry** for error tracking

### Backup and Recovery

1. **Configuration backup**

   ```bash
   # Backup configuration
   tar -czf config-backup-$(date +%Y%m%d).tar.gz .env* certs/
   ```

2. **Model backup**

   ```bash
   # Backup ML models
   tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
   ```

3. **Database backup** (if using database)
   ```bash
   # PostgreSQL example
   pg_dump voice_detection_db > backup-$(date +%Y%m%d).sql
   ```

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill process
   kill -9 <PID>
   ```

2. **Permission denied errors**

   ```bash
   # Fix file permissions
   chmod +x startup.sh
   # Fix directory permissions
   chmod -R 755 logs models
   ```

3. **SSL certificate issues**

   ```bash
   # Verify certificate
   openssl x509 -in certs/cert.pem -text -noout
   # Check certificate and key match
   openssl x509 -noout -modulus -in certs/cert.pem | openssl md5
   openssl rsa -noout -modulus -in certs/key.pem | openssl md5
   ```

4. **Model loading failures**

   ```bash
   # Validate models exist
   make validate-models
   # Check model file permissions
   ls -la models/
   ```

5. **Memory issues**
   ```bash
   # Check memory usage
   free -h
   # Monitor application memory
   docker stats voice-detection-api
   ```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug logging
python src/main.py
```

### Performance Tuning

1. **Optimize worker processes**

   ```bash
   # Calculate optimal workers: (2 x CPU cores) + 1
   export API_WORKERS=5
   ```

2. **Tune memory settings**

   ```bash
   # Increase model cache size
   export MODEL_CACHE_SIZE=10
   ```

3. **Configure timeouts**
   ```bash
   export REQUEST_TIMEOUT=300
   export MAX_REQUEST_SIZE=52428800
   ```

### Getting Support

1. **Check logs first**

   ```bash
   make logs
   tail -f logs/app-prod.log
   ```

2. **Validate configuration**

   ```bash
   python -c "from src.config import get_settings; print(get_settings())"
   ```

3. **Run health checks**

   ```bash
   make health-check
   ```

4. **Test API functionality**
   ```bash
   # Test with sample request
   curl -X POST "http://localhost:8000/api/voice-detection" \
     -H "x-api-key: dev-api-key-12345" \
     -H "Content-Type: application/json" \
     -d '{"language": "English", "audioFormat": "mp3", "audioBase64": "..."}'
   ```

For additional support, check the project documentation or contact the development team.
