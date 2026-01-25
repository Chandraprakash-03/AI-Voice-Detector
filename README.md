# AI-Generated Voice Detection API

A secure REST API that analyzes voice recordings to determine whether they are AI-generated or human-spoken. The system supports five languages (Tamil, English, Hindi, Malayalam, and Telugu) and provides classification results with confidence scores and explanations.

## Features

- **Multi-language Support**: Supports Tamil, English, Hindi, Malayalam, and Telugu
- **High Accuracy**: Uses advanced machine learning models for voice detection
- **Secure Authentication**: API key-based authentication system
- **Comprehensive Validation**: Input validation for audio format, language, and data integrity
- **Detailed Responses**: Provides confidence scores and explanations for classifications
- **Production Ready**: Docker support, health checks, and comprehensive logging
- **Property-Based Testing**: Extensive test coverage with property-based testing

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-voice-detection-api
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   make install-dev
   # or manually:
   pip install -r requirements.txt -r requirements-dev.txt
   ```

4. **Set up environment**

   ```bash
   make setup-dev
   # This copies .env.example to .env and creates necessary directories
   ```

5. **Run the API**
   ```bash
   make run-dev
   # or manually:
   ENVIRONMENT=development uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## API Usage

### Authentication

All API requests require an API key in the `x-api-key` header:

```bash
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "x-api-key: dev-api-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
  }'
```

### Request Format

```json
{
	"language": "English",
	"audioFormat": "mp3",
	"audioBase64": "<base64-encoded-mp3-data>"
}
```

**Parameters:**

- `language`: One of "Tamil", "English", "Hindi", "Malayalam", "Telugu"
- `audioFormat`: Must be "mp3"
- `audioBase64`: Base64-encoded MP3 audio data

### Response Format

**Success Response:**

```json
{
	"status": "success",
	"language": "English",
	"classification": "AI_GENERATED",
	"confidenceScore": 0.87,
	"explanation": "The audio exhibits characteristics typical of AI-generated speech, including consistent pitch patterns and synthetic vocal tract modeling."
}
```

**Error Response:**

```json
{
	"status": "error",
	"message": "Invalid audioFormat: 'wav'. Only 'mp3' format is supported"
}
```

## Configuration

The API uses environment-based configuration with support for development, production, and testing environments.

### Environment Variables

| Variable               | Description                                       | Default           | Required |
| ---------------------- | ------------------------------------------------- | ----------------- | -------- |
| `ENVIRONMENT`          | Environment type (development/production/testing) | development       | No       |
| `API_KEYS`             | Comma-separated list of valid API keys            | dev-api-key-12345 | Yes      |
| `API_HOST`             | Server host address                               | 0.0.0.0           | No       |
| `API_PORT`             | Server port                                       | 8000              | No       |
| `LOG_LEVEL`            | Logging level (DEBUG/INFO/WARNING/ERROR)          | INFO              | No       |
| `MODELS_BASE_PATH`     | Path to ML models directory                       | models            | No       |
| `CONFIDENCE_THRESHOLD` | Classification confidence threshold               | 0.5               | No       |

### Environment Files

- `.env.development` - Development configuration
- `.env.production` - Production configuration
- `.env.testing` - Testing configuration

Copy the appropriate file to `.env` or set `ENVIRONMENT` variable to auto-load.

## Deployment

### Docker Deployment

1. **Build and run with Docker Compose (Production)**

   ```bash
   make docker-prod
   # or manually:
   docker-compose up --build -d
   ```

2. **Development with Docker**
   ```bash
   make docker-dev
   # or manually:
   docker-compose -f docker-compose.dev.yml up --build
   ```

### Manual Deployment

1. **Set up production environment**

   ```bash
   make setup-prod
   ```

2. **Configure production settings**
   - Update API keys in `.env`
   - Add SSL certificates to `certs/` directory
   - Add ML models to `models/` directory

3. **Deploy**
   ```bash
   make deploy
   ```

### Health Checks

- Basic health check: `GET /health`
- Detailed health check: `GET /health/detailed`

## Development

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run specific test file
pytest tests/test_audio_unit.py -v
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Run all checks
make check
```

### Project Structure

```
ai-voice-detection-api/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── audio/             # Audio processing
│   ├── auth/              # Authentication
│   ├── detection/         # ML detection engine
│   ├── models/            # Data models
│   ├── config.py          # Configuration management
│   ├── exceptions.py      # Custom exceptions
│   ├── error_handlers.py  # Error handling
│   └── main.py           # FastAPI application
├── tests/                 # Test suite
├── models/               # ML model files
├── logs/                 # Application logs
├── certs/                # SSL certificates
├── docker-compose.yml    # Production Docker config
├── docker-compose.dev.yml # Development Docker config
├── Dockerfile            # Production Docker image
├── Dockerfile.dev        # Development Docker image
├── Makefile             # Build and deployment commands
└── README.md            # This file
```

## API Documentation

### Interactive Documentation

When running the API, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### Endpoints

| Method | Endpoint               | Description                            |
| ------ | ---------------------- | -------------------------------------- |
| GET    | `/`                    | Root endpoint with API information     |
| GET    | `/health`              | Basic health check                     |
| GET    | `/health/detailed`     | Detailed health check with system info |
| POST   | `/api/voice-detection` | Main voice detection endpoint          |

## Security Considerations

### API Key Management

- **Development**: Use the default key `dev-api-key-12345`
- **Production**: Generate strong, unique API keys
- **Rotation**: Regularly rotate API keys
- **Storage**: Store keys securely (environment variables, secrets management)

### Best Practices

1. **HTTPS Only**: Always use HTTPS in production
2. **Rate Limiting**: Implement rate limiting for production use
3. **Input Validation**: All inputs are validated before processing
4. **Error Handling**: Errors don't expose internal system details
5. **Logging**: Comprehensive logging for security monitoring

## Troubleshooting

### Common Issues

1. **"Invalid API key" error**
   - Check that `x-api-key` header is included
   - Verify the API key is in the `API_KEYS` environment variable

2. **"Invalid audioFormat" error**
   - Ensure `audioFormat` is set to "mp3"
   - Verify the audio file is actually in MP3 format

3. **"Invalid base64 encoding" error**
   - Check that the audio data is properly base64 encoded
   - Ensure no extra characters or line breaks in the base64 string

4. **Model loading errors**
   - Verify ML model files exist in the `models/` directory
   - Check file permissions and paths in configuration

### Getting Help

1. Check the logs: `make logs` or view files in `logs/` directory
2. Run health checks: `curl http://localhost:8000/health/detailed`
3. Validate configuration: `python -c "from src.config import get_settings; print(get_settings())"`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run code quality checks: `make check`
6. Submit a pull request

## License

[Add your license information here]

## Support

[Add support contact information here]
