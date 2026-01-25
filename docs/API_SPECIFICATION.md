# API Specification

## Overview

The AI-Generated Voice Detection API provides a RESTful interface for analyzing voice recordings to determine whether they are AI-generated or human-spoken. The API supports five languages and returns detailed classification results.

## Base URL

- Development: `http://localhost:8000`
- Production: `https://your-domain.com`

## Authentication

All API requests require authentication using an API key provided in the `x-api-key` header.

```http
x-api-key: your-api-key-here
```

## Content Type

All requests must use `application/json` content type:

```http
Content-Type: application/json
```

## Endpoints

### 1. Root Endpoint

**GET /**

Returns basic API information and available endpoints.

**Response:**

```json
{
	"message": "AI-Generated Voice Detection API",
	"version": "1.0.0",
	"environment": "development",
	"endpoints": {
		"health": "/health",
		"detailed_health": "/health/detailed",
		"voice_detection": "/api/voice-detection",
		"documentation": "/docs",
		"openapi": "/openapi.json"
	}
}
```

### 2. Health Check

**GET /health**

Basic health check endpoint to verify API availability.

**Response:**

```json
{
	"status": "healthy",
	"service": "ai-voice-detection-api",
	"version": "1.0.0",
	"environment": "development",
	"debug": false
}
```

### 3. Detailed Health Check

**GET /health/detailed**

Detailed health check with system information and configuration status.

**Response:**

```json
{
	"status": "healthy",
	"service": "ai-voice-detection-api",
	"version": "1.0.0",
	"environment": "production",
	"debug": false,
	"configuration": {
		"models_path": "/app/models",
		"models_available": true,
		"supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
		"confidence_threshold": 0.5,
		"max_audio_duration": 300,
		"min_audio_duration": 0.5
	},
	"system": {
		"cors_enabled": true,
		"ssl_enabled": true,
		"security_headers_enabled": true
	}
}
```

### 4. Voice Detection

**POST /api/voice-detection**

Main endpoint for voice detection analysis.

**Request Headers:**

- `x-api-key`: Required API key for authentication
- `Content-Type`: Must be `application/json`

**Request Body:**

```json
{
	"language": "English",
	"audioFormat": "mp3",
	"audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
}
```

**Request Parameters:**

| Field         | Type   | Required | Description                                                |
| ------------- | ------ | -------- | ---------------------------------------------------------- |
| `language`    | string | Yes      | One of: "Tamil", "English", "Hindi", "Malayalam", "Telugu" |
| `audioFormat` | string | Yes      | Must be "mp3"                                              |
| `audioBase64` | string | Yes      | Base64-encoded MP3 audio data                              |

**Success Response (200 OK):**

```json
{
	"status": "success",
	"language": "English",
	"classification": "AI_GENERATED",
	"confidenceScore": 0.87,
	"explanation": "The audio exhibits characteristics typical of AI-generated speech, including consistent pitch patterns and synthetic vocal tract modeling."
}
```

**Success Response Fields:**

| Field             | Type   | Description                                      |
| ----------------- | ------ | ------------------------------------------------ |
| `status`          | string | Always "success" for successful requests         |
| `language`        | string | The original language from the request           |
| `classification`  | string | Either "AI_GENERATED" or "HUMAN"                 |
| `confidenceScore` | number | Confidence score between 0.0 and 1.0             |
| `explanation`     | string | Human-readable explanation of the classification |

## Error Responses

All error responses follow a consistent format:

```json
{
	"status": "error",
	"message": "Descriptive error message"
}
```

### HTTP Status Codes

| Status Code | Description           | Example Scenarios                                  |
| ----------- | --------------------- | -------------------------------------------------- |
| 200         | OK                    | Successful voice detection                         |
| 400         | Bad Request           | Invalid JSON, missing fields, invalid field values |
| 401         | Unauthorized          | Missing or invalid API key                         |
| 422         | Unprocessable Entity  | Invalid audio format, corrupted audio data         |
| 500         | Internal Server Error | System errors, model loading failures              |

### Common Error Messages

**Authentication Errors (401):**

```json
{
	"status": "error",
	"message": "Invalid or missing API key. Please provide a valid x-api-key header."
}
```

**Validation Errors (400):**

```json
{
	"status": "error",
	"message": "Missing required field: language. Must be one of: Tamil, English, Hindi, Malayalam, Telugu"
}
```

```json
{
	"status": "error",
	"message": "Invalid audioFormat: 'wav'. Only 'mp3' format is supported"
}
```

```json
{
	"status": "error",
	"message": "Invalid language: 'French'. Supported languages: Tamil, English, Hindi, Malayalam, Telugu"
}
```

**Processing Errors (422):**

```json
{
	"status": "error",
	"message": "Invalid base64 encoding in audioBase64 field"
}
```

```json
{
	"status": "error",
	"message": "Audio file is corrupted or not a valid MP3 format"
}
```

## Rate Limiting

In production environments, the API implements rate limiting:

- **Default Limit**: 100 requests per hour per API key
- **Burst Limit**: 20 requests per minute
- **Headers**: Rate limit information is included in response headers

Rate limit headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Request/Response Examples

### cURL Examples

**Basic voice detection request:**

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

**Health check:**

```bash
curl -X GET "http://localhost:8000/health"
```

**Detailed health check:**

```bash
curl -X GET "http://localhost:8000/health/detailed"
```

### Python Examples

```python
import requests
import base64

# Read and encode audio file
with open("audio_file.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Make API request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "x-api-key": "dev-api-key-12345",
        "Content-Type": "application/json"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_data
    }
)

# Handle response
if response.status_code == 200:
    result = response.json()
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidenceScore']:.2f}")
    print(f"Explanation: {result['explanation']}")
else:
    error = response.json()
    print(f"Error: {error['message']}")
```

### JavaScript Examples

```javascript
// Read file and convert to base64
const fileInput = document.getElementById("audioFile");
const file = fileInput.files[0];

const reader = new FileReader();
reader.onload = async function (e) {
	const base64Data = e.target.result.split(",")[1]; // Remove data:audio/mp3;base64, prefix

	try {
		const response = await fetch("http://localhost:8000/api/voice-detection", {
			method: "POST",
			headers: {
				"x-api-key": "dev-api-key-12345",
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				language: "English",
				audioFormat: "mp3",
				audioBase64: base64Data,
			}),
		});

		const result = await response.json();

		if (response.ok) {
			console.log("Classification:", result.classification);
			console.log("Confidence:", result.confidenceScore);
			console.log("Explanation:", result.explanation);
		} else {
			console.error("Error:", result.message);
		}
	} catch (error) {
		console.error("Request failed:", error);
	}
};

reader.readAsDataURL(file);
```

## OpenAPI/Swagger Documentation

The API provides interactive documentation through Swagger UI and ReDoc:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

These interfaces allow you to:

- Explore all available endpoints
- View request/response schemas
- Test API calls directly from the browser
- Download the OpenAPI specification

## Supported Audio Formats

Currently, the API only supports MP3 format:

- **Format**: MP3
- **Encoding**: Any standard MP3 encoding
- **Sample Rate**: Automatically handled (recommended: 16kHz or 22kHz)
- **Duration**: 0.5 seconds to 300 seconds (5 minutes)
- **File Size**: Maximum 50MB

## Best Practices

1. **Error Handling**: Always check the response status and handle errors appropriately
2. **Base64 Encoding**: Ensure proper base64 encoding without line breaks or extra characters
3. **Audio Quality**: Use clear, high-quality audio for better detection accuracy
4. **Language Matching**: Ensure the specified language matches the actual audio language
5. **API Key Security**: Keep API keys secure and rotate them regularly
6. **Rate Limiting**: Implement client-side rate limiting to avoid hitting API limits
7. **Timeout Handling**: Set appropriate timeouts for requests (recommended: 30-60 seconds)

## Changelog

### Version 1.0.0

- Initial API release
- Support for 5 languages (Tamil, English, Hindi, Malayalam, Telugu)
- MP3 audio format support
- API key authentication
- Comprehensive error handling
- Health check endpoints
