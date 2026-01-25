# API Usage Examples

This document provides comprehensive examples of how to use the AI-Generated Voice Detection API in various programming languages and scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [cURL Examples](#curl-examples)
3. [Python Examples](#python-examples)
4. [JavaScript Examples](#javascript-examples)
5. [Java Examples](#java-examples)
6. [C# Examples](#c-examples)
7. [Error Handling](#error-handling)
8. [Advanced Usage](#advanced-usage)

## Basic Usage

### Authentication

All API requests require an API key in the `x-api-key` header:

```http
POST /api/voice-detection HTTP/1.1
Host: localhost:8000
Content-Type: application/json
x-api-key: dev-api-key-12345

{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
}
```

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

## cURL Examples

### Basic Voice Detection

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

### Voice Detection with File

```bash
# Encode audio file to base64 and make request
AUDIO_BASE64=$(base64 -w 0 audio_file.mp3)

curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "x-api-key: dev-api-key-12345" \
  -H "Content-Type: application/json" \
  -d "{
    \"language\": \"English\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"$AUDIO_BASE64\"
  }"
```

### Health Check

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"

# Detailed health check
curl -X GET "http://localhost:8000/health/detailed"
```

### Different Languages

```bash
# Tamil audio detection
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "x-api-key: dev-api-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
  }'

# Hindi audio detection
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "x-api-key: dev-api-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "Hindi",
    "audioFormat": "mp3",
    "audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
  }'
```

## Python Examples

### Basic Usage with requests

```python
import requests
import base64
import json

def detect_voice(audio_file_path, language, api_key):
    """
    Detect if voice in audio file is AI-generated or human.

    Args:
        audio_file_path (str): Path to MP3 audio file
        language (str): Language of the audio
        api_key (str): API key for authentication

    Returns:
        dict: API response containing classification result
    """
    # Read and encode audio file
    with open(audio_file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')

    # Prepare request
    url = "http://localhost:8000/api/voice-detection"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_data
    }

    # Make request
    response = requests.post(url, headers=headers, json=payload)

    # Handle response
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

# Example usage
try:
    result = detect_voice("sample_audio.mp3", "English", "dev-api-key-12345")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidenceScore']:.2f}")
    print(f"Explanation: {result['explanation']}")
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
```

### Async Python with aiohttp

```python
import aiohttp
import asyncio
import base64

async def detect_voice_async(audio_file_path, language, api_key):
    """
    Async version of voice detection.
    """
    # Read and encode audio file
    with open(audio_file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')

    # Prepare request
    url = "http://localhost:8000/api/voice-detection"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_data
    }

    # Make async request
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                response.raise_for_status()

# Example usage
async def main():
    try:
        result = await detect_voice_async("sample_audio.mp3", "English", "dev-api-key-12345")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidenceScore']:.2f}")
    except Exception as e:
        print(f"Error: {e}")

# Run async example
asyncio.run(main())
```

### Batch Processing

```python
import requests
import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_audio_file(file_info):
    """Process a single audio file."""
    file_path, language, api_key = file_info

    try:
        with open(file_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        response = requests.post(
            "http://localhost:8000/api/voice-detection",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": audio_data
            }
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "file": file_path,
                "classification": result["classification"],
                "confidence": result["confidenceScore"],
                "success": True
            }
        else:
            return {
                "file": file_path,
                "error": response.json().get("message", "Unknown error"),
                "success": False
            }
    except Exception as e:
        return {
            "file": file_path,
            "error": str(e),
            "success": False
        }

def batch_process_audio_files(audio_files, language, api_key, max_workers=5):
    """
    Process multiple audio files concurrently.

    Args:
        audio_files (list): List of audio file paths
        language (str): Language of the audio files
        api_key (str): API key for authentication
        max_workers (int): Maximum number of concurrent requests

    Returns:
        list: Results for each processed file
    """
    file_infos = [(file_path, language, api_key) for file_path in audio_files]
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_audio_file, file_info): file_info[0]
            for file_info in file_infos
        }

        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)

            if result["success"]:
                print(f"✓ {result['file']}: {result['classification']} ({result['confidence']:.2f})")
            else:
                print(f"✗ {result['file']}: {result['error']}")

    return results

# Example usage
audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
results = batch_process_audio_files(audio_files, "English", "dev-api-key-12345")
```

## JavaScript Examples

### Browser JavaScript with Fetch

```javascript
class VoiceDetectionAPI {
	constructor(baseUrl, apiKey) {
		this.baseUrl = baseUrl;
		this.apiKey = apiKey;
	}

	async detectVoice(audioFile, language) {
		try {
			// Convert file to base64
			const base64Audio = await this.fileToBase64(audioFile);

			// Make API request
			const response = await fetch(`${this.baseUrl}/api/voice-detection`, {
				method: "POST",
				headers: {
					"x-api-key": this.apiKey,
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					language: language,
					audioFormat: "mp3",
					audioBase64: base64Audio,
				}),
			});

			const result = await response.json();

			if (response.ok) {
				return {
					success: true,
					data: result,
				};
			} else {
				return {
					success: false,
					error: result.message,
				};
			}
		} catch (error) {
			return {
				success: false,
				error: error.message,
			};
		}
	}

	fileToBase64(file) {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = () => {
				// Remove data:audio/mp3;base64, prefix
				const base64 = reader.result.split(",")[1];
				resolve(base64);
			};
			reader.onerror = reject;
			reader.readAsDataURL(file);
		});
	}

	async checkHealth() {
		try {
			const response = await fetch(`${this.baseUrl}/health`);
			return await response.json();
		} catch (error) {
			throw new Error(`Health check failed: ${error.message}`);
		}
	}
}

// Example usage
const api = new VoiceDetectionAPI("http://localhost:8000", "dev-api-key-12345");

// Handle file input
document
	.getElementById("audioFile")
	.addEventListener("change", async (event) => {
		const file = event.target.files[0];
		const language = document.getElementById("language").value;

		if (file && language) {
			try {
				const result = await api.detectVoice(file, language);

				if (result.success) {
					console.log("Classification:", result.data.classification);
					console.log("Confidence:", result.data.confidenceScore);
					console.log("Explanation:", result.data.explanation);

					// Update UI
					document.getElementById("result").innerHTML = `
                    <h3>Result: ${result.data.classification}</h3>
                    <p>Confidence: ${(result.data.confidenceScore * 100).toFixed(1)}%</p>
                    <p>Explanation: ${result.data.explanation}</p>
                `;
				} else {
					console.error("Error:", result.error);
					document.getElementById("result").innerHTML = `
                    <p style="color: red;">Error: ${result.error}</p>
                `;
				}
			} catch (error) {
				console.error("Request failed:", error);
			}
		}
	});
```

### Node.js with axios

```javascript
const axios = require("axios");
const fs = require("fs");

class VoiceDetectionClient {
	constructor(baseUrl, apiKey) {
		this.baseUrl = baseUrl;
		this.apiKey = apiKey;
		this.client = axios.create({
			baseURL: baseUrl,
			headers: {
				"x-api-key": apiKey,
				"Content-Type": "application/json",
			},
			timeout: 60000, // 60 second timeout
		});
	}

	async detectVoice(audioFilePath, language) {
		try {
			// Read and encode audio file
			const audioBuffer = fs.readFileSync(audioFilePath);
			const audioBase64 = audioBuffer.toString("base64");

			// Make API request
			const response = await this.client.post("/api/voice-detection", {
				language: language,
				audioFormat: "mp3",
				audioBase64: audioBase64,
			});

			return response.data;
		} catch (error) {
			if (error.response) {
				// API returned an error response
				throw new Error(`API Error: ${error.response.data.message}`);
			} else if (error.request) {
				// Request was made but no response received
				throw new Error("No response from API server");
			} else {
				// Something else happened
				throw new Error(`Request failed: ${error.message}`);
			}
		}
	}

	async checkHealth() {
		try {
			const response = await this.client.get("/health");
			return response.data;
		} catch (error) {
			throw new Error(`Health check failed: ${error.message}`);
		}
	}

	async getDetailedHealth() {
		try {
			const response = await this.client.get("/health/detailed");
			return response.data;
		} catch (error) {
			throw new Error(`Detailed health check failed: ${error.message}`);
		}
	}
}

// Example usage
async function main() {
	const client = new VoiceDetectionClient(
		"http://localhost:8000",
		"dev-api-key-12345",
	);

	try {
		// Check API health
		const health = await client.checkHealth();
		console.log("API Health:", health);

		// Detect voice in audio file
		const result = await client.detectVoice("sample_audio.mp3", "English");
		console.log("Classification:", result.classification);
		console.log("Confidence:", result.confidenceScore);
		console.log("Explanation:", result.explanation);
	} catch (error) {
		console.error("Error:", error.message);
	}
}

main();
```

## Java Examples

### Basic Java with HttpClient

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

public class VoiceDetectionClient {
    private final String baseUrl;
    private final String apiKey;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public VoiceDetectionClient(String baseUrl, String apiKey) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.httpClient = HttpClient.newHttpClient();
        this.objectMapper = new ObjectMapper();
    }

    public VoiceDetectionResult detectVoice(String audioFilePath, String language) throws Exception {
        // Read and encode audio file
        byte[] audioBytes = Files.readAllBytes(Paths.get(audioFilePath));
        String audioBase64 = Base64.getEncoder().encodeToString(audioBytes);

        // Create request payload
        String requestBody = objectMapper.writeValueAsString(Map.of(
            "language", language,
            "audioFormat", "mp3",
            "audioBase64", audioBase64
        ));

        // Build HTTP request
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/api/voice-detection"))
            .header("x-api-key", apiKey)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        // Send request
        HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());

        // Parse response
        JsonNode responseJson = objectMapper.readTree(response.body());

        if (response.statusCode() == 200) {
            return new VoiceDetectionResult(
                responseJson.get("classification").asText(),
                responseJson.get("confidenceScore").asDouble(),
                responseJson.get("explanation").asText(),
                responseJson.get("language").asText()
            );
        } else {
            throw new RuntimeException("API Error: " + responseJson.get("message").asText());
        }
    }

    public HealthStatus checkHealth() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/health"))
            .GET()
            .build();

        HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());

        JsonNode responseJson = objectMapper.readTree(response.body());

        return new HealthStatus(
            responseJson.get("status").asText(),
            responseJson.get("service").asText(),
            responseJson.get("version").asText()
        );
    }

    // Result classes
    public static class VoiceDetectionResult {
        public final String classification;
        public final double confidenceScore;
        public final String explanation;
        public final String language;

        public VoiceDetectionResult(String classification, double confidenceScore,
                                  String explanation, String language) {
            this.classification = classification;
            this.confidenceScore = confidenceScore;
            this.explanation = explanation;
            this.language = language;
        }

        @Override
        public String toString() {
            return String.format("VoiceDetectionResult{classification='%s', confidence=%.2f, language='%s'}",
                classification, confidenceScore, language);
        }
    }

    public static class HealthStatus {
        public final String status;
        public final String service;
        public final String version;

        public HealthStatus(String status, String service, String version) {
            this.status = status;
            this.service = service;
            this.version = version;
        }
    }

    // Example usage
    public static void main(String[] args) {
        VoiceDetectionClient client = new VoiceDetectionClient(
            "http://localhost:8000",
            "dev-api-key-12345"
        );

        try {
            // Check health
            HealthStatus health = client.checkHealth();
            System.out.println("API Health: " + health.status);

            // Detect voice
            VoiceDetectionResult result = client.detectVoice("sample_audio.mp3", "English");
            System.out.println("Result: " + result);
            System.out.println("Explanation: " + result.explanation);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
```

## C# Examples

### C# with HttpClient

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

public class VoiceDetectionClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly string _apiKey;

    public VoiceDetectionClient(string baseUrl, string apiKey)
    {
        _baseUrl = baseUrl;
        _apiKey = apiKey;
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("x-api-key", apiKey);
    }

    public async Task<VoiceDetectionResult> DetectVoiceAsync(string audioFilePath, string language)
    {
        try
        {
            // Read and encode audio file
            byte[] audioBytes = await File.ReadAllBytesAsync(audioFilePath);
            string audioBase64 = Convert.ToBase64String(audioBytes);

            // Create request payload
            var requestPayload = new
            {
                language = language,
                audioFormat = "mp3",
                audioBase64 = audioBase64
            };

            string jsonPayload = JsonSerializer.Serialize(requestPayload);
            var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            // Send request
            HttpResponseMessage response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/voice-detection", content);

            string responseContent = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                using JsonDocument doc = JsonDocument.Parse(responseContent);
                JsonElement root = doc.RootElement;

                return new VoiceDetectionResult
                {
                    Classification = root.GetProperty("classification").GetString(),
                    ConfidenceScore = root.GetProperty("confidenceScore").GetDouble(),
                    Explanation = root.GetProperty("explanation").GetString(),
                    Language = root.GetProperty("language").GetString()
                };
            }
            else
            {
                using JsonDocument doc = JsonDocument.Parse(responseContent);
                string errorMessage = doc.RootElement.GetProperty("message").GetString();
                throw new Exception($"API Error: {errorMessage}");
            }
        }
        catch (Exception ex)
        {
            throw new Exception($"Voice detection failed: {ex.Message}", ex);
        }
    }

    public async Task<HealthStatus> CheckHealthAsync()
    {
        try
        {
            HttpResponseMessage response = await _httpClient.GetAsync($"{_baseUrl}/health");
            string responseContent = await response.Content.ReadAsStringAsync();

            using JsonDocument doc = JsonDocument.Parse(responseContent);
            JsonElement root = doc.RootElement;

            return new HealthStatus
            {
                Status = root.GetProperty("status").GetString(),
                Service = root.GetProperty("service").GetString(),
                Version = root.GetProperty("version").GetString()
            };
        }
        catch (Exception ex)
        {
            throw new Exception($"Health check failed: {ex.Message}", ex);
        }
    }

    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}

public class VoiceDetectionResult
{
    public string Classification { get; set; }
    public double ConfidenceScore { get; set; }
    public string Explanation { get; set; }
    public string Language { get; set; }

    public override string ToString()
    {
        return $"Classification: {Classification}, Confidence: {ConfidenceScore:F2}, Language: {Language}";
    }
}

public class HealthStatus
{
    public string Status { get; set; }
    public string Service { get; set; }
    public string Version { get; set; }
}

// Example usage
class Program
{
    static async Task Main(string[] args)
    {
        var client = new VoiceDetectionClient("http://localhost:8000", "dev-api-key-12345");

        try
        {
            // Check health
            HealthStatus health = await client.CheckHealthAsync();
            Console.WriteLine($"API Health: {health.Status}");

            // Detect voice
            VoiceDetectionResult result = await client.DetectVoiceAsync("sample_audio.mp3", "English");
            Console.WriteLine($"Result: {result}");
            Console.WriteLine($"Explanation: {result.Explanation}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        finally
        {
            client.Dispose();
        }
    }
}
```

## Error Handling

### Common Error Scenarios

```python
import requests
import base64

def robust_voice_detection(audio_file_path, language, api_key, max_retries=3):
    """
    Robust voice detection with comprehensive error handling.
    """
    for attempt in range(max_retries):
        try:
            # Read and encode audio file
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')

            # Make API request
            response = requests.post(
                "http://localhost:8000/api/voice-detection",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "language": language,
                    "audioFormat": "mp3",
                    "audioBase64": audio_data
                },
                timeout=60  # 60 second timeout
            )

            # Handle different response codes
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            elif response.status_code == 400:
                # Bad request - client error
                error_data = response.json()
                return {
                    "success": False,
                    "error": "Invalid request",
                    "message": error_data.get("message", "Bad request"),
                    "retry": False
                }
            elif response.status_code == 401:
                # Unauthorized - API key issue
                return {
                    "success": False,
                    "error": "Authentication failed",
                    "message": "Invalid or missing API key",
                    "retry": False
                }
            elif response.status_code == 422:
                # Unprocessable entity - invalid audio data
                error_data = response.json()
                return {
                    "success": False,
                    "error": "Invalid audio data",
                    "message": error_data.get("message", "Audio processing failed"),
                    "retry": False
                }
            elif response.status_code == 500:
                # Server error - might be temporary
                if attempt < max_retries - 1:
                    print(f"Server error, retrying... (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    return {
                        "success": False,
                        "error": "Server error",
                        "message": "Internal server error",
                        "retry": True
                    }
            else:
                # Other HTTP errors
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "message": response.text,
                    "retry": attempt < max_retries - 1
                }

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            else:
                return {
                    "success": False,
                    "error": "Timeout",
                    "message": "Request timed out",
                    "retry": True
                }
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"Connection error, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            else:
                return {
                    "success": False,
                    "error": "Connection error",
                    "message": "Could not connect to API server",
                    "retry": True
                }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "File not found",
                "message": f"Audio file not found: {audio_file_path}",
                "retry": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": "Unexpected error",
                "message": str(e),
                "retry": False
            }

    return {
        "success": False,
        "error": "Max retries exceeded",
        "message": "Failed after multiple attempts",
        "retry": False
    }

# Example usage with error handling
result = robust_voice_detection("sample_audio.mp3", "English", "dev-api-key-12345")

if result["success"]:
    data = result["data"]
    print(f"Classification: {data['classification']}")
    print(f"Confidence: {data['confidenceScore']:.2f}")
    print(f"Explanation: {data['explanation']}")
else:
    print(f"Error: {result['error']}")
    print(f"Message: {result['message']}")
    if result.get("retry", False):
        print("This error might be temporary, consider retrying later.")
```

## Advanced Usage

### Streaming Multiple Files

```python
import asyncio
import aiohttp
import base64
from pathlib import Path

async def process_audio_stream(audio_files, language, api_key, concurrent_limit=5):
    """
    Process multiple audio files with controlled concurrency.
    """
    semaphore = asyncio.Semaphore(concurrent_limit)

    async def process_single_file(session, file_path):
        async with semaphore:
            try:
                # Read and encode file
                with open(file_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode('utf-8')

                # Make request
                async with session.post(
                    "http://localhost:8000/api/voice-detection",
                    json={
                        "language": language,
                        "audioFormat": "mp3",
                        "audioBase64": audio_data
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "file": str(file_path),
                            "success": True,
                            "classification": result["classification"],
                            "confidence": result["confidenceScore"]
                        }
                    else:
                        error_data = await response.json()
                        return {
                            "file": str(file_path),
                            "success": False,
                            "error": error_data.get("message", "Unknown error")
                        }
            except Exception as e:
                return {
                    "file": str(file_path),
                    "success": False,
                    "error": str(e)
                }

    # Create session with API key
    headers = {"x-api-key": api_key}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [process_single_file(session, file_path) for file_path in audio_files]
        results = await asyncio.gather(*tasks)

    return results

# Example usage
async def main():
    audio_files = list(Path("audio_samples").glob("*.mp3"))
    results = await process_audio_stream(audio_files, "English", "dev-api-key-12345")

    for result in results:
        if result["success"]:
            print(f"✓ {result['file']}: {result['classification']} ({result['confidence']:.2f})")
        else:
            print(f"✗ {result['file']}: {result['error']}")

asyncio.run(main())
```

### Custom Client with Retry Logic

```python
import time
import random
from typing import Optional, Dict, Any

class VoiceDetectionClientWithRetry:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Content-Type": "application/json"
        })

    def detect_voice_with_retry(
        self,
        audio_file_path: str,
        language: str,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        jitter: bool = True
    ) -> Dict[str, Any]:
        """
        Detect voice with exponential backoff retry logic.
        """
        for attempt in range(max_retries + 1):
            try:
                # Read and encode audio
                with open(audio_file_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode('utf-8')

                # Make request
                response = self.session.post(
                    f"{self.base_url}/api/voice-detection",
                    json={
                        "language": language,
                        "audioFormat": "mp3",
                        "audioBase64": audio_data
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    return {
                        "success": True,
                        "data": response.json(),
                        "attempts": attempt + 1
                    }
                elif response.status_code in [400, 401, 422]:
                    # Client errors - don't retry
                    return {
                        "success": False,
                        "error": response.json().get("message", "Client error"),
                        "status_code": response.status_code,
                        "attempts": attempt + 1
                    }
                elif response.status_code >= 500 and attempt < max_retries:
                    # Server errors - retry with backoff
                    wait_time = backoff_factor * (2 ** attempt)
                    if jitter:
                        wait_time += random.uniform(0, 1)

                    print(f"Server error {response.status_code}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "attempts": attempt + 1
                    }

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    if jitter:
                        wait_time += random.uniform(0, 1)
                    print(f"Timeout, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": "Request timeout",
                        "attempts": attempt + 1
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "attempts": attempt + 1
                }

        return {
            "success": False,
            "error": "Max retries exceeded",
            "attempts": max_retries + 1
        }

# Example usage
client = VoiceDetectionClientWithRetry("http://localhost:8000", "dev-api-key-12345")
result = client.detect_voice_with_retry("sample_audio.mp3", "English")

if result["success"]:
    data = result["data"]
    print(f"Success after {result['attempts']} attempts")
    print(f"Classification: {data['classification']}")
else:
    print(f"Failed after {result['attempts']} attempts: {result['error']}")
```

These examples demonstrate comprehensive usage of the AI-Generated Voice Detection API across different programming languages and scenarios, including error handling, batch processing, and advanced features.
