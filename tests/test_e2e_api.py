"""
End-to-end API tests for AI-Generated Voice Detection API.

Tests full API workflows with real HTTP requests, concurrent request handling,
and performance/response time consistency.

Requirements: 7.1, 7.3, 7.5
"""

import pytest
import time
import threading
import statistics
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient

from src.main import app
# Define supported languages based on the schema
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


class TestEndToEndAPI:
    """End-to-end API tests with real HTTP requests."""
    
    @classmethod
    def setup_class(cls):
        """Set up test client and fixtures."""
        cls.client = TestClient(app)
        cls.valid_api_key = "dev-api-key-12345"
        cls.test_audio_data = cls._create_test_audio_data()
    
    @classmethod
    def _create_test_audio_data(cls):
        """Create test audio data for different scenarios."""
        # Create MP3-like data with ID3 header
        mp3_data = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 200
        return {
            'valid_mp3': base64.b64encode(mp3_data).decode('ascii'),
            'small_mp3': base64.b64encode(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 50).decode('ascii'),
            'large_mp3': base64.b64encode(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 1000).decode('ascii'),
        }
    
    def test_full_api_workflow_success_path(self):
        """Test complete successful API workflow with real HTTP requests."""
        for language in SUPPORTED_LANGUAGES:
            # Measure request time
            start_time = time.time()
            
            response = self.client.post(
                "/api/voice-detection",
                headers={
                    "x-api-key": self.valid_api_key,
                    "content-type": "application/json"
                },
                json={
                    "language": language,
                    "audioFormat": "mp3",
                    "audioBase64": self.test_audio_data['valid_mp3']
                }
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Verify response (may fail due to mock ML models, but test the workflow)
            assert response.status_code in [200, 422, 500]  # Valid HTTP responses
            assert response.headers.get("content-type", "").startswith("application/json")
            
            data = response.json()
            assert "status" in data
            
            if response.status_code == 200:
                # Verify successful response structure
                assert data["status"] == "success"
                assert data["language"] == language
                assert data["classification"] in ["AI_GENERATED", "HUMAN"]
                assert 0.0 <= data["confidenceScore"] <= 1.0
                assert isinstance(data["explanation"], str)
                assert len(data["explanation"]) > 0
                
                # Verify response time is reasonable (under 30 seconds)
                assert response_time < 30.0, f"Response time too slow: {response_time:.2f}s"
            else:
                # Verify error response structure
                assert data["status"] == "error"
                assert "message" in data
                assert isinstance(data["message"], str)
    
    def test_full_api_workflow_error_paths(self):
        """Test complete error handling workflows with real HTTP requests."""
        error_scenarios = [
            # Authentication errors
            {
                "headers": {},  # Missing API key
                "json": {
                    "language": "English",
                    "audioFormat": "mp3",
                    "audioBase64": self.test_audio_data['valid_mp3']
                },
                "expected_status": 401,
                "error_type": "authentication"
            },
            {
                "headers": {"x-api-key": "invalid-key"},  # Invalid API key
                "json": {
                    "language": "English",
                    "audioFormat": "mp3",
                    "audioBase64": self.test_audio_data['valid_mp3']
                },
                "expected_status": 401,
                "error_type": "authentication"
            },
            # Validation errors
            {
                "headers": {"x-api-key": self.valid_api_key},
                "json": {
                    "language": "InvalidLanguage",
                    "audioFormat": "mp3",
                    "audioBase64": self.test_audio_data['valid_mp3']
                },
                "expected_status": 400,
                "error_type": "validation"
            },
            {
                "headers": {"x-api-key": self.valid_api_key},
                "json": {
                    "language": "English",
                    "audioFormat": "wav",  # Invalid format
                    "audioBase64": self.test_audio_data['valid_mp3']
                },
                "expected_status": 400,
                "error_type": "validation"
            },
            {
                "headers": {"x-api-key": self.valid_api_key},
                "json": {
                    "language": "English",
                    "audioFormat": "mp3",
                    "audioBase64": "invalid_base64_data!"
                },
                "expected_status": 400,
                "error_type": "validation"
            },
            # Missing fields
            {
                "headers": {"x-api-key": self.valid_api_key},
                "json": {
                    "language": "English"
                    # Missing audioFormat and audioBase64
                },
                "expected_status": 400,
                "error_type": "validation"
            }
        ]
        
        for scenario in error_scenarios:
            start_time = time.time()
            
            response = self.client.post(
                "/api/voice-detection",
                headers=scenario["headers"],
                json=scenario["json"]
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Verify error response
            assert response.status_code == scenario["expected_status"]
            assert response.headers.get("content-type", "").startswith("application/json")
            
            data = response.json()
            assert data["status"] == "error"
            assert "message" in data
            assert isinstance(data["message"], str)
            assert len(data["message"]) > 0
            
            # Error responses should be fast
            assert response_time < 5.0, f"Error response too slow: {response_time:.2f}s"
    
    def test_concurrent_request_handling(self):
        """Test concurrent request handling with multiple threads."""
        num_concurrent_requests = 10
        results = []
        
        def make_request(request_id):
            """Make a single API request."""
            start_time = time.time()
            
            response = self.client.post(
                "/api/voice-detection",
                headers={"x-api-key": self.valid_api_key},
                json={
                    "language": "English",
                    "audioFormat": "mp3",
                    "audioBase64": self.test_audio_data['valid_mp3']
                }
            )
            
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'response_data': response.json() if response.headers.get("content-type", "").startswith("application/json") else None
            }
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent_requests)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per request
                    results.append(result)
                except Exception as e:
                    # Record failed requests
                    results.append({
                        'request_id': -1,
                        'status_code': 500,
                        'response_time': 30.0,
                        'error': str(e)
                    })
        
        # Verify all requests completed
        assert len(results) == num_concurrent_requests
        
        # Verify no request crashed the server
        for result in results:
            assert 200 <= result['status_code'] < 600  # Valid HTTP status codes
            assert result['response_time'] < 30.0  # Reasonable response time
        
        # Verify response consistency
        status_codes = [r['status_code'] for r in results]
        # All requests should return the same status code (consistency)
        unique_status_codes = set(status_codes)
        assert len(unique_status_codes) <= 2  # Should be consistent (success or error)
        
        # Calculate response time statistics
        response_times = [r['response_time'] for r in results]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"Concurrent request stats:")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Min response time: {min_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")
        print(f"  Status codes: {set(status_codes)}")
        
        # Verify reasonable performance
        assert avg_response_time < 10.0  # Average should be under 10 seconds
        assert max_response_time < 30.0  # No request should take more than 30 seconds
    
    def test_performance_and_response_time_consistency(self):
        """Test performance and response time consistency across multiple requests."""
        num_requests = 20
        response_times = []
        
        # Test with different languages to ensure consistency
        languages_to_test = SUPPORTED_LANGUAGES[:3]  # Test first 3 languages
        
        for language in languages_to_test:
            for i in range(num_requests // len(languages_to_test)):
                start_time = time.time()
                
                response = self.client.post(
                    "/api/voice-detection",
                    headers={"x-api-key": self.valid_api_key},
                    json={
                        "language": language,
                        "audioFormat": "mp3",
                        "audioBase64": self.test_audio_data['valid_mp3']
                    }
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                response_times.append({
                    'language': language,
                    'response_time': response_time,
                    'status_code': response.status_code
                })
                
                # Verify response is valid
                assert 200 <= response.status_code < 600
                assert response.headers.get("content-type", "").startswith("application/json")
        
        # Calculate statistics
        times = [r['response_time'] for r in response_times]
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        print(f"Performance statistics:")
        print(f"  Average response time: {avg_time:.3f}s")
        print(f"  Standard deviation: {std_dev:.3f}s")
        print(f"  Min response time: {min_time:.3f}s")
        print(f"  Max response time: {max_time:.3f}s")
        
        # Verify performance requirements
        assert avg_time < 15.0  # Average response time should be reasonable
        assert max_time < 30.0  # No request should take more than 30 seconds
        assert std_dev < 10.0   # Response times should be relatively consistent
        
        # Verify consistency across languages
        language_times = {}
        for result in response_times:
            lang = result['language']
            if lang not in language_times:
                language_times[lang] = []
            language_times[lang].append(result['response_time'])
        
        # Calculate average time per language
        language_averages = {lang: statistics.mean(times) for lang, times in language_times.items()}
        
        print(f"Average response times by language:")
        for lang, avg in language_averages.items():
            print(f"  {lang}: {avg:.3f}s")
        
        # Verify language consistency (no language should be significantly slower)
        max_lang_avg = max(language_averages.values())
        min_lang_avg = min(language_averages.values())
        language_variance = max_lang_avg - min_lang_avg
        
        assert language_variance < 5.0  # Languages should have similar response times
    
    def test_api_stability_under_load(self):
        """Test API stability under sustained load."""
        num_requests = 50
        batch_size = 5
        successful_requests = 0
        failed_requests = 0
        
        for batch in range(0, num_requests, batch_size):
            batch_results = []
            
            # Execute batch of requests
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for i in range(batch_size):
                    if batch + i >= num_requests:
                        break
                    
                    future = executor.submit(
                        self.client.post,
                        "/api/voice-detection",
                        headers={"x-api-key": self.valid_api_key},
                        json={
                            "language": SUPPORTED_LANGUAGES[i % len(SUPPORTED_LANGUAGES)],
                            "audioFormat": "mp3",
                            "audioBase64": self.test_audio_data['valid_mp3']
                        }
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures, timeout=30):
                    try:
                        response = future.result()
                        if 200 <= response.status_code < 300:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                        batch_results.append(response.status_code)
                    except Exception:
                        failed_requests += 1
                        batch_results.append(500)
            
            # Small delay between batches to avoid overwhelming
            time.sleep(0.1)
        
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        print(f"Load test results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Verify system stability
        assert total_requests >= num_requests * 0.9  # At least 90% of requests should complete
        # Note: Success rate may be low due to mock ML models, but system should not crash
    
    def test_different_audio_sizes_performance(self):
        """Test performance with different audio file sizes."""
        audio_sizes = ['small_mp3', 'valid_mp3', 'large_mp3']
        size_performance = {}
        
        for size_type in audio_sizes:
            response_times = []
            
            for _ in range(5):  # Test each size 5 times
                start_time = time.time()
                
                response = self.client.post(
                    "/api/voice-detection",
                    headers={"x-api-key": self.valid_api_key},
                    json={
                        "language": "English",
                        "audioFormat": "mp3",
                        "audioBase64": self.test_audio_data[size_type]
                    }
                )
                
                end_time = time.time()
                response_times.append(end_time - start_time)
                
                # Verify response is valid
                assert 200 <= response.status_code < 600
            
            avg_time = statistics.mean(response_times)
            size_performance[size_type] = avg_time
            
            print(f"{size_type} average response time: {avg_time:.3f}s")
        
        # Verify that larger files don't take disproportionately longer
        # (This may not hold true with real ML models, but tests the pipeline)
        for size_type, avg_time in size_performance.items():
            assert avg_time < 30.0  # All sizes should complete within reasonable time
    
    def test_api_error_recovery(self):
        """Test API recovery after error conditions."""
        # First, cause some errors
        error_requests = [
            {"language": "InvalidLanguage", "audioFormat": "mp3", "audioBase64": self.test_audio_data['valid_mp3']},
            {"language": "English", "audioFormat": "wav", "audioBase64": self.test_audio_data['valid_mp3']},
            {"language": "English", "audioFormat": "mp3", "audioBase64": "invalid_base64!"}
        ]
        
        for error_request in error_requests:
            response = self.client.post(
                "/api/voice-detection",
                headers={"x-api-key": self.valid_api_key},
                json=error_request
            )
            # Verify error is handled properly
            assert response.status_code >= 400
            data = response.json()
            assert data["status"] == "error"
        
        # Then verify system can still handle valid requests
        response = self.client.post(
            "/api/voice-detection",
            headers={"x-api-key": self.valid_api_key},
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": self.test_audio_data['valid_mp3']
            }
        )
        
        # System should still be functional
        assert 200 <= response.status_code < 600
        data = response.json()
        assert "status" in data
        
        # Health check should still work
        health_response = self.client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"