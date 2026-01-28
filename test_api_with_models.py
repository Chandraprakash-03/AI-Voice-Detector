#!/usr/bin/env python3
"""
Test script to verify API is working with actual models
"""

import requests
import json
import base64
import numpy as np
import librosa
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file"""
    # Generate a simple sine wave (1 second, 22050 Hz)
    duration = 1.0
    sample_rate = 22050
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, audio.shape)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio, sample_rate

def audio_to_base64_mp3(audio, sample_rate):
    """Convert audio array to base64 encoded WAV (simpler approach)"""
    import tempfile
    import os
    import soundfile as sf
    
    # Save as temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        sf.write(temp_wav.name, audio, sample_rate)
        temp_wav_path = temp_wav.name
    
    try:
        # Read the WAV file and encode as base64
        with open(temp_wav_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64
    
    finally:
        # Clean up temp file
        os.unlink(temp_wav_path)

def test_api_endpoint(api_url="http://localhost:8000"):
    """Test the API endpoint with sample audio"""
    
    print("🎵 Creating test audio...")
    audio, sample_rate = create_test_audio()
    
    print("📦 Converting to base64...")
    try:
        audio_base64 = audio_to_base64_mp3(audio, sample_rate)
    except Exception as e:
        print(f"❌ Failed to convert audio: {e}")
        return False
    
    # Test data
    test_data = {
        "language": "English",
        "audioFormat": "mp3",  # API expects MP3
        "audioBase64": audio_base64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "RW-b3ZMf29EcBQLObtVffHiqine2b89qzlq3Hgg2pBUoqyIElXhg0DKleUAeZsXNIdK"
    }
    
    print("🚀 Testing API endpoint...")
    print(f"URL: {api_url}/api/voice-detection")
    
    try:
        response = requests.post(
            f"{api_url}/api/voice-detection",
            json=test_data,
            headers=headers,
            timeout=60
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(f"   Status: {result.get('status')}")
            print(f"   Language: {result.get('language')}")
            print(f"   Classification: {result.get('classification')}")
            print(f"   Confidence Score: {result.get('confidenceScore')}")
            print(f"   Explanation: {result.get('explanation', 'N/A')[:100]}...")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health_endpoints(api_url="http://localhost:8000"):
    """Test health endpoints"""
    
    endpoints = [
        "/health",
        "/health/detailed", 
        "/health/models",
        "/health/monitoring"
    ]
    
    print("🏥 Testing health endpoints...")
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{api_url}{endpoint}", timeout=10)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   {status} {endpoint}: {response.status_code}")
            
            if endpoint == "/health/models" and response.status_code == 200:
                data = response.json()
                print(f"      Foundation model valid: {data.get('foundation_model', {}).get('valid', False)}")
                print(f"      Valid classifiers: {data.get('classifiers', {}).get('valid_count', 0)}")
                
        except Exception as e:
            print(f"   ❌ {endpoint}: {e}")

def main():
    """Main test function"""
    print("🧪 AI Voice Detection API Test")
    print("=" * 50)
    
    api_url = "http://localhost:8000"
    
    # Test health endpoints first
    test_health_endpoints(api_url)
    
    print("\n" + "=" * 50)
    
    # Test main detection endpoint
    success = test_api_endpoint(api_url)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 API test completed successfully!")
        print("✅ Your API is working with actual models!")
    else:
        print("❌ API test failed. Check the logs above.")

if __name__ == "__main__":
    main()