#!/usr/bin/env python3
"""
Test environment variables in the actual process
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_env_vars():
    """Test environment variables"""
    
    print("🔧 Environment Variables Test")
    print("=" * 50)
    
    print(f"ENVIRONMENT: {os.getenv('ENVIRONMENT', 'NOT SET')}")
    print(f"FOUNDATION_MODEL: {os.getenv('FOUNDATION_MODEL', 'NOT SET')}")
    
    # Test config loading
    from src.config import get_settings
    
    settings = get_settings()
    
    print(f"\nLoaded Settings:")
    print(f"Environment: {settings.environment}")
    print(f"Foundation model: {settings.ml.foundation_model}")
    
    # Test detection engine
    from src.detection.engine import DetectionEngine
    
    print(f"\nTesting DetectionEngine:")
    print(f"Settings foundation model: {settings.ml.foundation_model}")
    engine = DetectionEngine(settings, foundation_model=settings.ml.foundation_model)
    print(f"Selected foundation model: {engine.foundation_model_name}")

if __name__ == "__main__":
    test_env_vars()