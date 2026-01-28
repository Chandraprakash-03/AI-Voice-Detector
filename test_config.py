#!/usr/bin/env python3
"""
Test configuration loading
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_settings

def test_config():
    """Test configuration loading"""
    
    print("🔧 Testing Configuration Loading")
    print("=" * 50)
    
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Foundation model setting: {settings.ml.foundation_model}")
    print(f"Models base path: {settings.ml.models_base_path}")
    print(f"Foundation models path: {settings.ml.foundation_models_path}")
    
    # Check if the setting is being read correctly
    import os
    print(f"\nEnvironment variable FOUNDATION_MODEL: {os.getenv('FOUNDATION_MODEL')}")
    
    # Check all ML settings
    print(f"\nAll ML settings:")
    for attr in dir(settings.ml):
        if not attr.startswith('_'):
            value = getattr(settings.ml, attr)
            print(f"  {attr}: {value}")

if __name__ == "__main__":
    test_config()