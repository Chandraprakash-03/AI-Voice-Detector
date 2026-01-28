#!/usr/bin/env python3
"""
Quick test to verify the classifier loading fix works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.detection.engine import DetectionEngine

def test_classifier_loading():
    """Test that classifiers load properly (either from file or as dummy)."""
    try:
        print("Testing classifier loading fix...")
        
        # Initialize detection engine
        engine = DetectionEngine()
        
        # Test loading classifiers for all supported languages
        languages = ["English", "Tamil", "Hindi", "Malayalam", "Telugu"]
        
        for lang in languages:
            print(f"Testing {lang} classifier...")
            try:
                classifier = engine._load_classifier(lang)
                print(f"✓ {lang} classifier loaded successfully")
            except Exception as e:
                print(f"✗ {lang} classifier failed: {e}")
                return False
        
        print("\n✅ All classifiers loaded successfully!")
        print("The fix is working - empty H5 files are handled gracefully.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_classifier_loading()
    sys.exit(0 if success else 1)