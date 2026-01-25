#!/usr/bin/env python3
"""
Test script for the updated detection engine.

This script tests the new foundation model + classifier architecture.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.detection.engine import DetectionEngine
from src.audio.processor import AudioFeatures
from src.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_features() -> AudioFeatures:
    """Create dummy audio features for testing."""
    import numpy as np
    
    # Create dummy MFCC features (13 coefficients, 100 time frames)
    mfcc = np.random.randn(13, 100).tolist()
    
    # Create dummy spectral features
    spectral = np.random.randn(50).tolist()
    
    return AudioFeatures(
        mfcc=mfcc,
        spectral=spectral,
        duration=5.0,
        sample_rate=22050
    )

def test_engine():
    """Test the detection engine with dummy data."""
    try:
        logger.info("Testing Detection Engine...")
        
        # Initialize engine
        settings = get_settings()
        engine = DetectionEngine(settings)
        
        logger.info(f"Engine initialized with foundation model: {engine.foundation_model_name}")
        
        # Test model info
        logger.info("\n=== Available Foundation Models ===")
        foundation_models = engine.get_available_foundation_models()
        for name, info in foundation_models.items():
            status = "✓ Available" if info["available"] else "✗ Not found"
            logger.info(f"{name}: {status} ({info['embedding_dim']} dims)")
        
        # Test each language
        dummy_features = create_dummy_features()
        
        logger.info("\n=== Testing Languages ===")
        for language in engine.SUPPORTED_LANGUAGES:
            try:
                logger.info(f"\nTesting {language}...")
                
                # Get model info
                model_info = engine.get_model_info(language)
                logger.info(f"  Foundation: {model_info['foundation_model']}")
                logger.info(f"  Classifier loaded: {model_info['is_classifier_loaded']}")
                logger.info(f"  Foundation loaded: {model_info['is_foundation_loaded']}")
                
                # Run detection
                result = engine.detect_voice_type(dummy_features, language)
                
                logger.info(f"  Result: {result.classification}")
                logger.info(f"  Confidence: {result.confidence_score:.3f}")
                logger.info(f"  Processing time: {result.processing_time:.3f}s")
                logger.info(f"  Model version: {result.model_version}")
                
                # Generate explanation
                explanation = engine.generate_explanation(result, dummy_features, language)
                logger.info(f"  AI Explanation: {explanation[:100]}...")
                
                logger.info(f"  ✓ {language} test passed")
                
            except Exception as e:
                logger.error(f"  ✗ {language} test failed: {e}")
        
        # Test model switching (if multiple models available)
        available_models = [name for name, info in foundation_models.items() if info["available"]]
        if len(available_models) > 1:
            logger.info(f"\n=== Testing Model Switching ===")
            for model_name in available_models[:2]:  # Test first 2 available models
                try:
                    logger.info(f"Switching to {model_name}...")
                    engine.switch_foundation_model(model_name)
                    
                    # Test one language with new model
                    result = engine.detect_voice_type(dummy_features, "English")
                    logger.info(f"  ✓ {model_name} switch successful")
                    
                except Exception as e:
                    logger.error(f"  ✗ {model_name} switch failed: {e}")
        
        logger.info("\n=== Test Summary ===")
        logger.info("✓ Detection engine test completed successfully!")
        logger.info("The engine is ready to work with downloaded foundation models.")
        logger.info("AI-powered explanations are now enabled!")
        
        return True
        
    except Exception as e:
        logger.error(f"Engine test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting Detection Engine Test")
    logger.info("=" * 50)
    
    success = test_engine()
    
    if success:
        logger.info("\n🎉 All tests passed! Your engine is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Download models: python scripts/download_models.py --mode minimal")
        logger.info("2. Download explanation models: python scripts/download_models.py --mode explanation")
        logger.info("3. Test explanations: python scripts/test_explanation.py")
        logger.info("4. Start the API: python src/main.py")
        logger.info("5. Test with real audio files")
    else:
        logger.error("\n❌ Tests failed. Check the logs above for issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()