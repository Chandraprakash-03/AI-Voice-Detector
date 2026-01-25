#!/usr/bin/env python3
"""
Test script for the AI-powered explanation generator.

This script tests the explanation generation with different models and scenarios.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.explanation import ExplanationGenerator
from src.detection.engine import DetectionResult
from src.audio.processor import AudioFeatures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_scenarios():
    """Create test scenarios for explanation generation."""
    import numpy as np
    
    # Create dummy audio features
    mfcc = np.random.randn(13, 100).tolist()
    spectral = np.random.randn(50).tolist()
    
    features = AudioFeatures(
        mfcc=mfcc,
        spectral=spectral,
        duration=5.0,
        sample_rate=22050
    )
    
    # Test scenarios
    scenarios = [
        {
            "name": "High Confidence Human",
            "result": DetectionResult(
                classification="HUMAN",
                confidence_score=0.92,
                model_version="xls-r-300m-1.0.0",
                processing_time=0.15
            ),
            "language": "English"
        },
        {
            "name": "High Confidence AI",
            "result": DetectionResult(
                classification="AI_GENERATED",
                confidence_score=0.88,
                model_version="hubert-base-1.0.0",
                processing_time=0.12
            ),
            "language": "Tamil"
        },
        {
            "name": "Low Confidence Human",
            "result": DetectionResult(
                classification="HUMAN",
                confidence_score=0.55,
                model_version="xls-r-300m-1.0.0",
                processing_time=0.18
            ),
            "language": "Hindi"
        },
        {
            "name": "Low Confidence AI",
            "result": DetectionResult(
                classification="AI_GENERATED",
                confidence_score=0.62,
                model_version="hubert-base-1.0.0",
                processing_time=0.14
            ),
            "language": "Malayalam"
        }
    ]
    
    return features, scenarios

def test_explanation_models():
    """Test different explanation models."""
    features, scenarios = create_test_scenarios()
    
    # Test different models
    models_to_test = ["distilgpt2", "flan-t5-small"]
    
    for model_name in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Explanation Model: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize explanation generator
            generator = ExplanationGenerator(model_name)
            
            # Get model info
            model_info = generator.get_model_info()
            logger.info(f"Model loaded: {model_info['is_loaded']}")
            logger.info(f"Transformers available: {model_info['transformers_available']}")
            
            # Test each scenario
            for scenario in scenarios:
                logger.info(f"\n--- {scenario['name']} ---")
                logger.info(f"Classification: {scenario['result'].classification}")
                logger.info(f"Confidence: {scenario['result'].confidence_score:.2%}")
                logger.info(f"Language: {scenario['language']}")
                
                # Generate explanation
                explanation = generator.generate_explanation(
                    scenario['result'],
                    features,
                    scenario['language']
                )
                
                logger.info(f"Explanation: {explanation}")
                logger.info(f"Length: {len(explanation)} characters")
            
            # Clear cache
            generator.clear_cache()
            logger.info(f"✓ {model_name} test completed successfully")
            
        except Exception as e:
            logger.error(f"✗ {model_name} test failed: {e}")

def test_fallback_explanations():
    """Test fallback explanations when models are not available."""
    logger.info(f"\n{'='*60}")
    logger.info("Testing Fallback Explanations")
    logger.info(f"{'='*60}")
    
    features, scenarios = create_test_scenarios()
    
    try:
        # Initialize with a non-existent model to force fallback
        generator = ExplanationGenerator("non-existent-model")
        
        for scenario in scenarios:
            logger.info(f"\n--- {scenario['name']} (Fallback) ---")
            
            explanation = generator.generate_explanation(
                scenario['result'],
                features,
                scenario['language']
            )
            
            logger.info(f"Fallback Explanation: {explanation}")
        
        logger.info("✓ Fallback explanation test completed")
        
    except Exception as e:
        logger.error(f"✗ Fallback explanation test failed: {e}")

def test_integration_with_detection_engine():
    """Test integration with the detection engine."""
    logger.info(f"\n{'='*60}")
    logger.info("Testing Integration with Detection Engine")
    logger.info(f"{'='*60}")
    
    try:
        from src.detection.engine import DetectionEngine
        from src.config import get_settings
        
        # Initialize detection engine
        settings = get_settings()
        engine = DetectionEngine(settings)
        
        # Create test features
        features, _ = create_test_scenarios()
        
        # Test with one language
        language = "English"
        logger.info(f"Testing detection + explanation for {language}")
        
        # Run detection
        result = engine.detect_voice_type(features, language)
        logger.info(f"Detection result: {result.classification} ({result.confidence_score:.2%})")
        
        # Generate explanation using the engine's method (which now uses AI)
        explanation = engine.generate_explanation(result, features, language)
        logger.info(f"AI-generated explanation: {explanation}")
        
        logger.info("✓ Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")

def main():
    """Main test function."""
    logger.info("Starting AI Explanation Generator Tests")
    logger.info("=" * 80)
    
    # Test 1: Different explanation models
    test_explanation_models()
    
    # Test 2: Fallback explanations
    test_fallback_explanations()
    
    # Test 3: Integration with detection engine
    test_integration_with_detection_engine()
    
    logger.info(f"\n{'='*80}")
    logger.info("🎉 All explanation tests completed!")
    logger.info("\nNext steps:")
    logger.info("1. Download explanation models: python scripts/download_models.py --mode explanation")
    logger.info("2. Test with real models: python scripts/test_explanation.py")
    logger.info("3. Start the API with AI explanations: python src/main.py")

if __name__ == "__main__":
    main()