#!/usr/bin/env python3
"""
Demonstration of Enhanced Confidence Scoring System.

This script demonstrates the improved confidence calculation methods that replace
random confidence generation with meaningful uncertainty measures.
"""

import numpy as np
from src.detection.confidence import EnhancedConfidenceCalculator, ConfidenceLevel

def main():
    """Demonstrate enhanced confidence scoring capabilities."""
    print("=== Enhanced Confidence Scoring Demonstration ===\n")
    
    # Initialize the enhanced confidence calculator
    calculator = EnhancedConfidenceCalculator()
    
    # Test scenarios with different model outputs and conditions
    test_scenarios = [
        {
            "name": "High Confidence AI Detection",
            "model_output": 0.95,
            "language": "English",
            "features": np.random.randn(1024) * 2.0,  # Good quality features
            "model_metadata": {
                'validation_accuracy': 0.92,
                'model_size_bytes': 5000000,
                'model_version': '1.0.0'
            }
        },
        {
            "name": "Uncertain Prediction (Close to Boundary)",
            "model_output": 0.52,
            "language": "English",
            "features": np.random.randn(1024) * 1.5,
            "model_metadata": {
                'validation_accuracy': 0.85,
                'model_size_bytes': 2000000,
                'model_version': '1.0.0'
            }
        },
        {
            "name": "Poor Quality Features",
            "model_output": 0.8,
            "language": "Tamil",
            "features": np.ones(1024) * 0.001,  # Very low variance features
            "model_metadata": {
                'validation_accuracy': 0.88,
                'model_size_bytes': 3000000,
                'model_version': '1.0.0'
            }
        },
        {
            "name": "Unstable Predictions",
            "model_output": 0.75,
            "language": "Hindi",
            "features": np.random.randn(1024) * 2.0,
            "prediction_history": [0.2, 0.9, 0.3, 0.8, 0.1],  # Very unstable
            "model_metadata": {
                'validation_accuracy': 0.90,
                'model_size_bytes': 4000000,
                'model_version': '1.0.0'
            }
        },
        {
            "name": "High Confidence Human Detection",
            "model_output": 0.05,
            "language": "English",
            "features": np.random.randn(1024) * 2.5,
            "model_metadata": {
                'validation_accuracy': 0.94,
                'model_size_bytes': 6000000,
                'model_version': '1.0.0'
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print("-" * 50)
        
        # Calculate enhanced confidence metrics
        metrics = calculator.calculate_enhanced_confidence(
            model_output=scenario['model_output'],
            language=scenario['language'],
            features=scenario.get('features'),
            model_metadata=scenario.get('model_metadata'),
            prediction_history=scenario.get('prediction_history')
        )
        
        # Display results
        print(f"Model Output: {scenario['model_output']:.3f}")
        print(f"Raw Confidence: {metrics.raw_confidence:.3f}")
        print(f"Adjusted Confidence: {metrics.adjusted_confidence:.3f}")
        print(f"Final Confidence: {metrics.final_confidence:.3f}")
        print(f"Confidence Level: {metrics.confidence_level.value.upper()}")
        print(f"Feature Quality: {metrics.feature_quality_score:.3f}")
        print(f"Prediction Stability: {metrics.prediction_stability:.3f}")
        print(f"Model Certainty: {metrics.model_certainty:.3f}")
        print(f"Calibration Score: {metrics.calibration_score:.3f}")
        
        # Show uncertainty indicators
        if metrics.uncertainty_indicators:
            print(f"\nUncertainty Indicators ({len(metrics.uncertainty_indicators)}):")
            for indicator in metrics.uncertainty_indicators:
                print(f"  • {indicator.uncertainty_type.value.title()}: {indicator.value:.3f}")
                print(f"    {indicator.description}")
        else:
            print("\nNo significant uncertainty indicators detected.")
        
        # Generate explanation
        explanation = calculator.get_confidence_explanation(metrics)
        print(f"\nExplanation:")
        print(f"  {explanation}")
        
        print("\n" + "="*70 + "\n")
    
    # Demonstrate comparison with simple confidence calculation
    print("=== Comparison with Simple Confidence Calculation ===\n")
    
    test_output = 0.52  # Uncertain prediction
    
    # Simple calculation (distance from 0.5)
    simple_confidence = min(abs(test_output - 0.5) * 2.0, 1.0)
    
    # Enhanced calculation
    enhanced_metrics = calculator.calculate_enhanced_confidence(
        model_output=test_output,
        language="English",
        features=np.random.randn(1024) * 1.0,  # Moderate quality features
        model_metadata={'validation_accuracy': 0.85, 'model_size_bytes': 2000000}
    )
    
    print(f"Model Output: {test_output}")
    print(f"Simple Confidence: {simple_confidence:.3f}")
    print(f"Enhanced Confidence: {enhanced_metrics.final_confidence:.3f}")
    print(f"Enhancement Factor: {enhanced_metrics.final_confidence / simple_confidence:.2f}x")
    
    print(f"\nThe enhanced system provides:")
    print(f"  • Uncertainty analysis ({len(enhanced_metrics.uncertainty_indicators)} indicators)")
    print(f"  • Feature quality assessment ({enhanced_metrics.feature_quality_score:.3f})")
    print(f"  • Model certainty evaluation ({enhanced_metrics.model_certainty:.3f})")
    print(f"  • Confidence calibration ({enhanced_metrics.calibration_score:.3f})")
    print(f"  • Human-readable confidence level ({enhanced_metrics.confidence_level.value})")

if __name__ == "__main__":
    main()