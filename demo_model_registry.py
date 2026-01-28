#!/usr/bin/env python3
"""
Demo script for Model Registry and Lifecycle Management system.

This script demonstrates the key features of the ModelRegistry and
ModelLifecycleManager classes including model registration, A/B testing,
rollback capabilities, and backward compatibility validation.
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from registry import ModelRegistry, ModelLifecycleManager, LifecyclePolicy
from registry.model_registry import ModelStatus
from models.schemas import PerformanceMetrics


def create_dummy_model_file(path: str, size_bytes: int = 1024):
    """Create a dummy model file for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'0' * size_bytes)


def demo_model_registry():
    """Demonstrate ModelRegistry functionality."""
    print("=== Model Registry Demo ===\n")
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = os.path.join(temp_dir, "registry")
        models_path = os.path.join(temp_dir, "models")
        
        # Initialize registry
        registry = ModelRegistry(registry_path)
        print(f"✓ Initialized ModelRegistry at {registry_path}")
        
        # Create dummy model files
        english_classifier_path = os.path.join(models_path, "english_classifier_v1.h5")
        english_classifier_v2_path = os.path.join(models_path, "english_classifier_v2.h5")
        foundation_model_path = os.path.join(models_path, "hubert_base.pt")
        
        create_dummy_model_file(english_classifier_path, 2048)
        create_dummy_model_file(english_classifier_v2_path, 2560)
        create_dummy_model_file(foundation_model_path, 10240)
        
        print("✓ Created dummy model files")
        
        # Register models
        print("\n--- Registering Models ---")
        
        # Register foundation model
        foundation_metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            sample_count=5000
        )
        
        foundation_record = registry.register_model(
            model_id="hubert_base_v1",
            model_type="foundation",
            file_path=foundation_model_path,
            validation_metrics=foundation_metrics,
            version="1.0.0",
            additional_metadata={"embedding_dim": 768, "architecture": "transformer"}
        )
        print(f"✓ Registered foundation model: {foundation_record.model_id}")
        
        # Register English classifier v1
        classifier_v1_metrics = PerformanceMetrics(
            accuracy=0.89,
            precision=0.87,
            recall=0.91,
            f1_score=0.89,
            sample_count=2000
        )
        
        classifier_v1_record = registry.register_model(
            model_id="english_classifier_v1",
            model_type="classifier",
            file_path=english_classifier_path,
            validation_metrics=classifier_v1_metrics,
            language="English",
            version="1.0.0",
            additional_metadata={"embedding_dim": 768, "architecture": "MLP"}
        )
        print(f"✓ Registered classifier v1: {classifier_v1_record.model_id}")
        
        # Register English classifier v2 (improved version)
        classifier_v2_metrics = PerformanceMetrics(
            accuracy=0.92,
            precision=0.90,
            recall=0.94,
            f1_score=0.92,
            sample_count=2500
        )
        
        classifier_v2_record = registry.register_model(
            model_id="english_classifier_v2",
            model_type="classifier",
            file_path=english_classifier_v2_path,
            validation_metrics=classifier_v2_metrics,
            language="English",
            version="2.0.0",
            parent_model_id="english_classifier_v1",
            additional_metadata={"embedding_dim": 768, "architecture": "MLP"}
        )
        print(f"✓ Registered classifier v2: {classifier_v2_record.model_id}")
        
        # List models
        print("\n--- Listing Models ---")
        all_models = registry.list_models()
        for model in all_models:
            print(f"  {model.model_id} ({model.model_type}, {model.language or 'N/A'}) - {model.status.value}")
        
        # Show model lineage
        print("\n--- Model Lineage ---")
        v1_children = registry.get_model_lineage("english_classifier_v1")
        print(f"english_classifier_v1 children: {v1_children}")
        
        v2_ancestry = registry.get_model_ancestry("english_classifier_v2")
        print(f"english_classifier_v2 ancestry: {v2_ancestry}")
        
        # Register thresholds
        print("\n--- Registering Thresholds ---")
        threshold_v1 = registry.register_threshold(
            language="English",
            model_version="1.0.0",
            threshold=0.52,
            validation_f1_score=0.89,
            precision=0.87,
            recall=0.91,
            sample_count=2000
        )
        print(f"✓ Registered threshold for v1: {threshold_v1.threshold}")
        
        threshold_v2 = registry.register_threshold(
            language="English",
            model_version="2.0.0",
            threshold=0.48,
            validation_f1_score=0.92,
            precision=0.90,
            recall=0.94,
            sample_count=2500
        )
        print(f"✓ Registered threshold for v2: {threshold_v2.threshold}")
        
        # Test compatibility validation
        print("\n--- Compatibility Validation ---")
        is_compatible, issues = registry.validate_model_compatibility(
            "english_classifier_v1", "english_classifier_v2"
        )
        print(f"v1 <-> v2 compatibility: {is_compatible}")
        if issues:
            for issue in issues:
                print(f"  Issue: {issue}")
        
        # Show registry statistics
        print("\n--- Registry Statistics ---")
        stats = registry.get_registry_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return registry


def demo_lifecycle_management(registry: ModelRegistry):
    """Demonstrate ModelLifecycleManager functionality."""
    print("\n\n=== Lifecycle Management Demo ===\n")
    
    # Initialize lifecycle manager
    policy = LifecyclePolicy(
        min_accuracy_threshold=0.85,
        ab_test_duration_days=1,  # Short for demo
        ab_test_min_samples=100   # Low for demo
    )
    
    lifecycle_manager = ModelLifecycleManager(registry, policy)
    print("✓ Initialized ModelLifecycleManager")
    
    # Promote v1 to production first
    print("\n--- Model Promotion ---")
    registry.update_model_status("english_classifier_v1", ModelStatus.ACTIVE)
    success = lifecycle_manager.promote_model_to_production(
        "english_classifier_v1", 
        "Initial production deployment"
    )
    print(f"✓ Promoted v1 to production: {success}")
    
    # Start A/B test between v1 and v2
    print("\n--- A/B Testing ---")
    try:
        test_id = lifecycle_manager.start_ab_test(
            control_model_id="english_classifier_v1",
            test_model_id="english_classifier_v2",
            language="English",
            traffic_split=0.2
        )
        print(f"✓ Started A/B test: {test_id}")
        
        # Simulate test data collection
        ab_test = registry.get_ab_test(test_id)
        print(f"  Test status: {ab_test.status.value}")
        print(f"  Traffic split: {ab_test.traffic_split}")
        
        # Update test metrics (simulated)
        v1_test_metrics = PerformanceMetrics(
            accuracy=0.88,
            precision=0.86,
            recall=0.90,
            f1_score=0.88,
            sample_count=500
        )
        
        v2_test_metrics = PerformanceMetrics(
            accuracy=0.93,
            precision=0.91,
            recall=0.95,
            f1_score=0.93,
            sample_count=500
        )
        
        registry.update_ab_test_metrics(
            test_id,
            model_a_metrics=v1_test_metrics,
            model_b_metrics=v2_test_metrics,
            sample_count_a=500,
            sample_count_b=500
        )
        print("✓ Updated A/B test metrics")
        
        # Evaluate test results
        evaluation = lifecycle_manager.evaluate_ab_test(test_id)
        print(f"  Evaluation status: {evaluation['status']}")
        print(f"  Winner: {evaluation.get('winner', 'None')}")
        print(f"  Recommendation: {evaluation.get('recommendation', 'None')}")
        
        # Complete the test
        completion = lifecycle_manager.complete_ab_test(test_id, auto_promote=True)
        print(f"✓ Completed A/B test, auto-promoted: {completion.get('auto_promoted', False)}")
        
    except Exception as e:
        print(f"A/B test error: {e}")
    
    # Test backward compatibility
    print("\n--- Backward Compatibility ---")
    is_compatible, issues = lifecycle_manager.validate_backward_compatibility(
        "english_classifier_v1", "english_classifier_v2"
    )
    print(f"Backward compatibility: {is_compatible}")
    if issues:
        for issue in issues:
            print(f"  Issue: {issue}")
    
    # Create compatibility report
    compat_report = lifecycle_manager.create_compatibility_report("english_classifier_v2")
    print(f"Compatibility score: {compat_report.get('overall_compatibility_score', 'N/A'):.2f}")
    
    # Test rollback capability
    print("\n--- Rollback Testing ---")
    try:
        # Create backup first
        backup_path = registry.create_model_backup("english_classifier_v2")
        print(f"✓ Created backup: {backup_path}")
        
        # Simulate performance degradation
        degraded_metrics = PerformanceMetrics(
            accuracy=0.75,  # Below threshold
            precision=0.73,
            recall=0.77,
            f1_score=0.75,
            sample_count=1000
        )
        
        rollback_triggered = lifecycle_manager.auto_rollback_on_degradation(
            "english_classifier_v2", degraded_metrics
        )
        print(f"Auto-rollback triggered: {rollback_triggered}")
        
    except Exception as e:
        print(f"Rollback test error: {e}")
    
    # Show lifecycle events
    print("\n--- Lifecycle Events ---")
    events = lifecycle_manager.get_lifecycle_events(limit=5)
    for event in events:
        print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.event_type} - {event.model_id}")
        print(f"    Reason: {event.reason}")


def main():
    """Run the complete demo."""
    print("Model Registry and Lifecycle Management Demo")
    print("=" * 50)
    
    try:
        # Demo model registry
        registry = demo_model_registry()
        
        # Demo lifecycle management
        demo_lifecycle_management(registry)
        
        print("\n" + "=" * 50)
        print("✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()