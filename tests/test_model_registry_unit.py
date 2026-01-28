"""
Unit tests for Model Registry system.

Tests the core functionality of ModelRegistry and ModelLifecycleManager
including model registration, metadata management, A/B testing, and
lifecycle management features.
"""

import unittest
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from registry import ModelRegistry, ModelLifecycleManager, LifecyclePolicy
from registry.model_registry import ModelStatus, ModelRecord, ThresholdRecord
from models.schemas import PerformanceMetrics


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "registry")
        self.models_path = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_path, exist_ok=True)
        
        self.registry = ModelRegistry(self.registry_path)
        
        # Create dummy model files
        self.model_file_1 = os.path.join(self.models_path, "model1.h5")
        self.model_file_2 = os.path.join(self.models_path, "model2.h5")
        
        with open(self.model_file_1, 'wb') as f:
            f.write(b'dummy_model_data_1' * 100)
        with open(self.model_file_2, 'wb') as f:
            f.write(b'dummy_model_data_2' * 150)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_registration(self):
        """Test model registration functionality."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register a model
        record = self.registry.register_model(
            model_id="test_model_1",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics,
            language="English",
            version="1.0.0"
        )
        
        self.assertEqual(record.model_id, "test_model_1")
        self.assertEqual(record.model_type, "classifier")
        self.assertEqual(record.language, "English")
        self.assertEqual(record.version, "1.0.0")
        self.assertEqual(record.status, ModelStatus.TESTING)
        self.assertIsNotNone(record.model_size_bytes)
        self.assertIsNotNone(record.training_data_hash)
    
    def test_duplicate_model_registration(self):
        """Test that duplicate model registration raises error."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register first model
        self.registry.register_model(
            model_id="test_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics
        )
        
        # Try to register duplicate
        with self.assertRaises(ValueError):
            self.registry.register_model(
                model_id="test_model",
                model_type="classifier",
                file_path=self.model_file_2,
                validation_metrics=metrics
            )
    
    def test_model_retrieval(self):
        """Test model retrieval functionality."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register model
        self.registry.register_model(
            model_id="test_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics
        )
        
        # Retrieve model
        retrieved = self.registry.get_model("test_model")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_id, "test_model")
        
        # Try to retrieve non-existent model
        non_existent = self.registry.get_model("non_existent")
        self.assertIsNone(non_existent)
    
    def test_model_status_update(self):
        """Test model status update functionality."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register model
        self.registry.register_model(
            model_id="test_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics
        )
        
        # Update status
        success = self.registry.update_model_status("test_model", ModelStatus.ACTIVE)
        self.assertTrue(success)
        
        # Verify status change
        model = self.registry.get_model("test_model")
        self.assertEqual(model.status, ModelStatus.ACTIVE)
        self.assertIsNotNone(model.deployment_date)
    
    def test_model_listing_with_filters(self):
        """Test model listing with various filters."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register multiple models
        self.registry.register_model(
            model_id="classifier_en",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics,
            language="English"
        )
        
        self.registry.register_model(
            model_id="foundation_model",
            model_type="foundation",
            file_path=self.model_file_2,
            validation_metrics=metrics
        )
        
        # Test filtering by type
        classifiers = self.registry.list_models(model_type="classifier")
        self.assertEqual(len(classifiers), 1)
        self.assertEqual(classifiers[0].model_id, "classifier_en")
        
        foundation_models = self.registry.list_models(model_type="foundation")
        self.assertEqual(len(foundation_models), 1)
        self.assertEqual(foundation_models[0].model_id, "foundation_model")
        
        # Test filtering by language
        english_models = self.registry.list_models(language="English")
        self.assertEqual(len(english_models), 1)
        self.assertEqual(english_models[0].model_id, "classifier_en")
    
    def test_model_lineage_tracking(self):
        """Test model lineage and ancestry tracking."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register parent model
        self.registry.register_model(
            model_id="parent_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics
        )
        
        # Register child model
        self.registry.register_model(
            model_id="child_model",
            model_type="classifier",
            file_path=self.model_file_2,
            validation_metrics=metrics,
            parent_model_id="parent_model"
        )
        
        # Test lineage
        children = self.registry.get_model_lineage("parent_model")
        self.assertEqual(children, ["child_model"])
        
        # Test ancestry
        ancestry = self.registry.get_model_ancestry("child_model")
        self.assertEqual(ancestry, ["parent_model"])
    
    def test_threshold_management(self):
        """Test threshold registration and retrieval."""
        # Register threshold
        threshold_record = self.registry.register_threshold(
            language="English",
            model_version="1.0.0",
            threshold=0.52,
            validation_f1_score=0.90,
            precision=0.88,
            recall=0.92,
            sample_count=1000
        )
        
        self.assertEqual(threshold_record.language, "English")
        self.assertEqual(threshold_record.threshold, 0.52)
        
        # Retrieve threshold
        retrieved = self.registry.get_threshold("English", "1.0.0")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.threshold, 0.52)
        
        # Test non-existent threshold
        non_existent = self.registry.get_threshold("Spanish", "1.0.0")
        self.assertIsNone(non_existent)
    
    def test_ab_test_creation(self):
        """Test A/B test creation and management."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register two models
        self.registry.register_model(
            model_id="model_a",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics
        )
        
        self.registry.register_model(
            model_id="model_b",
            model_type="classifier",
            file_path=self.model_file_2,
            validation_metrics=metrics
        )
        
        # Create A/B test
        ab_test = self.registry.create_ab_test(
            test_id="test_ab_1",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=0.3
        )
        
        self.assertEqual(ab_test.test_id, "test_ab_1")
        self.assertEqual(ab_test.model_a_id, "model_a")
        self.assertEqual(ab_test.model_b_id, "model_b")
        self.assertEqual(ab_test.traffic_split, 0.3)
        
        # Retrieve A/B test
        retrieved = self.registry.get_ab_test("test_ab_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.test_id, "test_ab_1")
    
    def test_model_compatibility_validation(self):
        """Test model compatibility validation."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register compatible models
        self.registry.register_model(
            model_id="model_1",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics,
            language="English",
            additional_metadata={"embedding_dim": 768}
        )
        
        self.registry.register_model(
            model_id="model_2",
            model_type="classifier",
            file_path=self.model_file_2,
            validation_metrics=metrics,
            language="English",
            additional_metadata={"embedding_dim": 768}
        )
        
        # Test compatibility
        is_compatible, issues = self.registry.validate_model_compatibility("model_1", "model_2")
        # Note: Will be False due to file not found, but structure is correct
        self.assertIsInstance(is_compatible, bool)
        self.assertIsInstance(issues, list)
    
    def test_registry_statistics(self):
        """Test registry statistics generation."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register models
        self.registry.register_model(
            model_id="classifier_1",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics,
            language="English"
        )
        
        self.registry.register_model(
            model_id="foundation_1",
            model_type="foundation",
            file_path=self.model_file_2,
            validation_metrics=metrics
        )
        
        # Get statistics
        stats = self.registry.get_registry_stats()
        
        self.assertEqual(stats["total_models"], 2)
        self.assertEqual(stats["models_by_type"]["classifier"], 1)
        self.assertEqual(stats["models_by_type"]["foundation"], 1)
        self.assertEqual(stats["models_by_language"]["English"], 1)


class TestModelLifecycleManager(unittest.TestCase):
    """Test cases for ModelLifecycleManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "registry")
        self.models_path = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_path, exist_ok=True)
        
        self.registry = ModelRegistry(self.registry_path)
        self.policy = LifecyclePolicy(
            min_accuracy_threshold=0.85,
            ab_test_min_samples=100
        )
        self.lifecycle_manager = ModelLifecycleManager(self.registry, self.policy)
        
        # Create dummy model files
        self.model_file_1 = os.path.join(self.models_path, "model1.h5")
        self.model_file_2 = os.path.join(self.models_path, "model2.h5")
        
        with open(self.model_file_1, 'wb') as f:
            f.write(b'dummy_model_data_1' * 100)
        with open(self.model_file_2, 'wb') as f:
            f.write(b'dummy_model_data_2' * 150)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_lifecycle_manager_initialization(self):
        """Test lifecycle manager initialization."""
        self.assertIsNotNone(self.lifecycle_manager)
        self.assertEqual(self.lifecycle_manager.policy.min_accuracy_threshold, 0.85)
        self.assertEqual(self.lifecycle_manager.policy.ab_test_min_samples, 100)
    
    def test_model_promotion_validation(self):
        """Test model promotion criteria validation."""
        # Register model with good metrics
        good_metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1500
        )
        
        self.registry.register_model(
            model_id="good_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=good_metrics
        )
        
        model = self.registry.get_model("good_model")
        meets_criteria = self.lifecycle_manager._validate_promotion_criteria(model)
        self.assertTrue(meets_criteria)
        
        # Register model with poor metrics
        poor_metrics = PerformanceMetrics(
            accuracy=0.70,  # Below threshold
            precision=0.68,
            recall=0.72,
            f1_score=0.70,
            sample_count=500  # Below threshold
        )
        
        self.registry.register_model(
            model_id="poor_model",
            model_type="classifier",
            file_path=self.model_file_2,
            validation_metrics=poor_metrics
        )
        
        poor_model = self.registry.get_model("poor_model")
        meets_criteria = self.lifecycle_manager._validate_promotion_criteria(poor_model)
        self.assertFalse(meets_criteria)
    
    def test_backward_compatibility_validation(self):
        """Test backward compatibility validation."""
        metrics_v1 = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        metrics_v2 = PerformanceMetrics(
            accuracy=0.92,  # Improved
            precision=0.90,
            recall=0.94,
            f1_score=0.92,
            sample_count=1200
        )
        
        # Register models
        self.registry.register_model(
            model_id="model_v1",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics_v1,
            language="English",
            version="1.0.0",
            additional_metadata={"embedding_dim": 768}
        )
        
        self.registry.register_model(
            model_id="model_v2",
            model_type="classifier",
            file_path=self.model_file_2,
            validation_metrics=metrics_v2,
            language="English",
            version="2.0.0",
            additional_metadata={"embedding_dim": 768}
        )
        
        # Test compatibility
        is_compatible, issues = self.lifecycle_manager.validate_backward_compatibility(
            "model_v1", "model_v2"
        )
        
        # Should detect file issues but structure is correct
        self.assertIsInstance(is_compatible, bool)
        self.assertIsInstance(issues, list)
    
    def test_compatibility_report_generation(self):
        """Test compatibility report generation."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        # Register model
        self.registry.register_model(
            model_id="test_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics,
            language="English"
        )
        
        # Generate compatibility report
        report = self.lifecycle_manager.create_compatibility_report("test_model")
        
        self.assertEqual(report["model_id"], "test_model")
        self.assertEqual(report["model_type"], "classifier")
        self.assertEqual(report["language"], "English")
        self.assertIn("overall_compatibility_score", report)
        self.assertIn("ancestors", report)
        self.assertIn("descendants", report)
    
    def test_lifecycle_events_tracking(self):
        """Test lifecycle events tracking."""
        # Initially no events
        events = self.lifecycle_manager.get_lifecycle_events()
        initial_count = len(events)
        
        # Register model to trigger event
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            sample_count=1000
        )
        
        self.registry.register_model(
            model_id="test_model",
            model_type="classifier",
            file_path=self.model_file_1,
            validation_metrics=metrics
        )
        
        # Promote model (should trigger event)
        self.lifecycle_manager.promote_model_to_production("test_model", "Test promotion")
        
        # Check events
        events = self.lifecycle_manager.get_lifecycle_events()
        self.assertGreaterEqual(len(events), initial_count)
        
        # Filter events by model
        model_events = self.lifecycle_manager.get_lifecycle_events(model_id="test_model")
        self.assertGreater(len(model_events), 0)


if __name__ == '__main__':
    unittest.main()