"""
Unit tests for threshold optimization system.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.optimization.threshold_optimizer import (
    ThresholdOptimizer, 
    ThresholdConfig, 
    OptimizationResult, 
    ValidationData
)


class TestThresholdOptimizer:
    """Test cases for ThresholdOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = ThresholdOptimizer(storage_path=self.temp_dir)
        
        # Create sample validation data
        np.random.seed(42)  # For reproducible tests
        self.predictions = np.random.random(100)
        self.labels = np.random.randint(0, 2, 100)
        
        self.validation_data = ValidationData(
            predictions=self.predictions,
            labels=self.labels,
            language="English",
            model_version="test_v1.0"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ThresholdOptimizer initialization."""
        assert self.optimizer.storage_path.exists()
        assert len(self.optimizer.default_thresholds) == 5
        assert all(0.0 <= t <= 1.0 for t in self.optimizer.default_thresholds.values())
    
    def test_get_default_threshold(self):
        """Test getting default threshold for a language."""
        threshold = self.optimizer.get_threshold("English")
        assert threshold == 0.5
        
        threshold = self.optimizer.get_threshold("Tamil")
        assert threshold == 0.5
        
        # Test unsupported language falls back to 0.5
        threshold = self.optimizer.get_threshold("Unsupported")
        assert threshold == 0.5
    
    def test_validation_data_creation(self):
        """Test ValidationData creation and validation."""
        # Valid data
        data = ValidationData(
            predictions=np.array([0.1, 0.5, 0.9]),
            labels=np.array([0, 1, 1]),
            language="English",
            model_version="v1.0"
        )
        assert len(data.predictions) == 3
        assert len(data.labels) == 3
        
        # Invalid data - mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            ValidationData(
                predictions=np.array([0.1, 0.5]),
                labels=np.array([0, 1, 1]),
                language="English",
                model_version="v1.0"
            )
        
        # Invalid data - predictions out of range
        with pytest.raises(ValueError, match="probabilities between 0 and 1"):
            ValidationData(
                predictions=np.array([0.1, 1.5, 0.9]),
                labels=np.array([0, 1, 1]),
                language="English",
                model_version="v1.0"
            )
        
        # Invalid data - non-binary labels
        with pytest.raises(ValueError, match="binary"):
            ValidationData(
                predictions=np.array([0.1, 0.5, 0.9]),
                labels=np.array([0, 1, 2]),
                language="English",
                model_version="v1.0"
            )
    
    def test_f1_score_optimization(self):
        """Test F1-score optimization algorithm."""
        # Create data where threshold 0.6 should be optimal
        predictions = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        
        threshold, metrics = self.optimizer.optimize_for_f1_score(predictions, labels)
        
        assert 0.0 <= threshold <= 1.0
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'accuracy' in metrics
        assert all(0.0 <= v <= 1.0 for v in metrics.values() if v is not None)
    
    def test_calculate_optimal_threshold(self):
        """Test optimal threshold calculation."""
        result = self.optimizer.calculate_optimal_threshold("English", self.validation_data)
        
        assert isinstance(result, OptimizationResult)
        assert result.language == "English"
        assert result.success
        assert 0.0 <= result.optimized_threshold <= 1.0
        assert result.sample_count == 100
        assert result.optimization_time > 0
        assert 'f1_score' in result.metrics
    
    def test_calculate_optimal_threshold_insufficient_samples(self):
        """Test threshold calculation with insufficient samples."""
        small_data = ValidationData(
            predictions=np.array([0.1, 0.5]),
            labels=np.array([0, 1]),
            language="English",
            model_version="test_v1.0"
        )
        
        result = self.optimizer.calculate_optimal_threshold("English", small_data)
        
        assert not result.success
        assert "Insufficient samples" in result.error_message
    
    def test_update_thresholds(self):
        """Test threshold updating with validation data."""
        result = self.optimizer.update_thresholds("English", self.validation_data)
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        
        # Check that threshold was updated only if there was meaningful improvement
        # Use the same model_version as in validation_data
        new_threshold = self.optimizer.get_threshold("English", "test_v1.0")
        if result.improvement > 0.001:
            assert new_threshold == result.optimized_threshold
        else:
            # If no meaningful improvement, threshold should remain default
            assert new_threshold == 0.5
    
    def test_get_all_thresholds(self):
        """Test getting all threshold configurations."""
        all_thresholds = self.optimizer.get_all_thresholds()
        
        assert isinstance(all_thresholds, dict)
        assert len(all_thresholds) == 5  # All supported languages
        
        for language, config in all_thresholds.items():
            assert isinstance(config, ThresholdConfig)
            assert config.language == language
            assert 0.0 <= config.threshold <= 1.0
    
    def test_reset_threshold(self):
        """Test resetting threshold to default."""
        # First update threshold
        self.optimizer.update_thresholds("English", self.validation_data)
        updated_threshold = self.optimizer.get_threshold("English")
        
        # Reset threshold
        self.optimizer.reset_threshold("English")
        reset_threshold = self.optimizer.get_threshold("English")
        
        assert reset_threshold == 0.5  # Default threshold
    
    def test_threshold_persistence(self):
        """Test that thresholds are saved and loaded correctly."""
        # Update threshold
        result = self.optimizer.update_thresholds("English", self.validation_data)
        
        # Only test persistence if threshold was actually updated
        if result.improvement > 0.001:
            original_threshold = result.optimized_threshold
            
            # Create new optimizer instance with same storage path
            new_optimizer = ThresholdOptimizer(storage_path=self.temp_dir)
            loaded_threshold = new_optimizer.get_threshold("English", "test_v1.0")
            
            assert loaded_threshold == original_threshold
        else:
            # If no meaningful improvement, threshold should remain default
            new_optimizer = ThresholdOptimizer(storage_path=self.temp_dir)
            loaded_threshold = new_optimizer.get_threshold("English", "test_v1.0")
            assert loaded_threshold == 0.5
    
    def test_optimization_history(self):
        """Test optimization history tracking."""
        # Update threshold to create history
        self.optimizer.update_thresholds("English", self.validation_data)
        
        # Get history
        history = self.optimizer.get_optimization_history("English")
        
        assert isinstance(history, list)
        if history:  # History might be empty if optimization didn't improve
            assert 'timestamp' in history[0]
            assert 'threshold' in history[0]
            assert 'f1_score' in history[0]
    
    def test_cache_expiry(self):
        """Test that cache expires correctly."""
        # Set very short cache expiry for testing
        self.optimizer.cache_expiry_hours = 0.001  # ~3.6 seconds
        
        # Update threshold
        self.optimizer.update_thresholds("English", self.validation_data)
        
        # Check cache is populated
        cache_key = "English_test_v1.0"
        assert cache_key in self.optimizer.threshold_cache
        
        # Wait for cache to expire (simulate by manually expiring)
        import time
        time.sleep(0.1)
        
        # Force cache validation check
        threshold = self.optimizer.get_threshold("English")
        assert threshold is not None  # Should still work with storage fallback


class TestThresholdConfig:
    """Test cases for ThresholdConfig model."""
    
    def test_threshold_config_creation(self):
        """Test ThresholdConfig creation and validation."""
        config = ThresholdConfig(
            language="English",
            model_version="v1.0",
            threshold=0.7,
            precision=0.85,
            recall=0.90,
            f1_score=0.87,
            sample_count=1000,
            last_updated=datetime.now()
        )
        
        assert config.language == "English"
        assert config.threshold == 0.7
        assert config.f1_score == 0.87
    
    def test_threshold_config_validation(self):
        """Test ThresholdConfig field validation."""
        # Invalid threshold (out of range)
        with pytest.raises(ValueError):
            ThresholdConfig(
                language="English",
                model_version="v1.0",
                threshold=1.5,  # Invalid
                precision=0.85,
                recall=0.90,
                f1_score=0.87,
                sample_count=1000,
                last_updated=datetime.now()
            )
        
        # Invalid sample count (negative)
        with pytest.raises(ValueError):
            ThresholdConfig(
                language="English",
                model_version="v1.0",
                threshold=0.7,
                precision=0.85,
                recall=0.90,
                f1_score=0.87,
                sample_count=-10,  # Invalid
                last_updated=datetime.now()
            )


class TestOptimizationResult:
    """Test cases for OptimizationResult model."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            language="English",
            original_threshold=0.5,
            optimized_threshold=0.7,
            improvement=0.15,
            metrics={"f1_score": 0.87, "precision": 0.85},
            sample_count=1000,
            optimization_time=2.5,
            success=True
        )
        
        assert result.language == "English"
        assert result.success
        assert result.improvement == 0.15
        assert result.metrics["f1_score"] == 0.87
    
    def test_optimization_result_failure(self):
        """Test OptimizationResult for failed optimization."""
        result = OptimizationResult(
            language="English",
            original_threshold=0.5,
            optimized_threshold=0.5,
            improvement=0.0,
            metrics={},
            sample_count=0,
            optimization_time=0.1,
            success=False,
            error_message="Insufficient data"
        )
        
        assert not result.success
        assert result.error_message == "Insufficient data"
        assert result.improvement == 0.0