"""
Optimization Module

This module provides optimization components for the AI voice detection system,
including threshold optimization and performance tuning.
"""

from .threshold_optimizer import ThresholdOptimizer, ThresholdConfig, OptimizationResult

__all__ = [
    "ThresholdOptimizer",
    "ThresholdConfig", 
    "OptimizationResult"
]