"""
Model validation module for AI voice detection system.

This module provides validation infrastructure for foundation models
and classifiers to ensure they are properly trained and functional,
including compatibility validation for model switches.
"""

from .validator import ModelValidator, EmbeddingValidator, FoundationModelManager
from .compatibility_validator import (
    ModelCompatibilityValidator,
    ModelCompatibilityResult,
    CompatibilityIssue,
    CompatibilityLevel,
    ValidationSeverity,
    ModelArchitecture
)

__all__ = [
    "ModelValidator", 
    "EmbeddingValidator", 
    "FoundationModelManager",
    "ModelCompatibilityValidator",
    "ModelCompatibilityResult",
    "CompatibilityIssue",
    "CompatibilityLevel",
    "ValidationSeverity",
    "ModelArchitecture"
]