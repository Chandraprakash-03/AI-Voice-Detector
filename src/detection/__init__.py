"""
Detection engine module for AI-Generated Voice Detection API.

This module provides the core ML-based detection functionality for classifying
voice recordings as AI-generated or human speech.
"""

from .engine import DetectionEngine, DetectionResult, DetectionEngineError

__all__ = ["DetectionEngine", "DetectionResult", "DetectionEngineError"]