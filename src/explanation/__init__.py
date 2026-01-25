"""
Explanation generation module for AI voice detection results.

This module provides AI-powered explanation generation using small language models
to create natural, contextual explanations instead of hardcoded templates.
"""

from .generator import ExplanationGenerator, get_explanation_generator

__all__ = ["ExplanationGenerator", "get_explanation_generator"]