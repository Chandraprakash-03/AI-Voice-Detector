"""
Enhanced error handling system for AI voice detection.

This module provides comprehensive error handling with specific error messages,
detailed logging for fallback scenarios, and error recovery mechanisms.
"""

from .error_manager import ErrorManager, ErrorContext, ErrorRecoveryResult
from .logging_manager import LoggingManager, FallbackLogger
from .recovery_strategies import RecoveryStrategy, RecoveryLevel

__all__ = [
    'ErrorManager',
    'ErrorContext', 
    'ErrorRecoveryResult',
    'LoggingManager',
    'FallbackLogger',
    'RecoveryStrategy',
    'RecoveryLevel'
]