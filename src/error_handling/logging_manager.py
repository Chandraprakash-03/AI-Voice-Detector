"""
Enhanced logging management for fallback scenarios and error tracking.

This module provides detailed logging for all fallback scenarios with
structured logging, performance metrics, and error correlation.
"""

import logging
import json
import time
from typing import Dict, Optional, List, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Enhanced log levels for fallback scenarios."""
    FALLBACK_INFO = "fallback_info"
    FALLBACK_WARNING = "fallback_warning"
    FALLBACK_ERROR = "fallback_error"
    RECOVERY_SUCCESS = "recovery_success"
    RECOVERY_FAILURE = "recovery_failure"


@dataclass
class FallbackLogEntry:
    """Structured log entry for fallback scenarios."""
    timestamp: datetime
    component: str
    operation: str
    fallback_trigger: str
    fallback_method: str
    language: Optional[str] = None
    model_name: Optional[str] = None
    original_error: Optional[str] = None
    fallback_success: bool = True
    processing_time: float = 0.0
    performance_impact: Optional[float] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FallbackLogger:
    """
    Specialized logger for fallback scenarios.
    
    Provides detailed logging for all fallback scenarios with structured
    logging, performance metrics, and error correlation.
    """
    
    def __init__(self, component_name: str, settings=None):
        """
        Initialize the FallbackLogger.
        
        Args:
            component_name: Name of the component using this logger
            settings: Application settings object containing configuration
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.component_name = component_name
        self.settings = settings
        
        # Fallback log entries for analysis
        self.fallback_entries: List[FallbackLogEntry] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Setup structured logging
        self.logger = logging.getLogger(f"fallback.{component_name}")
        
        logger.info(f"FallbackLogger initialized for {component_name}")
    
    def log_fallback_trigger(self, operation: str, trigger_reason: str, 
                           original_error: Exception, language: Optional[str] = None,
                           model_name: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log the triggering of a fallback scenario.
        
        Args:
            operation: Operation that triggered fallback
            trigger_reason: Reason for fallback activation
            original_error: Original error that caused fallback
            language: Target language (if applicable)
            model_name: Model name (if applicable)
            metadata: Additional metadata
            
        Returns:
            Fallback session ID for tracking
        """
        try:
            session_id = f"fallback_{int(time.time() * 1000)}"
            
            log_data = {
                "session_id": session_id,
                "component": self.component_name,
                "operation": operation,
                "trigger_reason": trigger_reason,
                "original_error_type": type(original_error).__name__,
                "original_error_message": str(original_error),
                "language": language,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            self.logger.warning(
                f"Fallback triggered in {self.component_name} for {operation}: {trigger_reason}",
                extra={"fallback_trigger": log_data}
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to log fallback trigger: {e}")
            return f"error_{int(time.time() * 1000)}"
    
    def log_fallback_attempt(self, session_id: str, fallback_method: str,
                           start_time: float, language: Optional[str] = None,
                           model_name: Optional[str] = None,
                           attempt_metadata: Optional[Dict[str, Any]] = None):
        """
        Log a fallback attempt.
        
        Args:
            session_id: Fallback session ID
            fallback_method: Method being attempted
            start_time: Start time of attempt
            language: Target language (if applicable)
            model_name: Model name (if applicable)
            attempt_metadata: Additional metadata for this attempt
        """
        try:
            log_data = {
                "session_id": session_id,
                "component": self.component_name,
                "fallback_method": fallback_method,
                "language": language,
                "model_name": model_name,
                "start_time": start_time,
                "timestamp": datetime.now().isoformat(),
                "attempt_metadata": attempt_metadata or {}
            }
            
            self.logger.info(
                f"Attempting fallback method '{fallback_method}' in {self.component_name}",
                extra={"fallback_attempt": log_data}
            )
            
        except Exception as e:
            logger.error(f"Failed to log fallback attempt: {e}")
    
    def log_fallback_success(self, session_id: str, fallback_method: str,
                           processing_time: float, performance_impact: Optional[float] = None,
                           result_metadata: Optional[Dict[str, Any]] = None,
                           resource_usage: Optional[Dict[str, Any]] = None):
        """
        Log successful fallback completion.
        
        Args:
            session_id: Fallback session ID
            fallback_method: Method that succeeded
            processing_time: Time taken for fallback
            performance_impact: Performance impact (0.0-1.0)
            result_metadata: Metadata about the result
            resource_usage: Resource usage information
        """
        try:
            # Create fallback log entry
            entry = FallbackLogEntry(
                timestamp=datetime.now(),
                component=self.component_name,
                operation="fallback_execution",
                fallback_trigger=session_id,
                fallback_method=fallback_method,
                fallback_success=True,
                processing_time=processing_time,
                performance_impact=performance_impact,
                resource_usage=resource_usage or {},
                metadata=result_metadata or {}
            )
            
            self.fallback_entries.append(entry)
            
            # Track performance metrics
            if fallback_method not in self.performance_metrics:
                self.performance_metrics[fallback_method] = []
            self.performance_metrics[fallback_method].append(processing_time)
            
            log_data = {
                "session_id": session_id,
                "component": self.component_name,
                "fallback_method": fallback_method,
                "processing_time": processing_time,
                "performance_impact": performance_impact,
                "resource_usage": resource_usage or {},
                "result_metadata": result_metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Fallback successful in {self.component_name}: {fallback_method} "
                f"completed in {processing_time:.3f}s",
                extra={"fallback_success": log_data}
            )
            
        except Exception as e:
            logger.error(f"Failed to log fallback success: {e}")
    
    def log_fallback_failure(self, session_id: str, fallback_method: str,
                           error: Exception, processing_time: float,
                           retry_possible: bool = False,
                           failure_metadata: Optional[Dict[str, Any]] = None):
        """
        Log fallback failure.
        
        Args:
            session_id: Fallback session ID
            fallback_method: Method that failed
            error: Error that occurred during fallback
            processing_time: Time taken before failure
            retry_possible: Whether retry is possible
            failure_metadata: Additional failure information
        """
        try:
            # Create fallback log entry
            entry = FallbackLogEntry(
                timestamp=datetime.now(),
                component=self.component_name,
                operation="fallback_execution",
                fallback_trigger=session_id,
                fallback_method=fallback_method,
                original_error=str(error),
                fallback_success=False,
                processing_time=processing_time,
                metadata=failure_metadata or {}
            )
            
            self.fallback_entries.append(entry)
            
            log_data = {
                "session_id": session_id,
                "component": self.component_name,
                "fallback_method": fallback_method,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "processing_time": processing_time,
                "retry_possible": retry_possible,
                "failure_metadata": failure_metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.error(
                f"Fallback failed in {self.component_name}: {fallback_method} "
                f"failed after {processing_time:.3f}s - {str(error)}",
                extra={"fallback_failure": log_data}
            )
            
        except Exception as e:
            logger.error(f"Failed to log fallback failure: {e}")
    
    def log_fallback_chain(self, session_id: str, fallback_chain: List[Dict[str, Any]],
                         final_success: bool, total_time: float):
        """
        Log a complete fallback chain execution.
        
        Args:
            session_id: Fallback session ID
            fallback_chain: List of fallback attempts with results
            final_success: Whether the chain ultimately succeeded
            total_time: Total time for entire chain
        """
        try:
            chain_summary = {
                "session_id": session_id,
                "component": self.component_name,
                "total_attempts": len(fallback_chain),
                "final_success": final_success,
                "total_time": total_time,
                "chain_details": fallback_chain,
                "timestamp": datetime.now().isoformat()
            }
            
            if final_success:
                self.logger.info(
                    f"Fallback chain completed successfully in {self.component_name}: "
                    f"{len(fallback_chain)} attempts in {total_time:.3f}s",
                    extra={"fallback_chain": chain_summary}
                )
            else:
                self.logger.error(
                    f"Fallback chain failed in {self.component_name}: "
                    f"{len(fallback_chain)} attempts failed in {total_time:.3f}s",
                    extra={"fallback_chain": chain_summary}
                )
                
        except Exception as e:
            logger.error(f"Failed to log fallback chain: {e}")
    
    def log_performance_degradation(self, operation: str, baseline_time: float,
                                  fallback_time: float, degradation_factor: float,
                                  impact_metadata: Optional[Dict[str, Any]] = None):
        """
        Log performance degradation due to fallback.
        
        Args:
            operation: Operation that experienced degradation
            baseline_time: Expected baseline processing time
            fallback_time: Actual fallback processing time
            degradation_factor: Performance degradation factor
            impact_metadata: Additional impact information
        """
        try:
            degradation_data = {
                "component": self.component_name,
                "operation": operation,
                "baseline_time": baseline_time,
                "fallback_time": fallback_time,
                "degradation_factor": degradation_factor,
                "performance_impact_percent": ((fallback_time - baseline_time) / baseline_time) * 100,
                "impact_metadata": impact_metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            if degradation_factor > 2.0:  # Significant degradation
                self.logger.warning(
                    f"Significant performance degradation in {self.component_name} for {operation}: "
                    f"{degradation_factor:.1f}x slower ({baseline_time:.3f}s -> {fallback_time:.3f}s)",
                    extra={"performance_degradation": degradation_data}
                )
            else:
                self.logger.info(
                    f"Performance impact in {self.component_name} for {operation}: "
                    f"{degradation_factor:.1f}x slower ({baseline_time:.3f}s -> {fallback_time:.3f}s)",
                    extra={"performance_degradation": degradation_data}
                )
                
        except Exception as e:
            logger.error(f"Failed to log performance degradation: {e}")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fallback usage.
        
        Returns:
            Dictionary containing fallback statistics
        """
        try:
            if not self.fallback_entries:
                return {
                    "component": self.component_name,
                    "total_fallbacks": 0,
                    "success_rate": 0.0,
                    "methods": {},
                    "performance_metrics": {}
                }
            
            total_fallbacks = len(self.fallback_entries)
            successful_fallbacks = sum(1 for entry in self.fallback_entries if entry.fallback_success)
            success_rate = successful_fallbacks / total_fallbacks
            
            # Analyze fallback methods
            method_stats = {}
            for entry in self.fallback_entries:
                method = entry.fallback_method
                if method not in method_stats:
                    method_stats[method] = {
                        "count": 0,
                        "successes": 0,
                        "avg_time": 0.0,
                        "total_time": 0.0
                    }
                
                method_stats[method]["count"] += 1
                method_stats[method]["total_time"] += entry.processing_time
                
                if entry.fallback_success:
                    method_stats[method]["successes"] += 1
            
            # Calculate averages
            for method, stats in method_stats.items():
                if stats["count"] > 0:
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    stats["success_rate"] = stats["successes"] / stats["count"]
            
            # Performance metrics summary
            perf_summary = {}
            for method, times in self.performance_metrics.items():
                if times:
                    perf_summary[method] = {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
            
            return {
                "component": self.component_name,
                "total_fallbacks": total_fallbacks,
                "successful_fallbacks": successful_fallbacks,
                "success_rate": success_rate,
                "method_statistics": method_stats,
                "performance_metrics": perf_summary,
                "recent_entries": len([e for e in self.fallback_entries if (datetime.now() - e.timestamp).seconds < 3600])
            }
            
        except Exception as e:
            logger.error(f"Failed to get fallback statistics: {e}")
            return {"error": str(e)}
    
    def export_fallback_logs(self, output_path: Optional[str] = None) -> str:
        """
        Export fallback logs to file for analysis.
        
        Args:
            output_path: Optional output file path
            
        Returns:
            Path to exported log file
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"fallback_logs_{self.component_name}_{timestamp}.json"
            
            export_data = {
                "component": self.component_name,
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(self.fallback_entries),
                "statistics": self.get_fallback_statistics(),
                "entries": [
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "operation": entry.operation,
                        "fallback_trigger": entry.fallback_trigger,
                        "fallback_method": entry.fallback_method,
                        "language": entry.language,
                        "model_name": entry.model_name,
                        "original_error": entry.original_error,
                        "fallback_success": entry.fallback_success,
                        "processing_time": entry.processing_time,
                        "performance_impact": entry.performance_impact,
                        "resource_usage": entry.resource_usage,
                        "metadata": entry.metadata
                    }
                    for entry in self.fallback_entries
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.fallback_entries)} fallback log entries to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export fallback logs: {e}")
            return ""
    
    def clear_fallback_logs(self):
        """Clear fallback log entries and performance metrics."""
        self.fallback_entries.clear()
        self.performance_metrics.clear()
        logger.info(f"Cleared fallback logs for {self.component_name}")


class LoggingManager:
    """
    Central logging manager for error handling and fallback scenarios.
    
    Coordinates logging across all components and provides centralized
    log analysis and reporting capabilities.
    """
    
    def __init__(self, settings=None):
        """
        Initialize the LoggingManager.
        
        Args:
            settings: Application settings object containing configuration
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        
        # Component loggers
        self.component_loggers: Dict[str, FallbackLogger] = {}
        
        # Global logging configuration
        self._configure_logging()
        
        logger.info("LoggingManager initialized with enhanced fallback logging")
    
    def get_component_logger(self, component_name: str) -> FallbackLogger:
        """
        Get or create a fallback logger for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            FallbackLogger instance for the component
        """
        if component_name not in self.component_loggers:
            self.component_loggers[component_name] = FallbackLogger(component_name, self.settings)
        
        return self.component_loggers[component_name]
    
    def _configure_logging(self):
        """Configure enhanced logging for fallback scenarios."""
        try:
            # Configure structured logging format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Ensure fallback loggers use appropriate formatting
            fallback_logger = logging.getLogger("fallback")
            fallback_logger.setLevel(logging.INFO)
            
            # Add handler if not already present
            if not fallback_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                fallback_logger.addHandler(handler)
                
        except Exception as e:
            logger.error(f"Failed to configure enhanced logging: {e}")
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global fallback statistics across all components.
        
        Returns:
            Dictionary containing global statistics
        """
        try:
            global_stats = {
                "total_components": len(self.component_loggers),
                "component_statistics": {},
                "overall_fallback_count": 0,
                "overall_success_rate": 0.0,
                "most_active_components": [],
                "most_common_fallback_methods": {}
            }
            
            total_fallbacks = 0
            total_successes = 0
            method_counts = {}
            
            for component_name, component_logger in self.component_loggers.items():
                component_stats = component_logger.get_fallback_statistics()
                global_stats["component_statistics"][component_name] = component_stats
                
                component_fallbacks = component_stats.get("total_fallbacks", 0)
                component_successes = component_stats.get("successful_fallbacks", 0)
                
                total_fallbacks += component_fallbacks
                total_successes += component_successes
                
                # Aggregate method statistics
                for method, stats in component_stats.get("method_statistics", {}).items():
                    if method not in method_counts:
                        method_counts[method] = 0
                    method_counts[method] += stats.get("count", 0)
            
            global_stats["overall_fallback_count"] = total_fallbacks
            global_stats["overall_success_rate"] = total_successes / total_fallbacks if total_fallbacks > 0 else 0.0
            
            # Most active components
            component_activity = [
                (name, stats.get("total_fallbacks", 0))
                for name, stats in global_stats["component_statistics"].items()
            ]
            global_stats["most_active_components"] = sorted(component_activity, key=lambda x: x[1], reverse=True)[:5]
            
            # Most common fallback methods
            global_stats["most_common_fallback_methods"] = dict(
                sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            return global_stats
            
        except Exception as e:
            logger.error(f"Failed to get global statistics: {e}")
            return {"error": str(e)}
    
    def export_all_logs(self, output_directory: Optional[str] = None) -> List[str]:
        """
        Export logs from all components.
        
        Args:
            output_directory: Optional output directory
            
        Returns:
            List of exported file paths
        """
        try:
            if output_directory is None:
                output_directory = "fallback_logs_export"
            
            Path(output_directory).mkdir(exist_ok=True)
            exported_files = []
            
            for component_name, component_logger in self.component_loggers.items():
                output_path = Path(output_directory) / f"{component_name}_fallback_logs.json"
                exported_path = component_logger.export_fallback_logs(str(output_path))
                if exported_path:
                    exported_files.append(exported_path)
            
            # Export global statistics
            global_stats_path = Path(output_directory) / "global_statistics.json"
            with open(global_stats_path, 'w') as f:
                json.dump(self.get_global_statistics(), f, indent=2)
            exported_files.append(str(global_stats_path))
            
            logger.info(f"Exported logs from {len(self.component_loggers)} components to {output_directory}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export all logs: {e}")
            return []