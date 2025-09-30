"""
Structured logging for composer service.
Follows inference service logging patterns.
"""

import structlog
import structlog.typing
from datetime import datetime
from typing import Dict, Any, Optional


class ComposerLogger:
    """Structured logging for composer workflows."""

    def __init__(self, service_name: str = "composer"):
        self.logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(
            service_name
        )

    def log_workflow_start(
        self,
        workflow_id: str,
        workflow_type: str,
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Log workflow start event."""
        context = {
            "event": "workflow_started",
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }
        if additional_context:
            context.update(additional_context)

        self.logger.info("Workflow started", **context)

    def log_workflow_complete(
        self,
        workflow_id: str,
        duration_ms: float,
        success: bool = True,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Log workflow completion."""
        context = {
            "event": "workflow_completed",
            "workflow_id": workflow_id,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        if additional_context:
            context.update(additional_context)

        level_fn = self.logger.info if success else self.logger.error
        level_fn("Workflow completed", **context)

    def log_node_execution(
        self,
        node_name: str,
        duration_ms: float,
        success: bool = True,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Log individual node execution."""
        context = {
            "event": "node_executed",
            "node_name": node_name,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        if additional_context:
            context.update(additional_context)

        self.logger.debug("Node executed", **context)

    def log_tool_generation(
        self,
        tool_spec: str,
        method: str,  # "existing", "modified", "new"
        success: bool = True,
        tool_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Log tool generation or retrieval."""
        context = {
            "event": "tool_generation",
            "tool_spec": tool_spec,
            "method": method,
            "success": success,
            "tool_id": tool_id,
            "timestamp": datetime.now().isoformat(),
        }
        if additional_context:
            context.update(additional_context)

        level_fn = self.logger.info if success else self.logger.warning
        level_fn("Tool generation", **context)

    def log_intent_analysis(
        self,
        intent_result: Dict[str, Any],
        confidence: float,
        processing_time_ms: float,
    ):
        """Log intent analysis results."""
        self.logger.debug(
            "Intent analysis completed",
            intent_result=intent_result,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now().isoformat(),
        )

    def log_cache_operation(
        self,
        operation: str,  # "hit", "miss", "set", "evict"
        cache_key: str,
        success: bool = True,
    ):
        """Log workflow cache operations."""
        self.logger.debug(
            f"Cache operation: {operation}",
            operation=operation,
            cache_key=cache_key,
            success=success,
            timestamp=datetime.now().isoformat(),
        )

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log errors with structured context."""
        error_context = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
        }
        if context:
            error_context.update(context)

        self.logger.error("Composer error", **error_context, exc_info=True)


# Global logger instance
composer_logger = ComposerLogger()
