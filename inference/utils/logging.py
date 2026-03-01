"""
Structured logging for composer service.
Follows inference service logging patterns.
"""

import logging
from datetime import datetime
import os
import sys
from typing import Dict, Any, Optional
import json
from pydantic import BaseModel
import structlog
import structlog.typing
import structlog.stdlib
import structlog.processors


def serialize_event_data(
    data: Any, max_depth: int = 10, current_depth: int = 0, indent: int = 2
) -> str:
    """
    Recursively serialize event data for logging/debugging.
    Handles nested BaseModel objects, dicts, lists, and other complex structures.

    Args:
        data: The data to serialize
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        indent: Number of spaces to use for JSON indentation

    Returns:
        Formatted JSON string with proper indentation and multiple lines
    """

    def _serialize_recursive(obj: Any, depth: int = 0) -> Any:
        """Internal recursive function that returns serializable data."""
        if depth >= max_depth:
            return f"<max_depth_reached:{type(obj).__name__}>"

        if isinstance(obj, BaseModel):
            try:
                # For BaseModel objects, get the dict and recursively process it
                # Use mode='json' to ensure enums are properly serialized as string values
                model_dict = obj.model_dump(exclude_none=True, mode="json")
                return _serialize_recursive(model_dict, depth + 1)
            except Exception as e:
                return f"<BaseModel_error:{str(e)}>"

        elif isinstance(obj, dict):
            serialized_dict = {}
            for k, v in obj.items():
                try:
                    serialized_dict[str(k)] = _serialize_recursive(v, depth + 1)
                except Exception as e:
                    serialized_dict[str(k)] = f"<dict_value_error:{str(e)}>"
            return serialized_dict

        elif isinstance(obj, (list, tuple, set)):
            try:
                return [_serialize_recursive(item, depth + 1) for item in obj]
            except Exception as e:
                return f"<list_error:{str(e)}>"

        elif hasattr(obj, "__dict__"):
            # Handle objects with __dict__ attribute
            try:
                obj_dict = {
                    k: _serialize_recursive(v, depth + 1)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
                return obj_dict
            except Exception as e:
                return f"<object_error:{str(e)}>"

        elif callable(obj):
            return (
                f"<callable:{obj.__name__ if hasattr(obj, '__name__') else 'unknown'}>"
            )

        else:
            # For primitive types and other objects, convert to string
            try:
                # Try to JSON serialize first to check if it's already serializable
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    # Serialize the data structure
    serialized_data = _serialize_recursive(data, current_depth)

    # Convert to formatted JSON string
    try:
        return json.dumps(
            serialized_data,
            indent=indent,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "),
        )
    except Exception as e:
        # Fallback to string representation if JSON serialization fails
        return f"<json_serialization_error: {str(e)}>\nFallback representation:\n{str(serialized_data)}"


class LlmmlLogger:
    """Structured logging with colorized output for both direct execution and Kubernetes logs."""

    def __init__(self, service_name: str = "llmmllab"):
        # Set up logging
        log_level = os.environ.get("LOG_LEVEL", "info").lower()
        log_level_map = {
            "trace": logging.DEBUG,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        logging_level = log_level_map.get(log_level, "info")

        # Check if we should force colors (useful for Kubernetes logs)
        force_colors = os.environ.get("FORCE_COLOR", "0") == "1"

        # Determine if we should use colors
        # Force colors if FORCE_COLOR is set, or if we're in a TTY
        use_colors = force_colors or (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )

        # Configure structured logging with enhanced processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            # structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        # Add colorized console renderer if colors are enabled
        if use_colors:
            processors.append(
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.RichTracebackFormatter(
                        color_system="truecolor", show_locals=False
                    ),
                )
            )
        else:
            processors.append(
                structlog.dev.ConsoleRenderer(
                    colors=False,
                    exception_formatter=structlog.dev.RichTracebackFormatter(
                        color_system=None, show_locals=False
                    ),
                )
            )

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(logging_level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            level=logging_level,
            stream=sys.stdout,
        )

        self.logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(
            service_name
        )

        self.logger.info(
            "Logger initialized",
            service=service_name,
            log_lvl=log_level,
            colors=use_colors,
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
    ):
        """Log intent analysis results."""
        self.logger.debug(
            "Intent analysis completed",
            intent_result=intent_result,
            confidence=confidence,
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
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
        }
        if context:
            error_context.update(context)

        self.logger.error("Composer error occurred", extra=error_context, exc_info=True)

    def bind(self, **kwargs) -> structlog.typing.FilteringBoundLogger:
        """Create a new logger with additional bound context."""
        return self.logger.bind(**kwargs)


# Global logger instance
llmmllogger = LlmmlLogger()
