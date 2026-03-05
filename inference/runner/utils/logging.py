"""
Structured logging for runner service.
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
    """

    def _serialize_recursive(obj: Any, depth: int = 0) -> Any:
        """Internal recursive function that returns serializable data."""
        if depth >= max_depth:
            return f"<max_depth_reached:{type(obj).__name__}>"

        if isinstance(obj, BaseModel):
            try:
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
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    serialized_data = _serialize_recursive(data, current_depth)

    try:
        return json.dumps(
            serialized_data,
            indent=indent,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "),
        )
    except Exception as e:
        return f"<json_serialization_error: {str(e)}>\nFallback representation:\n{str(serialized_data)}"


class LlmmlLogger:
    """Structured logging with colorized output for both direct execution and Kubernetes logs."""

    def __init__(self, service_name: str = "llmmllab"):
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

        force_colors = os.environ.get("FORCE_COLOR", "0") == "1"
        use_colors = force_colors or (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

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

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(logging_level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

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

    def bind(self, **kwargs) -> structlog.typing.FilteringBoundLogger:
        """Create a new logger with additional bound context."""
        return self.logger.bind(**kwargs)


# Global logger instance
llmmllogger = LlmmlLogger()