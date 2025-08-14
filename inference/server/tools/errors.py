"""
Centralized error handling for the tools module.

This module defines custom exceptions and utility functions for handling 
errors consistently across the tools system.
"""

from typing import Any, Dict, Optional, Type, Union
import logging
import traceback

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Base exception for all tool-related errors"""
    pass


class ToolExecutionError(ToolError):
    """Exception raised for errors during tool execution"""
    pass


class ToolValidationError(ToolError):
    """Exception raised when tool validation fails"""
    pass


class ToolCreationError(ToolError):
    """Exception raised when tool creation fails"""
    pass


class ToolRegistryError(ToolError):
    """Exception raised for errors in the tool registry"""
    pass


def log_error(exc: Exception, context: Optional[Dict[str, Any]] = None, 
              level: str = "error") -> None:
    """Log an error with optional context information
    
    Args:
        exc: The exception that occurred
        context: Optional dictionary with context about the error
        level: Log level to use (default: "error")
    """
    log_func = getattr(logger, level.lower())
    
    message = f"{exc.__class__.__name__}: {str(exc)}"
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message = f"{message} [Context: {context_str}]"
        
    log_func(message)
    
    if level in ("error", "critical"):
        logger.debug(f"Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")


def handle_error(exc: Exception, 
                 error_type: Optional[Type[ToolError]] = None,
                 message: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 log_level: str = "error",
                 raise_error: bool = True) -> None:
    """Handle an exception by logging it and optionally re-raising
    
    Args:
        exc: The exception that occurred
        error_type: Optional custom error type to raise
        message: Optional custom error message
        context: Optional dictionary with context about the error
        log_level: Log level to use (default: "error")
        raise_error: Whether to raise a new error (default: True)
    
    Raises:
        ToolError: If raise_error is True
    """
    log_error(exc, context, log_level)
    
    if raise_error:
        if error_type is None:
            error_type = ToolExecutionError
            
        if message is None:
            message = str(exc)
            
        raise error_type(message) from exc
