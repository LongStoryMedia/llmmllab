"""
Error definitions and handling for composer service.
"""

from typing import Optional


class ComposerError(Exception):
    """Base exception for composer errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class WorkflowConstructionError(ComposerError):
    """Failed to construct workflow."""

    pass


class NodeExecutionError(ComposerError):
    """Node execution failed."""

    def __init__(self, node_name: str, original_error: Optional[Exception] = None):
        self.node_name = node_name
        self.original_error = original_error

        if original_error:
            message = f"Node '{node_name}' failed: {original_error}"
            details = {
                "node_name": node_name,
                "original_error": str(original_error),
            }
        else:
            message = f"Node '{node_name}' failed"
            details = {"node_name": node_name}

        super().__init__(message, details)


class ToolGenerationError(ComposerError):
    """Failed to generate dynamic tool."""

    pass


class CircuitOpenError(ComposerError):
    """Circuit breaker is open."""

    pass


class IntentAnalysisError(ComposerError):
    """Intent analysis failed."""

    pass


class StateManagementError(ComposerError):
    """State management operation failed."""

    pass


class StreamingError(ComposerError):
    """Streaming operation failed."""

    pass
