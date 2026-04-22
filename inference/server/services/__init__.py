"""
Service layer — business logic shared across all LLM interaction endpoints.

Services are stateless and composable.  Each service has a single clear
responsibility:

  • ``TokenService``       — token counting and context-window scaling
  • ``ToolService``        — server-tool separation and preparation
  • ``CompletionService``  — workflow execution, retry, continuation, nudge
"""

from .token_service import TokenService
from .tool_service import ToolService, PreparedTools
from .completion_service import (
    CompletionService,
    CompletionResult,
    StreamAccumulator,
)

__all__ = [
    "TokenService",
    "ToolService",
    "PreparedTools",
    "CompletionService",
    "CompletionResult",
    "StreamAccumulator",
]
