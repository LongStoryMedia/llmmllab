"""
Composer middleware components for agent efficiency optimizations.

This module contains custom middleware for LangGraph agents, focusing on:
- Vision processing optimization
- Context management
- Performance improvements
"""

from .vision_summarization import VisionSummarizationMiddleware

__all__ = [
    "VisionSummarizationMiddleware",
]