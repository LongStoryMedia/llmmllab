"""
Specialized summarization nodes for LangGraph workflows.

This module provides discrete, specialized nodes for different types of
summarization operations. Workflows should use these nodes directly or
through the factory function for optimal architecture.
"""

from .search_summary_node import SearchSummaryNode
from .conversation import ConsolidationNode

__all__ = [
    # Discrete specialized nodes
    "SearchSummaryNode",
    "ConsolidationNode",
]
