"""Static composer tools with consistent behavior."""

from .web_search_tool import WebSearchTool
from .memory_retrieval_tool import MemoryRetrievalTool
from .summarization_tool import SummarizationTool

# Also import from rag_tools for backward compatibility
from .rag_tools import WebSearchTool as RagWebSearchTool, MemoryRetrievalTool as RagMemoryRetrievalTool, SummarizationTool as RagSummarizationTool

__all__ = [
    "WebSearchTool",
    "MemoryRetrievalTool", 
    "SummarizationTool",
]
