"""
Compatibility aliases for native composer RAG tools.
"""

from .rag_tools import (
    PipelineInterface,
    SearchProviderInterface, 
    MemoryStoreInterface,
    WebSearchTool,
    MemoryRetrievalTool,
    SummarizationTool,
    RAGToolFactory
)

# Aliases for backward compatibility
ComposerWebSearchTool = WebSearchTool
ComposerMemoryTool = MemoryRetrievalTool
ComposerSummarizationTool = SummarizationTool
ComposerToolFactory = RAGToolFactory
