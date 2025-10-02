"""
RAG (Retrieval-Augmented Generation) nodes for composer workflows.
DEPRECATED: Use composer.nodes.search instead for new modern agentic search architecture.
"""

# Legacy RAG router components from original router.py
from .router import RAGRouter, ShallowRAGExecutor, DeepRAGExecutor

# Legacy RAG executor components remain available
from .executor import RAGExecutorNode, EnhancedRAGExecutor

__all__ = [
    "RAGRouter",
    "ShallowRAGExecutor",
    "DeepRAGExecutor", 
    "RAGExecutorNode",
    "EnhancedRAGExecutor",
]
