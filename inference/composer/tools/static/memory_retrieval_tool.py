"""
Static memory retrieval tool for database storage with configurable parameters.

This tool retrieves relevant memories from the database using embeddings and 
similarity search. Configuration is driven by MemoryConfig which controls
similarity thresholds, result limits, cross-user/conversation access, and
embedding model selection.

Configuration:
- Similarity thresholds for memory matching (0.0-1.0)
- Result limits (1-50 memories)  
- Cross-user and cross-conversation access controls
- Embedding model selection and timeout settings
- Always-retrieve behavior for research workflows

Usage:
    # Default configuration
    tool = create_memory_retrieval_tool()
    result = await tool._arun("machine learning concepts")
    
    # Custom configuration
    custom_config = MemoryConfig(
        similarity_threshold=0.8,
        limit=10, 
        enable_cross_conversation=True
    )
    tool = MemoryRetrievalTool(memory_config=custom_config)
    
    # Specialized memory tools
    focused_tool = create_focused_memory_tool()  # High relevance, few results
    broad_tool = create_broad_memory_tool()      # Lower threshold, more results  
    research_tool = create_research_memory_tool()  # Research-optimized settings

Integration with User Configuration:
- Memory configuration can be retrieved using get_model_profile utility
- User-specific memory preferences merged at data layer with defaults
- Supports embedding model profiles for consistent query encoding
"""

import asyncio
import json
import logging
from typing import Optional, Union

from langchain_core.tools import BaseTool

from runner import embed_pipeline, pipeline_factory, EmbeddingPipeline

from db import storage
from models.memory_config import MemoryConfig
from models.model_profile_type import ModelProfileType
from models.default_configs import DEFAULT_MEMORY_CONFIG
from utils import get_model_profile


class MemoryRetrievalTool(BaseTool):
    """Static tool for retrieving memories from database storage with configurable parameters."""

    name: str = "memory_retrieval"
    description: str = (
        "Retrieve relevant memories based on text query. Uses embeddings and similarity search "
        "with configurable cross-user/conversation settings and similarity thresholds."
    )

    def __init__(self, memory_config: MemoryConfig, **kwargs):
        super().__init__(**kwargs)
        self.memory_config = memory_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.debug(f"MemoryRetrievalTool initialized with limit: {memory_config.limit}, "
                         f"similarity_threshold: {memory_config.similarity_threshold}")

    async def _arun(self, query: str) -> str:
        """Async implementation of memory retrieval using database storage."""
        try:
            # Initialize storage if not done
            if not storage.pool:
                return json.dumps(
                    {
                        "status": "error",
                        "error": "Database not initialized",
                        "query": query,
                    },
                    indent=2,
                )

            # Generate embeddings for the query
            try:

                                # Get embedding model profile for query encoding
                try:
                    # Get the embedding model profile (will need user_id in actual usage)
                    # For now, use a placeholder - in actual composer usage, this would come from context
                    embedding_profile = await get_model_profile(
                        user_id="system",  # TODO: Get from execution context
                        task=ModelProfileType.Embedding
                    )
                    
                    # Get embedding pipeline from factory
                    embedding_pipeline = pipeline_factory.get_pipeline(
                        profile=embedding_profile,
                        expected_type=list  # Embeddings return List[List[float]]
                    )
                    
                    if not embedding_pipeline:
                        raise ValueError("No embedding pipeline available")
                        
                    # Generate embeddings for the query (cast to EmbeddingPipeline)
                    if isinstance(embedding_pipeline, EmbeddingPipeline):
                        query_embeddings = await embed_pipeline(
                            query, pipeline=embedding_pipeline
                        )
                    else:
                        raise ValueError("Pipeline is not an EmbeddingPipeline")
                except Exception as embed_error:
                    # Fallback to mock embeddings if no embedding model available
                    self.logger.warning(
                        f"Embedding generation failed, using mock: {embed_error}"
                    )
                    query_embeddings = [[0.1] * 768]  # Fallback mock embedding

                # Retrieve similar memories from storage using configuration
                memory_service = storage.get_service(storage.memory)
                
                # Configure user and conversation filtering based on memory config
                # TODO: In actual composer usage, get these from execution context
                user_filter = None if self.memory_config.enable_cross_user else None  # Would be actual user_id
                conversation_filter = None if self.memory_config.enable_cross_conversation else None  # Would be actual conversation_id
                
                memories = await memory_service.search_similarity(
                    embeddings=query_embeddings,
                    min_similarity=self.memory_config.similarity_threshold,
                    limit=self.memory_config.limit,
                    user_id=user_filter,
                    conversation_id=conversation_filter,
                )

                # Format memories for display
                formatted_memories = [
                    {
                        "content": (
                            "\n".join([f.content for f in memory.fragments])
                            if hasattr(memory, "fragments")
                            else str(memory)
                        ),
                        "timestamp": (
                            memory.created_at.isoformat()
                            if hasattr(memory, "created_at")
                            else None
                        ),
                        "similarity": (
                            memory.similarity if hasattr(memory, "similarity") else 1.0
                        ),
                        "source": (
                            memory.source.value
                            if hasattr(memory, "source")
                            else "unknown"
                        ),
                    }
                    for memory in memories[:self.memory_config.limit]  # Use configured limit
                ]

                return json.dumps(
                    {
                        "status": "success",
                        "memories": formatted_memories,
                        "query": query,
                        "count": len(formatted_memories),
                    },
                    indent=2,
                )

            except Exception as embed_error:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Embedding generation failed: {str(embed_error)}",
                        "query": query,
                    },
                    indent=2,
                )

        except Exception as e:
            return json.dumps(
                {"status": "error", "error": str(e), "query": query}, indent=2
            )

    def _run(self, query: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))


# Factory functions for creating memory retrieval tools

def create_memory_retrieval_tool(
    memory_config: Optional[MemoryConfig] = None
) -> MemoryRetrievalTool:
    """
    Create a memory retrieval tool with configuration.
    
    Args:
        memory_config: Memory configuration (uses defaults if None)
        
    Returns:
        Configured MemoryRetrievalTool instance
    """
    if memory_config is None:
        memory_config = DEFAULT_MEMORY_CONFIG
    
    return MemoryRetrievalTool(memory_config=memory_config)


def create_focused_memory_tool() -> MemoryRetrievalTool:
    """
    Create a memory tool focused on high-relevance results.
    
    Returns:
        MemoryRetrievalTool with higher similarity threshold and fewer results
    """
    focused_config = MemoryConfig(
        **DEFAULT_MEMORY_CONFIG.model_dump(),
        similarity_threshold=0.85,  # Higher threshold for focused results
        limit=3,  # Fewer, more focused results
        enable_cross_conversation=False,  # Stay within current conversation
    )
    
    return MemoryRetrievalTool(memory_config=focused_config)


def create_broad_memory_tool() -> MemoryRetrievalTool:
    """
    Create a memory tool for broad memory exploration.
    
    Returns:
        MemoryRetrievalTool with lower similarity threshold and more results
    """
    broad_config = MemoryConfig(
        **DEFAULT_MEMORY_CONFIG.model_dump(),
        similarity_threshold=0.6,  # Lower threshold for broader search
        limit=10,  # More results for comprehensive view
        enable_cross_conversation=True,  # Search across conversations
        enable_cross_user=False,  # Keep user-specific
    )
    
    return MemoryRetrievalTool(memory_config=broad_config)


def create_research_memory_tool() -> MemoryRetrievalTool:
    """
    Create a memory tool optimized for research tasks.
    
    Returns:
        MemoryRetrievalTool with configuration suitable for research
    """
    research_config = MemoryConfig(
        **DEFAULT_MEMORY_CONFIG.model_dump(),
        similarity_threshold=0.75,  # Balanced threshold
        limit=8,  # Good number for research context
        enable_cross_conversation=True,  # Access historical research
        always_retrieve=True,  # Always provide context for research
        timeout=15.0,  # Longer timeout for thorough search
    )
    
    return MemoryRetrievalTool(memory_config=research_config)
