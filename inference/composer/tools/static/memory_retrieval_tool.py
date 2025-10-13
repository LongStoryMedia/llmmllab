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
    # Create tool using user_id - configuration retrieved from data layer
    tool = create_memory_retrieval_tool(user_id="user_123")
    result = await tool._arun("machine learning concepts")

    # Direct instantiation
    tool = MemoryRetrievalTool(user_id="user_123")
    result = await tool._arun("search query")

User Configuration Integration:
- Configuration retrieved from shared data layer via storage.user_config.get_user_config(user_id)
- User-specific memory preferences merged with system defaults at data layer
- Uses actual user_id for embedding model profile retrieval
- Ensures user preferences are always respected for similarity thresholds, limits, etc.
- Proper user and conversation filtering based on user's privacy settings
"""

import asyncio
import json
import logging

from langchain_core.tools import BaseTool

from runner import embed_pipeline, pipeline_factory, EmbeddingPipeline

from db import storage
from models import MemoryConfig, ModelProfileType
from models.default_configs import DEFAULT_MEMORY_CONFIG
from utils.model_profile import get_model_profile

from utils.logging import llmmllogger


class MemoryRetrievalTool(BaseTool):
    """Static tool for retrieving memories from database storage with configurable parameters."""

    name: str = "memory_retrieval"
    description: str = (
        "Retrieve relevant memories based on text query. Uses embeddings and similarity search "
        "with configurable cross-user/conversation settings and similarity thresholds."
    )

    # Declare fields as proper Pydantic fields
    user_id: str
    conversation_id: int

    def __init__(self, user_id: str, conversation_id: int, **kwargs):
        super().__init__(user_id=user_id, conversation_id=conversation_id, **kwargs)

    @property
    def logger(self):
        """Get logger for this tool instance."""
        return llmmllogger.logger.bind(component=self.__class__.__name__)

    async def _get_memory_config(self) -> MemoryConfig:
        """Get memory configuration from user config via shared data layer."""
        try:
            # Get complete user config with defaults merged at data layer
            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(self.user_id)
            if not user_config:
                self.logger.warning(
                    f"No user config found for {self.user_id}, using defaults"
                )
                return DEFAULT_MEMORY_CONFIG
            return user_config.memory
        except Exception as e:
            self.logger.error(
                f"Failed to get user config for {self.user_id}: {e}, using defaults"
            )
            return DEFAULT_MEMORY_CONFIG

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

            # Get memory configuration from user config
            memory_config = await self._get_memory_config()

            # Generate embeddings for the query with fallback handling
            query_embeddings = None

            # Try to get embedding model profile and generate embeddings
            embedding_profile = await get_model_profile(
                user_id=self.user_id, task=ModelProfileType.Embedding
            )

            # Get embedding pipeline from factory
            embedding_pipeline = pipeline_factory.get_pipeline(
                profile=embedding_profile,
                expected_type=list,  # Embeddings return List[List[float]]
            )

            if embedding_pipeline and isinstance(embedding_pipeline, EmbeddingPipeline):
                # Generate embeddings for the query
                query_embeddings = await embed_pipeline(
                    query, pipeline=embedding_pipeline
                )
            else:
                # Use fallback if no valid pipeline available
                self.logger.warning(
                    "No valid embedding pipeline available, using mock embeddings"
                )
                query_embeddings = [[0.1] * 768]  # Fallback mock embedding

            # If embeddings are still None, use fallback
            if query_embeddings is None:
                self.logger.warning(
                    "Embedding generation returned None, using mock embeddings"
                )
                query_embeddings = [[0.1] * 768]  # Fallback mock embedding

            # Retrieve similar memories from storage using configuration
            memory_service = storage.get_service(storage.memory)

            # Configure user and conversation filtering based on memory config
            user_filter = None if memory_config.enable_cross_user else self.user_id
            conversation_filter = (
                None
                if memory_config.enable_cross_conversation
                else getattr(self, "conversation_id", None)
            )

            memories = await memory_service.search_similarity(
                embeddings=query_embeddings,
                min_similarity=memory_config.similarity_threshold,
                limit=memory_config.limit,
                user_id=user_filter,
                conversation_id=conversation_filter,
            )

            # Format memories for display
            formatted_memories = [
                {
                    "content": (
                        "\n".join([f.content for f in memory.fragments if f.content])
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
                        memory.source.value if hasattr(memory, "source") else "unknown"
                    ),
                }
                for memory in memories[: memory_config.limit]  # Use configured limit
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

        except Exception as e:
            # Log the full exception for debugging
            self.logger.error(
                f"Memory retrieval failed for query '{query}': {e}", exc_info=True
            )
            return json.dumps(
                {"status": "error", "error": str(e), "query": query}, indent=2
            )

    def _run(self, query: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))


# Factory functions for creating memory retrieval tools


def create_memory_retrieval_tool(
    user_id: str, conversation_id: int
) -> MemoryRetrievalTool:
    """
    Create memory retrieval tool for user with configuration from shared data layer.

    Args:
        user_id: User identifier for configuration retrieval
        conversation_id: Optional conversation identifier for context filtering

    Returns:
        Configured MemoryRetrievalTool instance
    """
    return MemoryRetrievalTool(user_id=user_id, conversation_id=conversation_id)


# Note: Specialized memory behavior should be configured through user preferences
# in the user_config.memory settings rather than factory function overrides.
# This ensures user preferences are always respected and configuration is centralized.
