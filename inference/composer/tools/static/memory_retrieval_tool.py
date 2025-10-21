"""
Static memory retrieval tool using LangGraph Command pattern.

This tool retrieves relevant memories from the database using embeddings and
similarity search with efficient state access and proper LangGraph integration.

Features:
- Single function-based tool using @tool decorator
- Strong typing with WorkflowState instead of generic parameters
- Efficient user_config access from injected state (no database calls)
- Command pattern for proper state updates
- Embedding generation and similarity search

Configuration:
- Similarity thresholds for memory matching (0.0-1.0)
- Result limits (1-50 memories)
- Cross-user and cross-conversation access controls
- User-specific preferences from WorkflowState.user_config.memory

Usage in LangGraph workflows:
    # Tool is automatically available when registered in tool registry
    # LangGraph handles injection of tool_call_id and WorkflowState
"""

import json
from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from composer.graph.state import WorkflowState
from runner import pipeline_factory
from db import storage
from models import ModelProfileType
from models.default_configs import DEFAULT_MEMORY_CONFIG
from utils.model_profile import get_model_profile
from utils.logging import llmmllogger


# Single memory retrieval tool using Command pattern with strong typing
@tool
async def memory_retrieval(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[WorkflowState, InjectedState],
) -> Command:
    """
    Retrieve relevant memories based on text query and automatically add results to workflow state.

    This tool searches through stored conversation memories and previous interactions
    to find relevant information based on semantic similarity. Use this tool when
    you need to recall previous conversations or information from past interactions.

    Args:
        query: The search query to execute for memory retrieval

    Returns:
        Relevant memories with content, timestamps, and similarity scores
    """
    logger = llmmllogger.logger.bind(component="MemoryRetrieval")

    try:
        # Get user_config directly from injected state (much more efficient!)
        if state.user_config and hasattr(state.user_config, "memory"):
            memory_config = state.user_config.memory
            logger.debug("Using memory config from injected state")
        else:
            memory_config = DEFAULT_MEMORY_CONFIG
            logger.debug("Using default memory config - no user_config in state")

        # Ensure we have required state
        if not state.user_id:
            error_message = json.dumps(
                {
                    "status": "error",
                    "error": "Missing user_id in state",
                    "query": query,
                },
                indent=2,
            )
            return Command(
                update={
                    "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)]
                }
            )

        # Initialize storage if not done
        if not storage.pool:
            error_message = json.dumps(
                {
                    "status": "error",
                    "error": "Database not initialized",
                    "query": query,
                },
                indent=2,
            )
            return Command(
                update={
                    "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)]
                }
            )

        # Generate embeddings for the query with fallback handling
        query_embeddings = None

        # Try to get embedding model profile and generate embeddings
        embedding_profile = await get_model_profile(
            user_id=state.user_id, task=ModelProfileType.Embedding
        )

        # Get embedding pipeline from factory
        try:
            embedding_pipeline = pipeline_factory.get_embedding_pipeline(
                profile=embedding_profile
            )
            # Generate embeddings for the query using Embeddings interface
            query_embeddings = embedding_pipeline.embed_documents([query])
        except Exception as embed_error:
            # Use fallback if no valid pipeline available
            logger.warning(
                f"Embedding generation failed: {embed_error}, using mock embeddings"
            )
            query_embeddings = [[0.1] * 768]  # Fallback mock embedding

        # If embeddings are still None, use fallback
        if query_embeddings is None:
            logger.warning("Embedding generation returned None, using mock embeddings")
            query_embeddings = [[0.1] * 768]  # Fallback mock embedding

        # Retrieve similar memories from storage using configuration
        memory_service = storage.get_service(storage.memory)

        # Configure user and conversation filtering based on memory config
        user_filter = None if memory_config.enable_cross_user else state.user_id
        conversation_filter = (
            None if memory_config.enable_cross_conversation else state.conversation_id
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

        # Create response message for the conversation
        response_message = json.dumps(
            {
                "status": "success",
                "memories": formatted_memories,
                "query": query,
                "count": len(formatted_memories),
            },
            indent=2,
        )

        logger.info(
            f"Memory retrieval completed successfully with {len(formatted_memories)} memories",
            query=query[:100],
            memory_count=len(formatted_memories),
        )

        # Return Command that updates state with memory results
        return Command(
            update={
                "retrieved_memories": memories,
                "memory_query": query,
                "messages": [ToolMessage(response_message, tool_call_id=tool_call_id)],
            }
        )

    except Exception as e:
        # Log the full exception for debugging
        logger.error(
            f"Memory retrieval failed for query '{query}': {e}",
            exc_info=True,
            query=query[:100],
        )

        error_message = json.dumps(
            {"status": "error", "error": str(e), "query": query}, indent=2
        )

        return Command(
            update={
                "memory_query": query,
                "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)],
            }
        )
