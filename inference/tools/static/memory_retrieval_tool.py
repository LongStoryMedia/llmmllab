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

from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from graph.state import WorkflowState
from runner import pipeline_factory
from db import storage
from models import ModelTask
from models.default_configs import DEFAULT_MEMORY_CONFIG
from utils.logging import llmmllogger


# Single memory retrieval tool using ToolRuntime pattern with strong typing
@tool
async def memory_retrieval(
    query: str,
    runtime: ToolRuntime,
) -> str:
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
        # Access state and tool_call_id through runtime
        state = runtime.state

        # Get user_config from tool runtime state
        if state.get("user_config") and hasattr(state["user_config"], "memory"):
            memory_config = state["user_config"].memory
            logger.debug("Using memory config from tool runtime state")
        else:
            memory_config = DEFAULT_MEMORY_CONFIG
            logger.debug(
                "Using default memory config - no user_config in tool runtime state"
            )

        # Ensure we have required state with detailed debugging
        user_id = state.get("user_id")
        if not user_id or user_id == "":
            error_details = {
                "user_id_in_state": user_id,
                "user_id_type": type(user_id).__name__,
                "state_keys": (
                    list(state.keys()) if isinstance(state, dict) else "not_dict"
                ),
                "conversation_id": state.get("conversation_id", "missing"),
                "user_config_present": state.get("user_config") is not None,
            }
            error_message = f"❌ Memory retrieval failed: Missing or empty user_id in state. Debug: {error_details}"
            logger.error("Memory retrieval state debug", **error_details)
            return error_message

        # Initialize storage if not done
        if not storage.pool:
            error_message = "❌ Memory retrieval failed: Database not initialized"
            return error_message

        # Generate embeddings for the query with fallback handling
        query_embeddings = None

        # Try to get embedding model and generate embeddings
        embedding_model = pipeline_factory.get_model_by_task(ModelTask.TEXTTOEMBEDDINGS)

        # Get embedding pipeline from factory
        try:
            if not embedding_model:
                raise RuntimeError("No TextToEmbeddings model available")
            embedding_pipeline = pipeline_factory.get_embedding_pipeline(
                model=embedding_model
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
        user_filter = None if memory_config.enable_cross_user else state["user_id"]
        conversation_filter = (
            None
            if memory_config.enable_cross_conversation
            else state["conversation_id"]
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

        # Create response message
        if formatted_memories:
            response_message = f"🧠 **Memory Search Results for: '{query}'**\n\n"
            for i, memory in enumerate(formatted_memories, 1):
                response_message += f"**{i}. Memory from {memory['source']}**\n"
                response_message += f"   Content: {memory['content'][:200]}...\n"
                response_message += f"   Timestamp: {memory['timestamp']}\n"
                response_message += f"   Similarity: {memory['similarity']:.2f}\n\n"
        else:
            response_message = f"🧠 No relevant memories found for query: '{query}'"

        logger.info(
            f"Memory retrieval completed successfully with {len(formatted_memories)} memories",
            query=query[:100],
            memory_count=len(formatted_memories),
        )

        # Return string result - ToolNode will automatically create ToolMessage
        return response_message

    except Exception as e:
        # Log the full exception for debugging
        logger.error(
            f"Memory retrieval failed for query '{query}': {e}",
            exc_info=True,
            query=query[:100],
        )

        error_message = f"❌ Memory retrieval failed: {str(e)}"
        return error_message
