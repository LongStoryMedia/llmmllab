"""
Memory Search Node for LangGraph workflows.
Searches for similar memories using embeddings.
"""

from typing import TYPE_CHECKING
from graph.state import WorkflowState
from agents.embed import EmbeddingAgent
from utils import extract_text_from_message
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from db.memory_storage import MemoryStorage


class MemorySearchNode:
    """
    Node for searching memories relevant to the current user query by embedding similarity.

    Takes query embeddings from workflow state and searches for
    similar memories using the memory agent.
    """

    def __init__(
        self,
        embedding_agent: "EmbeddingAgent",
        memory_storage: "MemoryStorage",
    ):
        """Initialize memory search node with dependency injection.

        Args:
            embedding_agent: Required EmbeddingAgent instance
            memory_storage: Required MemoryStorage instance
        """
        self.memory_storage = memory_storage
        self.embedding_agent = embedding_agent
        self.logger = llmmllogger.logger.bind(component="MemorySearchNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Search for memories similar to query embedding.

        Args:
            state: Current workflow state with query_embedding

        Returns:
            Updated workflow state with retrieved_memories
        """
        try:
            assert state.user_id
            assert state.conversation_id
            assert state.user_config
            assert state.user_config.memory
            assert state.current_user_message

            user_id = state.user_id
            conversation_id = state.conversation_id
            max_results = state.user_config.memory.limit
            similarity_threshold = state.user_config.memory.similarity_threshold
            enable_cross_conversation = (
                state.user_config.memory.enable_cross_conversation
            )
            enable_cross_user = state.user_config.memory.enable_cross_user

            # Extract message text and generate embeddings using injected EmbeddingAgent
            message = state.current_user_message
            message_text = extract_text_from_message(message)

            # Use injected EmbeddingAgent to generate embeddings
            embeddings = await self.embedding_agent.embed([message_text])

            if not embeddings:
                # Gracefully skip memory search if embeddings fail (instead of raising)
                self.logger.warning(
                    "Embedding generation failed, skipping memory search",
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                state.retrieved_memories = []
                return state

            self.logger.info(
                "Searching for similar memories",
                user_id=user_id,
                conversation_id=conversation_id,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
            )

            # Use storage layer for similarity search with conversation filtering
            memories = await self.memory_storage.search_similarity(
                embeddings=embeddings,
                min_similarity=similarity_threshold,
                limit=max_results,
                user_id=(user_id if not enable_cross_user else None),
                conversation_id=(
                    int(conversation_id) if not enable_cross_conversation else None
                ),
                start_date=None,
                end_date=None,
            )

            # Store retrieved memories in state
            state.retrieved_memories = memories

            self.logger.info(
                "Memory search completed",
                user_id=user_id,
                memories_found=len(memories),
                has_context=len(memories) > 0,
            )

            return state

        except Exception as e:
            self.logger.error(
                "Memory search failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            # Gracefully continue with empty memories instead of crashing workflow
            state.retrieved_memories = []
            return state
