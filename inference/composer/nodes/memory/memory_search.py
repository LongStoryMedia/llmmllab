"""
Memory Search Node for LangGraph workflows.
Searches for similar memories using embeddings.
"""

from composer.agents.memory_agent import MemoryAgent
from composer.graph.state import WorkflowState
from composer.utils.conversion import langchain_message_to_message
from composer.agents.embedding_agent import EmbeddingAgent
from utils.message import extract_message_text
from utils.logging import llmmllogger


class MemorySearchNode:
    """
    Node for searching memories relevant to the current user query by embedding similarity.

    Takes query embeddings from workflow state and searches for
    similar memories using the memory agent.
    """

    def __init__(
        self,
        memory_agent: "MemoryAgent",
        embedding_agent: "EmbeddingAgent",
    ):
        """Initialize memory search node with dependency injection.

        Args:
            memory_agent: Required MemoryAgent instance
            embedding_agent: Required EmbeddingAgent instance
        """
        self.agent = memory_agent
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
            message = langchain_message_to_message(state.current_user_message)
            message_text = extract_message_text(message)

            # Use injected EmbeddingAgent to generate embeddings
            embeddings = await self.embedding_agent.generate_embeddings([message_text])

            if not embeddings:
                err = "Embedding generation failed or returned empty result"
                self.logger.error(err, user_id=user_id)
                raise RuntimeError(err)

            self.logger.info(
                "Searching for similar memories",
                user_id=user_id,
                conversation_id=conversation_id,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
            )

            # Search for similar memories
            memories = await self.agent.search_memories_by_embedding(
                query_embeddings=embeddings,
                user_id=user_id,
                conversation_id=conversation_id,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                enable_cross_conversation=enable_cross_conversation,
                enable_cross_user=enable_cross_user,
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
            raise
