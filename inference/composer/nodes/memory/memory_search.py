"""
Memory Search Node for LangGraph workflows.
Searches for similar memories using embeddings.
"""

from typing import cast

from composer.agents.memory_agent import MemoryAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.utils.conversion import langchain_message_to_message
from runner import PipelineFactory, EmbeddingPipeline
from utils.model_profile import get_model_profile_for_task
from models import ModelProfileType


class MemorySearchNode:
    """
    Node for searching memories relevant to the current user query by embedding similarity.

    Takes query embeddings from workflow state and searches for
    similar memories using the memory agent.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        """Initialize memory search node."""
        self.agent = MemoryAgent()
        self.logger = composer_logger.logger.bind(component="MemorySearchNode")
        self.pipeline_factory = pipeline_factory

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

            from runner import embed_pipeline  # pylint: disable=import-outside-toplevel

            # get embedding profile
            profile = await get_model_profile_for_task(
                state.user_config.model_profiles, ModelProfileType.Embedding, user_id
            )
            # embed current user message
            with self.pipeline_factory.pipeline(
                profile,
                list[list[float]],
                user_circuit_breaker=state.user_config.circuit_breaker,
            ) as pipe:
                embeddings = await embed_pipeline(
                    langchain_message_to_message(state.current_user_message),
                    cast(EmbeddingPipeline, pipe),
                )

                if not embeddings:
                    err = "Embedding generation failed or returned empty result"
                    self.logger.error(err, user_id=user_id)
                    state.execution_metadata.add_error(err)
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

            # Format memories for context if needed
            if memories:
                state.execution_metadata.has_memory_context = True
            else:
                state.execution_metadata.has_memory_context = False

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
            # Escalate by recording in execution metadata and re-raising so test fails
            state.execution_metadata.add_error(f"Memory search failed: {str(e)}")
            state.execution_metadata.has_memory_context = False
            raise
