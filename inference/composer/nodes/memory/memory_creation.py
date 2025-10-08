"""
Memory Creation Node for LangGraph workflows.
Creates Memory objects from summaries, messages, or search topic synthesis.
"""

from typing import List, Optional, cast, Union
from datetime import datetime, timezone

from models import (
    Memory,
    MemoryFragment,
    LangChainMessage,
    Summary,
    SearchTopicSynthesis,
    ModelProfile,
    MemorySource,
    MessageRole,
    PipelinePriority,
    CircuitBreakerConfig,
    ModelProfileType,
    Message,
)
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.utils.conversion import langchain_message_to_message
from composer.agents.embedding_agent import EmbeddingAgent
from runner import PipelineFactory, EmbeddingPipeline
import runner
from utils.model_profile import get_model_profile_for_task
from utils.message import extract_message_text
import asyncio


class MemoryCreationNode:
    """
    Node for creating Memory objects from various sources.

    Takes summaries, messages, or search topic synthesis from workflow state
    and converts them into Memory objects that can be stored later.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        """Initialize memory creation node."""
        self.logger = composer_logger.logger.bind(component="MemoryCreationNode")
        self.pipeline_factory = pipeline_factory

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Create Memory objects from available content sources.

        Args:
            state: Current workflow state with summaries, messages, or search synthesis

        Returns:
            Updated workflow state with created memories populated
        """
        try:
            created_memories = []
            assert state.user_config
            assert state.user_id

            cbc = state.user_config.circuit_breaker or DEFAULT_CIRCUIT_BREAKER_CONFIG
            profile = await get_model_profile_for_task(
                state.user_config.model_profiles,
                ModelProfileType.Embedding,
                state.user_id,
            )

            if state.things_to_remember:
                memories_from_things = await self._create_memories(
                    state.things_to_remember,
                    profile,
                    cbc,
                )
                created_memories.extend(memories_from_things)
                self.logger.info(
                    "Created memories from things to remember",
                    user_id=getattr(state, "user_id", "unknown"),
                    item_count=len(state.things_to_remember),
                    memories_created=len(memories_from_things),
                )

            # Add created memories to state
            for memory in created_memories:
                state.created_memories.append(memory)

            self.logger.info(
                "Memory creation completed",
                user_id=getattr(state, "user_id", "unknown"),
                total_memories_created=len(created_memories),
            )

            return state

        except Exception as e:
            self.logger.error(
                "Memory creation node failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            # Don't raise - add error to state and continue workflow
            if hasattr(state, "error_details"):
                state.error_details.append(f"Memory creation failed: {str(e)}")
            elif hasattr(state, "execution_metadata"):
                state.execution_metadata.add_error(f"Memory creation failed: {str(e)}")
            return state

    async def _create_memories(
        self,
        things_to_remember: List[
            Union[Message, LangChainMessage, Summary, SearchTopicSynthesis]
        ],
        profile: ModelProfile,
        cbc: CircuitBreakerConfig,
    ) -> List[Memory]:
        """
        Create Memory objects from a mixed list of content sources.

        Args:
            things_to_remember: List of Message, LangChainMessage, Summary, or SearchTopicSynthesis objects

        Returns:
            List of Memory objects created from the input content sources
        """
        memories = []
        summaries = []
        messages = []
        search_syntheses = []

        for item in things_to_remember:
            if isinstance(item, Summary):
                summaries.append(item)
            elif isinstance(item, LangChainMessage):
                messages.append(langchain_message_to_message(item))
            elif isinstance(item, Message):
                messages.append(item)
            elif isinstance(item, SearchTopicSynthesis):
                search_syntheses.append(item)

        # Process different types concurrently
        tasks = []

        if summaries:
            tasks.append(
                self._create_memories_from_summaries(
                    summaries,
                    conversation_id=0,
                    profile=profile,
                    cbc=cbc,
                )
            )

        if messages:
            tasks.append(
                self._create_memories_from_messages(
                    messages,
                    conversation_id=0,
                    profile=profile,
                    cbc=cbc,
                )
            )

        if search_syntheses:
            tasks.append(
                self._create_memories_from_search_syntheses(
                    search_syntheses,
                    conversation_id=0,
                    profile=profile,
                    cbc=cbc,
                )
            )

        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Append successful results to memories
            for result in results:
                if isinstance(result, list):
                    memories.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(
                        "Failed to create memories from source", error=str(result)
                    )

        return memories

    async def _create_memories_from_summaries(
        self,
        summaries: List[Summary],
        conversation_id: int,
        profile: ModelProfile,
        cbc: CircuitBreakerConfig,
    ) -> List[Memory]:
        """
        Create Memory objects from Summary objects.

        Args:
            summaries: List of Summary objects
            conversation_id: Current conversation ID

        Returns:
            List of Memory objects created from summaries
        """
        memories = []

        for summary in summaries:
            # Create a memory fragment from the summary content
            with self.pipeline_factory.pipeline(
                profile, List[List[float]], PipelinePriority.NORMAL, cbc
            ) as pipe:
                embeddings = await runner.embed_pipeline(
                    summary.content, cast(EmbeddingPipeline, pipe)
                )

                fragment = MemoryFragment(
                    id=summary.id,
                    role=MessageRole.SYSTEM,  # Summaries are generated content
                    content=summary.content,
                    embeddings=embeddings,
                )

                # Create memory object
                memory = Memory(
                    fragments=[fragment],
                    source=MemorySource.SUMMARY,
                    created_at=summary.created_at,
                    similarity=1.0,  # Not applicable for new memories
                    source_id=summary.id,
                    conversation_id=conversation_id,
                )

                memories.append(memory)

        return memories

    async def _create_memories_from_messages(
        self,
        messages: List[Message],
        conversation_id: int,
        profile: ModelProfile,
        cbc: CircuitBreakerConfig,
    ) -> List[Memory]:
        """
        Create Memory objects from LangChainMessage objects.

        Args:
            messages: List of LangChainMessage objects
            conversation_id: Current conversation ID

        Returns:
            List of Memory objects created from messages
        """
        memories = []

        for msg in messages:
            with self.pipeline_factory.pipeline(
                profile, List[List[float]], PipelinePriority.NORMAL, cbc
            ) as pipe:
                embeddings = await runner.embed_pipeline(
                    extract_message_text(msg),
                    cast(EmbeddingPipeline, pipe),
                )

                fragment = MemoryFragment(
                    id=msg.id or 0,
                    role=msg.role,
                    embeddings=embeddings,
                )

                # Create memory object
                memory = Memory(
                    fragments=[fragment],
                    source=MemorySource.MESSAGE,
                    created_at=datetime.now(timezone.utc),
                    similarity=1.0,  # Not applicable for new memories
                    source_id=msg.id or 0,
                    conversation_id=conversation_id,
                )

                memories.append(memory)

        return memories

    async def _create_memories_from_search_syntheses(
        self,
        search_syntheses: List[SearchTopicSynthesis],
        conversation_id: int,
        profile: ModelProfile,
        cbc: CircuitBreakerConfig,
    ) -> List[Memory]:
        """
        Create Memory object from SearchTopicSynthesis.

        Args:
            search_synthesis: SearchTopicSynthesis object
            conversation_id: Current conversation ID

        Returns:
            Memory object created from search synthesis, or None if no content
        """
        memories = []
        for synthesis in search_syntheses:
            with self.pipeline_factory.pipeline(
                profile, List[List[float]], PipelinePriority.NORMAL, cbc
            ) as pipe:
                embeddings = await runner.embed_pipeline(
                    synthesis.synthesis,
                    cast(EmbeddingPipeline, pipe),
                )
            # Create a memory fragment from the synthesis content
            fragment = MemoryFragment(
                id=synthesis.id or 0,
                role=MessageRole.SYSTEM,  # Search synthesis is generated content
                embeddings=embeddings,
            )

            # Create memory object
            memory = Memory(
                fragments=[fragment],
                source=MemorySource.SEARCH,
                created_at=synthesis.created_at,
                similarity=1.0,  # Not applicable for new memories
                source_id=synthesis.id or 0,
                conversation_id=conversation_id,
            )

            memories.append(memory)

        return memories
