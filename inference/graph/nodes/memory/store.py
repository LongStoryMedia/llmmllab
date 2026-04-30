"""
Memory Storage Node for LangGraph workflows.
Stores messages as memories with their embeddings.
"""

from typing import List, TYPE_CHECKING
from graph.state import WorkflowState
from models import Memory
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from db.memory_storage import MemoryStorage


class MemoryStorageNode:
    """
    Node for storing memories.

    Takes messages and their corresponding embeddings from workflow state
    and stores them using the memory agent.
    """

    def __init__(
        self,
        memory_storage: "MemoryStorage",
    ):
        """Initialize memory storage node with dependency injection.

        Args:
            memory_agent: Required MemoryAgent instance
        """
        self.memory_storage = memory_storage
        self.logger = llmmllogger.logger.bind(component="MemoryStorageNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Store messages as memories with their embeddings.

        Args:
            state: Current workflow state with messages and embeddings

        Returns:
            Updated workflow state with storage results
        """
        try:
            assert state.user_id is not None
            assert state.conversation_id is not None

            if not state.created_memories:
                self.logger.info("No new memories to store", user_id=state.user_id)
                return state

            self.logger.info(
                "Storing memories",
                user_id=state.user_id,
                memory_count=len(state.created_memories),
                conversation_id=state.conversation_id,
            )

            # Store memories using the agent
            success = await self.store_memories(
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                memories=state.created_memories,
            )

            if success:
                self.logger.info("Successfully stored memories", user_id=state.user_id)
            else:
                self.logger.warning("Memory storage failed", user_id=state.user_id)

            return state

        except Exception as e:
            self.logger.error(
                "Memory storage node failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            return state

    async def store_memories(
        self,
        user_id: str,
        conversation_id: int,
        memories: List[Memory],
    ) -> bool:
        """
        Store messages as memories with their corresponding embeddings.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            messages: List of message data
            embeddings: Corresponding embeddings for messages

        Returns:
            True if storage succeeded, False otherwise
        """
        try:
            # Use injected storage service
            memory_svc = self.memory_storage

            # Store each message with its embedding
            success_count = 0
            for mem in memories:
                for frag in mem.fragments:
                    try:
                        if frag.embeddings is None:
                            self.logger.warning(
                                "Skipping memory fragment with missing embeddings",
                                user_id=user_id,
                                fragment_id=frag.id,
                            )
                            continue
                        await memory_svc.store_memory(
                            user_id=user_id,
                            source=mem.source,
                            role=frag.role,
                            source_id=mem.source_id,
                            embeddings=frag.embeddings,
                        )
                        success_count += 1
                    except Exception as e:
                        self.logger.warning(
                            "Failed to store individual memory fragment",
                            user_id=user_id,
                            fragment_id=frag.id,
                            error=str(e),
                        )

            self.logger.info(
                "Memory storage completed",
                user_id=user_id,
                conversation_id=conversation_id,
                total_messages=len(memories),
                successful_stores=success_count,
            )

            return success_count > 0

        except Exception as e:
            self.logger.error(
                "Memory storage failed",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e),
            )
            return False
