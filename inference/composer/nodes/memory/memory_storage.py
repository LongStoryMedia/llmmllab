"""
Memory Storage Node for LangGraph workflows.
Stores messages as memories with their embeddings.
"""

from composer.agents.memory_agent import MemoryAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class MemoryStorageNode:
    """
    Node for storing memories.

    Takes messages and their corresponding embeddings from workflow state
    and stores them using the memory agent.
    """

    def __init__(self):
        """Initialize memory storage node."""
        self.agent = MemoryAgent()
        self.logger = composer_logger.logger.bind(component="MemoryStorageNode")

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
                state.execution_metadata.memories_stored = True
                return state

            self.logger.info(
                "Storing memories",
                user_id=state.user_id,
                memory_count=len(state.created_memories),
                conversation_id=state.conversation_id,
            )

            # Store memories using the agent
            success = await self.agent.store_memories(
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                memories=state.created_memories,
            )

            # Store result in state metadata
            state.execution_metadata.memories_stored = success

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
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Memory storage failed: {str(e)}")
            state.execution_metadata.memories_stored = False
            return state
