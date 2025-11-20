"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from composer.agents.base_agent import BaseAgent
from models import NodeMetadata
from runner import PipelineFactory
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger


class TitleGenerationNode:
    """
    Generates a conversation title if none exists.

    Uses grammar-constrained LLM to generate concise, descriptive titles
    based on conversation content.
    """

    def __init__(self, agent: BaseAgent, node_metadata: NodeMetadata):
        """Initialize title generation node with dependency injection.

        Args:
            agent: Required ClassifierAgent instance
        """
        self.agent = agent.bind_node_metadata(node_metadata)
        self.logger = llmmllogger.logger.bind(component="TitleGenerationNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Generate conversation title if needed.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with title
        """
        try:
            assert (
                state.conversation_id is not None
            ), "Conversation ID must be set in state"
            assert state.user_id is not None, "User ID must be set in state"

            self.logger.info(
                "TitleGenerationNode called",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "existing_title": getattr(state, "title", None),
                    "conversation_id": getattr(state, "conversation_id", None),
                },
            )

            # Primary check: Skip if title already exists in state
            if hasattr(state, "title") and state.title and state.title.strip():
                self.logger.info(
                    "Title already exists, skipping generation",
                    extra={"existing_title": state.title},
                )
                return state

            title = await self.agent.generate_title(state.messages)

            if title and title.strip():
                self.logger.info("Title generated successfully", extra={"title": title})
                state.title = title
            else:
                self.logger.warning("Title generation returned empty result")

            from db import storage  # pylint: disable=import-outside-toplevel

            # Persist generated title to database
            await storage.get_service(storage.conversation).update_conversation_title(
                title, state.conversation_id, state.user_id
            )

            # Cleanup classifier agent resources after completion
            self.agent.cleanup()

            return state

        except Exception as e:
            self.logger.error(
                "Title generation failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )

            # Cleanup classifier agent resources even on error
            self.agent.cleanup()

            # Escalate by raising so tests fail visibly
            raise
