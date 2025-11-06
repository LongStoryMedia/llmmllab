"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from typing import TYPE_CHECKING


from runner import PipelineFactory
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger
from utils.message_conversion import lc_messages_to_messages

if TYPE_CHECKING:
    from composer.agents.classifier_agent import ClassifierAgent


class TitleGenerationNode:
    """
    Generates a conversation title if none exists.

    Uses grammar-constrained LLM to generate concise, descriptive titles
    based on conversation content.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        analysis_agent: "ClassifierAgent",
    ):
        """Initialize title generation node with dependency injection.

        Args:
            pipeline_factory: Factory for creating pipelines
            analysis_agent: Required ClassifierAgent instance
        """
        self.pipeline_factory = pipeline_factory
        self.classifier_agent = analysis_agent
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
            title = await self.classifier_agent.generate_title(state.messages)

            if title and title.strip():
                self.logger.info("Title generated successfully", extra={"title": title})
                state.title = title
            else:
                self.logger.warning("Title generation returned empty result")

            # Cleanup classifier agent resources after completion
            self.classifier_agent.cleanup()

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
            self.classifier_agent.cleanup()

            # Escalate by raising so tests fail visibly
            raise
