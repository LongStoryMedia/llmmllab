"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from typing import TYPE_CHECKING


from runner import PipelineFactory
from composer.graph.state import WorkflowState
from composer.utils.conversion import convert_langchain_messages_to_messages
from utils.logging import llmmllogger

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
            # Skip if title already exists (check progress_updates for title info)
            title_exists = any(
                "title" in str(update).lower()
                for update in getattr(state, "progress_updates", [])
            )
            if title_exists:
                return state

            title = await self.classifier_agent.generate_title(
                convert_langchain_messages_to_messages(state.messages)
            )

            state.title = title

            return state

        except Exception as e:
            self.logger.error(
                "Title generation failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            # Escalate by raising so tests fail visibly
            raise
