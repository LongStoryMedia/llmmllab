"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from typing import List, TYPE_CHECKING


from runner import PipelineFactory
from utils.model_profile import get_model_profile_for_task
from models import (
    ModelProfileType,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    LangChainMessage,
)
from composer.graph.state import WorkflowState
from composer.nodes.base_node import BaseNode

if TYPE_CHECKING:
    from composer.agents.classifier_agent import ClassifierAgent


class TitleGenerationNode(BaseNode):
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
        super().__init__("title_generation", pipeline_factory=pipeline_factory, analysis_agent=analysis_agent)
        
    def _initialize_node(self, pipeline_factory=None, **kwargs) -> None:
        """Initialize TitleGenerationNode with dependency injection."""
        self.pipeline_factory = pipeline_factory
        self.classifier_agent = kwargs.get('analysis_agent')

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

            # Need at least 2 messages (user + assistant) to generate meaningful title
            if len(state.messages) < 2:
                return state

            user_id = getattr(state, "user_id", None)
            if not user_id:
                return state

            # User configuration and model profile will be accessed by pipeline factory internally

            self.logger.info(
                "Generating conversation title",
                extra={"user_id": user_id, "message_count": len(state.messages)},
            )

            prompt_template = self._get_title_prompt()
            conversation_context = self._format_conversation_context(state.messages)
            full_prompt = prompt_template.format(conversation=conversation_context)

            if not state.user_config:
                raise RuntimeError("User config missing for title generation")
            profile = await get_model_profile_for_task(
                state.user_config.model_profiles,
                ModelProfileType.PrimarySummary,
                user_id,
            )

            # Use existing text pipeline
            from runner import run_pipeline  # pylint: disable=import-outside-toplevel

            msg = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=full_prompt,
                    )
                ],
            )
            with self.pipeline_factory.pipeline(profile, str) as pipe:
                resp = await run_pipeline(messages=[msg], pipeline=pipe, tools=None)
            title = (
                resp.message.content[0].text
                if resp and resp.message
                else "Untitled Conversation"
            )

            # Update state with generated title via progress updates
            raw_title = (
                title.strip() if isinstance(title, str) else "Untitled Conversation"
            )

            # Validate and clean up the title
            generated_title = self._validate_and_clean_title(raw_title)

            # Add title to progress updates since WorkflowState doesn't have conversation_title field
            if not hasattr(state, "progress_updates"):
                state.progress_updates = []
            state.progress_updates.append(f"Generated title: {generated_title}")

            self.logger.info(
                "Title generated successfully",
                extra={
                    "user_id": user_id,
                    "title": generated_title,
                    "word_count": len(generated_title.split()),
                },
            )

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

    def _get_title_prompt(self) -> str:
        """Get the title generation prompt template."""
        return """Generate a concise, descriptive title for this conversation.

Conversation:
{conversation}

Requirements:
- Maximum 5 words only
- Capture the main topic or question
- Use clear, simple language
- No quotes or special characters
- Examples: "Workout Tips", "Tech News Summary"

Title:"""

    def _validate_and_clean_title(self, title: str) -> str:
        """Validate and clean up the generated title to meet requirements."""
        if not title or not title.strip():
            return "Untitled Conversation"

        # Clean up the title
        cleaned = title.strip()

        # Remove quotes and common prefixes
        cleaned = cleaned.strip("\"'")
        prefixes_to_remove = [
            "Title:",
            "title:",
            "TITLE:",
            "Generated title:",
            "Conversation title:",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        # Split into words and enforce 5-word limit
        words = cleaned.split()
        if len(words) > 5:
            words = words[:5]
            cleaned = " ".join(words)

        # Fallback if empty after cleaning
        if not cleaned or len(words) == 0:
            return "AI Discussion"

        # Capitalize properly
        cleaned = " ".join(word.capitalize() for word in cleaned.split())

        return cleaned

    def _format_conversation_context(self, messages: List[LangChainMessage]) -> str:
        """Format messages for title generation context."""
        context_lines = []

        for message in messages[:6]:  # Use first 6 messages
            role = "User" if getattr(message, "role", "user") == "user" else "Assistant"
            content = getattr(message, "content", "")[:200]  # Truncate long messages
            context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines)
