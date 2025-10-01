"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from typing import List

from models.lang_chain_message import LangChainMessage
from models.model_profile_type import ModelProfileType
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
# Lazy imports to avoid circular dependencies
from db import storage  # pylint: disable=import-outside-toplevel
from utils.model_profile import get_model_profile_for_task  # pylint: disable=import-outside-toplevel


class TitleGenerationNode:
    """
    Generates a conversation title if none exists.
    
    Uses grammar-constrained LLM to generate concise, descriptive titles
    based on conversation content.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize title generation node.
        
        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger

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
                for update in getattr(state, 'progress_updates', [])
            )
            if title_exists:
                return state

            # Need at least 2 messages (user + assistant) to generate meaningful title
            if len(state.messages) < 2:
                return state

            user_id = getattr(state, 'user_id', None)
            if not user_id:
                return state

            # User configuration and model profile will be accessed by pipeline factory internally

            self.logger.info(
                "Generating conversation title",
                extra={
                    "user_id": user_id,
                    "message_count": len(state.messages)
                }
            )

            # Create title generation pipeline with grammar constraints
            title_pipeline = await self.pipeline_factory.create_structured_pipeline(
                prompt_template=self._get_title_prompt(),
                output_schema=str,  # Simple string output
                enable_fallback=True
            )

            # Format conversation context
            conversation_context = self._format_conversation_context(state.messages)

            # Generate title
            title = await title_pipeline.execute({
                "conversation": conversation_context
            })

            # Update state with generated title via progress updates
            generated_title = title.strip() if isinstance(title, str) else "Untitled Conversation"
            
            # Add title to progress updates since WorkflowState doesn't have conversation_title field
            if not hasattr(state, 'progress_updates'):
                state.progress_updates = []
            state.progress_updates.append(f"Generated title: {generated_title}")
            
            self.logger.info(
                "Title generated successfully",
                extra={
                    "user_id": user_id,
                    "title": generated_title
                }
            )

            return state

        except Exception as e:
            self.logger.error(
                "Title generation failed",
                extra={
                    "user_id": getattr(state, 'user_id', 'unknown'),
                    "error": str(e)
                }
            )
            
            # Continue without title on error
            if not hasattr(state, 'progress_updates'):
                state.progress_updates = []
            state.progress_updates.append("Generated title: Untitled Conversation")
            return state

    def _get_title_prompt(self) -> str:
        """Get the title generation prompt template."""
        return """Generate a concise, descriptive title for this conversation.

Conversation:
{conversation}

Requirements:
- Maximum 8 words
- Capture the main topic or question
- Use clear, simple language
- No quotes or special characters

Title:"""

    def _format_conversation_context(self, messages: List[LangChainMessage]) -> str:
        """Format messages for title generation context."""
        context_lines = []
        
        for message in messages[:6]:  # Use first 6 messages
            role = "User" if getattr(message, 'role', 'user') == "user" else "Assistant"
            content = getattr(message, 'content', '')[:200]  # Truncate long messages
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)