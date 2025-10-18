"""
Primary Summary Agent for detailed content summarization.
Specialized agent for creating comprehensive primary summaries with logical progression focus.
"""

import datetime
from typing import List, Optional, TYPE_CHECKING

from models import (
    PipelinePriority,
    ModelProfile,
    Message,
    SummaryStyle,
    Summary,
    SearchTopicSynthesis,
    SearchResult,
    UserConfig,
    NodeMetadata,
)
from runner import PipelineFactory
from composer.core.errors import NodeExecutionError
from utils.message import extract_message_text
from .base_agent import BaseAgent

if TYPE_CHECKING:
    from db.summary_storage import SummaryStorage
    from db.search_storage import SearchStorage


class PrimarySummaryAgent(BaseAgent[str]):
    """
    Primary Summary Agent for comprehensive content summarization.

    Specializes in creating detailed primary summaries that focus on logical
    progression and evolution of topics and ideas. Uses PrimarySummary model profile.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile: ModelProfile,
        node_metadata: NodeMetadata,
        summary_storage: "SummaryStorage",
        search_storage: "SearchStorage",
        user_config: UserConfig,
    ):
        """
        Initialize primary summary agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating summarization pipelines
            profile: Model profile for primary summary operations
            node_metadata: Node execution metadata for tracking
            summary_storage: Injected summary storage service
            search_storage: Injected search storage service
            user_config: User configuration object
        """
        super().__init__(
            pipeline_factory, profile, node_metadata, "PrimarySummaryAgent"
        )
        self.summary_storage = summary_storage
        self.search_storage = search_storage
        self.user_config = user_config

    async def summarize_search_results(
        self,
        search_results: List[SearchResult],
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.DETAILED,
    ) -> SearchTopicSynthesis:
        """
        Create primary summary synthesis from search results.

        Args:
            search_results: List of search results to synthesize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length
            style: Summary style preference
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            SearchTopicSynthesis with comprehensive primary analysis
        """
        try:
            self.logger.info(
                "Generating primary search results synthesis",
                results_count=len(search_results),
                style=style,
            )

            # Combine search content with primary focus
            content = await self._combine_search_content(search_results)
            prompt = await self._create_primary_search_prompt(
                content, style, max_length
            )

            summary = await self._execute_summarization_with_profile(prompt)

            # Extract comprehensive structured data
            topics = await self._extract_topics(summary)

            # Collect URLs from search results
            urls = []
            for result in search_results:
                if result.contents:
                    for content in result.contents:
                        if hasattr(content, "url") and content.url:
                            urls.append(content.url)

            synthesis = SearchTopicSynthesis(
                urls=urls,
                topics=topics,
                synthesis=summary,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                conversation_id=0,
            )

            self.logger.info(
                "Generated primary search synthesis",
                summary_length=len(summary),
                topics_count=len(topics),
                sources_count=len(synthesis.urls),
            )

            return synthesis

        except Exception as e:
            self.logger.error(
                "Failed to generate primary search synthesis",
                error=str(e),
            )
            raise NodeExecutionError(f"Primary search synthesis failed: {e}") from e

    async def summarize_conversation(
        self,
        messages: List[Message],
        conversation_id: int,
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.CONCISE,
    ) -> Summary:
        """
        Create primary summary of conversation messages.

        Args:
            messages: Conversation messages to summarize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length
            style: Summary style preference
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            Comprehensive primary conversation summary
        """
        try:
            self.logger.info(
                "Generating primary conversation summary",
                messages_count=len(messages),
                style=style,
            )

            # Create conversation text with primary focus on progression
            conversation_text = "\n".join(
                [f"{msg.role}: {extract_message_text(msg)}" for msg in messages]
            )

            prompt = await self._create_primary_conversation_prompt(
                conversation_text, style, max_length
            )

            summary_text = await self._execute_summarization_with_profile(prompt)

            summary = Summary(
                id=0,  # ID to be set when storing
                content=summary_text,
                level=1,
                conversation_id=conversation_id,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                source_ids=[msg.id for msg in messages if msg.id is not None],
            )

            # Store primary conversation summary
            await self.summary_storage.create_summary(summary)

            self.logger.info(
                "Generated primary conversation summary",
                summary_length=len(summary_text),
            )

            return summary

        except Exception as e:
            self.logger.error(
                "Failed to generate primary conversation summary",
                error=str(e),
            )
            raise NodeExecutionError(
                f"Primary conversation summarization failed: {e}"
            ) from e

    async def _create_primary_text_prompt(
        self,
        text: str,
        style: SummaryStyle,
        max_length: Optional[int],
    ) -> str:
        """Create specialized prompt for primary text summarization."""
        style_instruction = self._get_style_instruction(style)
        length_constraint = (
            f"Keep the summary under {max_length} words." if max_length else ""
        )

        return f"""As a comprehensive analysis expert, create a detailed primary summary focusing on logical progression and evolution of ideas.

PRIMARY SUMMARY REQUIREMENTS:
- Analyze the logical flow and development of concepts
- Identify key themes and their interconnections
- Highlight cause-and-effect relationships
- Capture the essence and significance of the content
- Maintain analytical depth while being accessible

CONTENT TO SUMMARIZE:
{text}

STYLE: {style_instruction}
{length_constraint}

Focus on the logical progression and evolution of topics and ideas throughout the content. Provide a comprehensive analysis that captures both detail and meaning."""

    async def _create_primary_search_prompt(
        self, content: str, style: SummaryStyle, max_length: Optional[int]
    ) -> str:
        """Create specialized prompt for primary search results synthesis."""
        style_instruction = self._get_style_instruction(style)
        length_constraint = (
            f"Keep the synthesis under {max_length} words." if max_length else ""
        )

        return f"""As a research synthesis expert, create a comprehensive primary analysis of these search results focusing on logical progression of information.

PRIMARY SYNTHESIS REQUIREMENTS:
- Synthesize information across all sources with analytical depth
- Identify patterns, trends, and relationships between findings
- Highlight consensus and contradictions in the information
- Provide comprehensive coverage of key themes
- Focus on logical progression and evolution of topics

SEARCH RESULTS TO SYNTHESIZE:
{content}

STYLE: {style_instruction}
{length_constraint}

Create a comprehensive synthesis that captures the logical progression and evolution of topics and ideas across all search results."""

    async def _create_primary_conversation_prompt(
        self, conversation_text: str, style: SummaryStyle, max_length: Optional[int]
    ) -> str:
        """Create specialized prompt for primary conversation summarization."""
        style_instruction = self._get_style_instruction(style)
        length_constraint = (
            f"Keep the summary under {max_length} words." if max_length else ""
        )

        return f"""As a conversation analysis expert, create a comprehensive primary summary focusing on the logical progression of the discussion.

PRIMARY CONVERSATION ANALYSIS REQUIREMENTS:
- Trace the evolution of topics and ideas throughout the conversation
- Identify key decision points and their rationale
- Highlight agreements, disagreements, and resolution processes
- Capture the flow of reasoning and argumentation
- Focus on logical progression and development of concepts

CONVERSATION TO SUMMARIZE:
{conversation_text}

STYLE: {style_instruction}
{length_constraint}

Focus on the logical progression and evolution of topics and ideas throughout the conversation. Provide comprehensive analysis of how the discussion developed."""

    def _get_style_instruction(self, style: SummaryStyle) -> str:
        """Get style-specific instruction for primary summaries."""
        style_instructions = {
            SummaryStyle.CONCISE: "Provide a focused yet comprehensive analysis.",
            SummaryStyle.DETAILED: "Provide extensive detail and comprehensive coverage.",
            SummaryStyle.BULLET_POINTS: "Use structured bullet points for comprehensive analysis.",
        }
        return style_instructions.get(
            style, "Provide a balanced comprehensive analysis."
        )

    # Import shared helper methods from the base class or create simplified versions
    async def _combine_search_content(self, search_results: List[SearchResult]) -> str:
        """Combine search results content for primary analysis."""
        content_parts = []
        for i, result in enumerate(search_results, 1):
            if result.contents:
                for j, content in enumerate(result.contents):
                    content_parts.append(
                        f"Source {i}.{j}: {content.title or 'No title'}"
                    )
                    if content.content:
                        content_parts.append(f"Content: {content.content}")
                    content_parts.append("")

        return "\n".join(content_parts)

    async def _execute_summarization_with_profile(
        self,
        prompt: str,
    ) -> str:
        """Execute summarization using the specified profile."""
        try:
            # Use the base agent's streaming interface
            messages = [prompt]
            response_chunks = []

            async for chunk in self.stream(
                messages=messages,
                priority=PipelinePriority.NORMAL,
            ):
                if chunk.message and chunk.message.content:
                    response_chunks.append(extract_message_text(chunk.message))

            return "".join(response_chunks)

        except Exception as e:
            self.logger.error(
                "Primary summarization execution failed",
                error=str(e),
            )
            raise NodeExecutionError(
                f"Failed to execute primary summarization: {e}"
            ) from e

    # Extraction methods for primary summaries
    async def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points using primary analysis approach."""
        # Simplified extraction - in production could use LLM
        sentences = summary.split(". ")
        # Return first few sentences as key points for primary summaries
        return [s.strip() + "." for s in sentences[:5] if len(s.strip()) > 20]

    async def _extract_topics(self, summary: str) -> List[str]:
        """Extract topics using primary analysis approach."""
        # Simplified extraction - in production could use LLM
        # For primary summaries, focus on main conceptual areas
        words = summary.lower().split()
        # Simple keyword extraction (could be enhanced with NLP)
        common_topics = [
            "analysis",
            "discussion",
            "decision",
            "development",
            "process",
            "result",
            "conclusion",
        ]
        found_topics = [topic for topic in common_topics if topic in words]
        return found_topics[:3]  # Return top 3 for primary focus

    async def _extract_decisions(self, summary: str) -> List[str]:
        """Extract decisions using primary analysis approach."""
        # Look for decision indicators in text
        decision_indicators = [
            "decided",
            "determined",
            "concluded",
            "agreed",
            "resolved",
        ]
        sentences = summary.split(". ")
        decisions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in decision_indicators):
                decisions.append(sentence.strip() + ".")
        return decisions[:3]

    async def _extract_action_items(self, summary: str) -> List[str]:
        """Extract action items using primary analysis approach."""
        # Look for action indicators in text
        action_indicators = [
            "will",
            "should",
            "must",
            "need to",
            "plan to",
            "next step",
        ]
        sentences = summary.split(". ")
        actions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in action_indicators):
                actions.append(sentence.strip() + ".")
        return actions[:3]
