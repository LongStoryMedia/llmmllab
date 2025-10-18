"""
Master Summary Agent for consolidating and synthesizing multiple summaries.
Specialized agent for creating master summaries that combine and synthesize key information from multiple sources.
"""

import datetime
from typing import List, Optional, Any, TYPE_CHECKING

from models import (
    PipelinePriority,
    ModelProfile,
    Message,
    Summary,
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


class MasterSummaryAgent(BaseAgent[str]):
    """
    Master Summary Agent for consolidating multiple summaries.

    Specializes in combining and synthesizing key information from multiple
    summaries into coherent, comprehensive overviews. Uses MasterSummary model profile.
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
        Initialize master summary agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating summarization pipelines
            profile: Model profile for master summary operations
            node_metadata: Node execution metadata for tracking
            summary_storage: Injected summary storage service
            search_storage: Injected search storage service
            user_config: User configuration object
        """
        super().__init__(pipeline_factory, profile, node_metadata, "MasterSummaryAgent")
        self.summary_storage = summary_storage
        self.search_storage = search_storage
        self.user_config = user_config

    async def consolidate_summaries(
        self,
        summaries: List[Summary],
        user_id: str,
        conversation_id: int,
        level: int = 1,
        target_level: int = 2,
        grammar: Optional[Any] = None,
    ) -> Summary:
        """
        Consolidate multiple summaries into a higher-level master summary.

        Args:
            summaries: List of summaries to consolidate
            user_id: User identifier for model profile retrieval
            conversation_id: ID of the conversation being summarized
            level: Current summary level
            target_level: Target consolidation level
            grammar: Optional grammar constraints for structured output

        Returns:
            Master summary consolidating all input summaries
        """
        try:
            self.logger.info(
                "Consolidating summaries into master summary",
                user_id=user_id,
                conversation_id=conversation_id,
                summaries_count=len(summaries),
                level=level,
                target_level=target_level,
            )

            # Create consolidation prompt from multiple summaries
            prompt = await self._create_master_consolidation_prompt(
                summaries, level, target_level
            )

            # Use base agent to generate the master summary
            consolidated_content = await self._execute_summarization_with_profile(
                prompt, user_id, grammar=grammar
            )

            # Extract comprehensive master-level insights
            all_key_points = []
            all_topics = []

            # Note: Could extract from Summary content or stored metadata
            # For now, simplified approach - in production these could be extracted
            # using LLM analysis or stored as separate fields

            # Deduplicate and synthesize
            synthesized_key_points = await self._synthesize_key_points(all_key_points)
            synthesized_topics = await self._synthesize_topics(all_topics)
            # Note: decisions and actions synthesis skipped for simplicity            # Create master summary object with correct Summary schema
            master_summary = Summary(
                id=hash(
                    f"master_{conversation_id}_{level}_{datetime.datetime.now().isoformat()}"
                )
                % 2**31,
                content=consolidated_content,
                level=target_level,
                conversation_id=conversation_id,
                source_ids=[s.id for s in summaries],
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )

            # Store the master summary
            await self.summary_storage.create_summary(master_summary)

            self.logger.info(
                "Generated master consolidated summary",
                user_id=user_id,
                conversation_id=conversation_id,
                summary_id=master_summary.id,
                content_length=len(consolidated_content),
                key_points_count=len(synthesized_key_points),
                topics_count=len(synthesized_topics),
            )

            return master_summary

        except Exception as e:
            self.logger.error(
                "Master summary consolidation failed",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Master summary consolidation failed: {e}") from e

    async def synthesize_conversation_master(
        self,
        messages: List[Message],
        user_id: str,
        conversation_id: int,
        max_length: Optional[int] = None,
        grammar: Optional[Any] = None,
    ) -> Summary:
        """
        Create master-level conversation summary with comprehensive synthesis.

        Args:
            messages: Conversation messages to synthesize
            user_id: User identifier for model profile retrieval
            conversation_id: ID of the conversation being summarized
            max_length: Optional maximum summary length
            grammar: Optional grammar constraints for structured output

        Returns:
            Master conversation summary with comprehensive synthesis
        """
        try:
            self.logger.info(
                "Generating master conversation synthesis",
                user_id=user_id,
                conversation_id=conversation_id,
                messages_count=len(messages),
            )

            # Create comprehensive conversation text for master analysis
            conversation_text = await self._create_master_conversation_text(messages)

            prompt = await self._create_master_conversation_prompt(
                conversation_text, max_length
            )

            # Use base agent to generate the master summary
            master_content = await self._execute_summarization_with_profile(
                prompt, user_id, grammar=grammar
            )

            # Extract comprehensive master-level structured data
            key_points = await self._extract_master_key_points(master_content)
            # Note: topics, decisions, and actions extraction skipped for simplicity

            # Create master conversation summary with correct Summary schema
            master_summary = Summary(
                id=hash(
                    f"master_conv_{conversation_id}_{datetime.datetime.now().isoformat()}"
                )
                % 2**31,
                content=master_content,
                level=2,  # Master level
                conversation_id=conversation_id,
                source_ids=[],  # No specific source summaries for conversation synthesis
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )

            # Store the master conversation summary
            await self.summary_storage.create_summary(master_summary)

            self.logger.info(
                "Generated master conversation summary",
                user_id=user_id,
                conversation_id=conversation_id,
                summary_id=master_summary.id,
                content_length=len(master_content),
                key_points_count=len(key_points),
            )

            return master_summary

        except Exception as e:
            self.logger.error(
                "Failed to generate master conversation synthesis",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e),
            )
            raise NodeExecutionError(
                f"Master conversation synthesis failed: {e}"
            ) from e

    async def synthesize_cross_conversation_master(
        self,
        summaries: List[Summary],
        user_id: str,
        topic: str,
        max_length: Optional[int] = None,
        grammar: Optional[Any] = None,
    ) -> Summary:
        """
        Create master synthesis across multiple conversation summaries.

        Args:
            summaries: List of conversation summaries to synthesize across
            user_id: User identifier for model profile retrieval
            topic: Topic or theme connecting the conversations
            max_length: Optional maximum summary length
            grammar: Optional grammar constraints for structured output

        Returns:
            Cross-conversation master synthesis
        """
        try:
            self.logger.info(
                "Generating cross-conversation master synthesis",
                user_id=user_id,
                topic=topic,
                summaries_count=len(summaries),
            )

            prompt = await self._create_cross_conversation_master_prompt(
                summaries, topic, max_length
            )

            # Use base agent to generate the master synthesis
            synthesis_content = await self._execute_summarization_with_profile(
                prompt, user_id, grammar=grammar
            )

            # Note: For simplicity, cross-conversation synthesis skipped
            # In production, this would extract and synthesize data across conversations

            # Create cross-conversation master synthesis with correct Summary schema
            master_synthesis = Summary(
                id=hash(f"master_cross_{topic}_{datetime.datetime.now().isoformat()}")
                % 2**31,
                content=synthesis_content,
                level=3,  # Cross-conversation level
                conversation_id=0,  # Not tied to a specific conversation
                source_ids=[s.id for s in summaries],
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )

            # Store the cross-conversation master synthesis
            await self.summary_storage.create_summary(master_synthesis)

            self.logger.info(
                "Generated cross-conversation master synthesis",
                user_id=user_id,
                topic=topic,
                summary_id=master_synthesis.id,
                content_length=len(synthesis_content),
            )

            return master_synthesis

        except Exception as e:
            self.logger.error(
                "Failed to generate cross-conversation master synthesis",
                user_id=user_id,
                topic=topic,
                error=str(e),
            )
            raise NodeExecutionError(
                f"Cross-conversation master synthesis failed: {e}"
            ) from e

    async def _create_master_consolidation_prompt(
        self, summaries: List[Summary], level: int, target_level: int
    ) -> str:
        """Create specialized prompt for master summary consolidation."""
        summaries_text = "\n\n".join(
            [
                f"Summary {i+1} (Level {getattr(s, 'level', 1)}):\n{s.content}"
                for i, s in enumerate(summaries)
            ]
        )

        return f"""As a master synthesis expert, consolidate these summaries into a comprehensive master summary that combines and synthesizes key information.

MASTER CONSOLIDATION REQUIREMENTS:
- Combine and synthesize key information from all summaries into a coherent, comprehensive overview
- Identify patterns, themes, and connections across summaries
- Eliminate redundancy while preserving essential information
- Create a unified narrative that encompasses all key insights
- Focus on synthesis rather than just compilation

SUMMARIES TO CONSOLIDATE (Current Level {level} → Target Level {target_level}):
{summaries_text}

Create a master summary that provides a comprehensive overview by combining and synthesizing the key information from all input summaries. Focus on creating a coherent, unified perspective."""

    async def _create_master_conversation_prompt(
        self, conversation_text: str, max_length: Optional[int]
    ) -> str:
        """Create specialized prompt for master conversation synthesis."""
        length_constraint = (
            f"Keep the summary under {max_length} words." if max_length else ""
        )

        return f"""As a master conversation analyst, create a comprehensive synthesis that combines and synthesizes all key information from this conversation.

MASTER CONVERSATION SYNTHESIS REQUIREMENTS:
- Provide a comprehensive overview that synthesizes all major themes
- Identify and connect key decision points, agreements, and outcomes
- Highlight the overall progression and resolution of topics
- Combine individual insights into a unified understanding
- Focus on synthesis and comprehensive coverage

CONVERSATION TO SYNTHESIZE:
{conversation_text}

{length_constraint}

Create a master-level synthesis that combines and synthesizes the key information from the entire conversation into a coherent, comprehensive overview."""

    async def _create_cross_conversation_master_prompt(
        self, summaries: List[Summary], topic: str, max_length: Optional[int]
    ) -> str:
        """Create specialized prompt for cross-conversation master synthesis."""
        length_constraint = (
            f"Keep the synthesis under {max_length} words." if max_length else ""
        )

        summaries_text = "\n\n".join(
            [f"Conversation {i+1}:\n{s.content}" for i, s in enumerate(summaries)]
        )

        return f"""As a cross-conversation synthesis expert, create a master synthesis that combines insights across multiple conversations on the topic: {topic}

CROSS-CONVERSATION MASTER SYNTHESIS REQUIREMENTS:
- Synthesize information and insights across all conversations
- Identify patterns, evolution, and development of ideas over time
- Combine perspectives and approaches from different conversations
- Create a unified understanding of the topic across contexts
- Focus on comprehensive synthesis and meta-insights

TOPIC: {topic}

CONVERSATION SUMMARIES:
{summaries_text}

{length_constraint}

Create a master synthesis that combines and synthesizes key information from all conversations to provide a comprehensive overview of {topic}."""

    async def _create_master_conversation_text(self, messages: List[Message]) -> str:
        """Create comprehensive conversation text for master analysis."""
        # Group messages by participant for better master analysis
        conversation_parts = []
        current_speaker = None
        current_block = []

        for msg in messages:
            if msg.role != current_speaker:
                if current_block and current_speaker:
                    conversation_parts.append(
                        f"{current_speaker}:\n" + "\n".join(current_block)
                    )
                current_speaker = msg.role
                current_block = [extract_message_text(msg)]
            else:
                current_block.append(extract_message_text(msg))

        # Add the last block
        if current_block and current_speaker:
            conversation_parts.append(
                f"{current_speaker}:\n" + "\n".join(current_block)
            )

        return "\n\n".join(conversation_parts)

    async def _execute_summarization_with_profile(
        self,
        prompt: str,
        user_id: str,
        grammar: Optional[Any] = None,
    ) -> str:
        """Execute summarization using the master summary profile."""
        try:
            # Use the base agent's streaming interface
            messages = [prompt]
            response_chunks = []

            async for chunk in self.stream(
                messages=messages,
                priority=PipelinePriority.NORMAL,
                grammar=grammar,
            ):
                if chunk.message and chunk.message.content:
                    for content in chunk.message.content:
                        if hasattr(content, "text") and content.text:
                            response_chunks.append(content.text)

            return "".join(response_chunks).strip()

        except Exception as e:
            self.logger.error(
                "Master summarization execution failed",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(
                f"Failed to execute master summarization: {e}"
            ) from e

    # Synthesis methods for master-level analysis
    async def _synthesize_key_points(self, all_key_points: List[str]) -> List[str]:
        """Synthesize and deduplicate key points for master summary."""
        if not all_key_points:
            return []

        # Simple deduplication and ranking for master synthesis
        unique_points = list(set(all_key_points))
        # Sort by length to prioritize more detailed points
        unique_points.sort(key=len, reverse=True)
        return unique_points[:10]  # Return top 10 for master summary

    async def _synthesize_topics(self, all_topics: List[str]) -> List[str]:
        """Synthesize and consolidate topics for master summary."""
        if not all_topics:
            return []

        # Count frequency and return most common topics
        topic_counts = {}
        for topic in all_topics:
            topic_lower = topic.lower()
            topic_counts[topic_lower] = topic_counts.get(topic_lower, 0) + 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:8]]

    async def _synthesize_decisions(self, all_decisions: List[str]) -> List[str]:
        """Synthesize decisions for master summary."""
        if not all_decisions:
            return []

        # Deduplicate and prioritize decisions
        unique_decisions = list(set(all_decisions))
        return unique_decisions[:5]  # Return top 5 for master synthesis

    async def _synthesize_action_items(self, all_action_items: List[str]) -> List[str]:
        """Synthesize action items for master summary."""
        if not all_action_items:
            return []

        # Deduplicate and prioritize action items
        unique_actions = list(set(all_action_items))
        return unique_actions[:7]  # Return top 7 for master synthesis

    # Master-level extraction methods
    async def _extract_master_key_points(self, content: str) -> List[str]:
        """Extract key points with master-level analysis depth."""
        # Enhanced extraction for master summaries
        sentences = content.split(". ")
        # Focus on substantive sentences for master analysis
        key_sentences = [s.strip() + "." for s in sentences if len(s.strip()) > 30]
        return key_sentences[:8]

    async def _extract_master_topics(self, content: str) -> List[str]:
        """Extract topics with master-level synthesis focus."""
        # Enhanced topic extraction for master summaries
        words = content.lower().split()
        master_topics = [
            "strategy",
            "decision",
            "outcome",
            "analysis",
            "synthesis",
            "conclusion",
            "recommendation",
            "insight",
        ]
        found_topics = [topic for topic in master_topics if topic in words]
        return found_topics[:5]

    async def _extract_master_decisions(self, content: str) -> List[str]:
        """Extract decisions with master-level comprehensive analysis."""
        decision_indicators = [
            "decided",
            "determined",
            "concluded",
            "agreed",
            "resolved",
            "established",
            "finalized",
        ]
        sentences = content.split(". ")
        decisions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in decision_indicators):
                decisions.append(sentence.strip() + ".")
        return decisions[:5]

    async def _extract_master_action_items(self, content: str) -> List[str]:
        """Extract action items with master-level strategic focus."""
        action_indicators = [
            "will implement",
            "should develop",
            "must establish",
            "plan to execute",
            "next steps include",
            "strategic priority",
        ]
        sentences = content.split(". ")
        actions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in action_indicators):
                actions.append(sentence.strip() + ".")
        return actions[:5]
