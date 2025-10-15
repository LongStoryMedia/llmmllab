"""
Brief Summary Agent for concise, action-oriented summarization.
Specialized agent for creating brief summaries focused on key decisions, conclusions, and actionable outcomes.
"""

import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from models import (
    ModelProfileType,
    PipelinePriority,
    ModelProfile,
    Message,
    SummaryType,
    SummaryStyle,
    Summary,
    SearchTopicSynthesis,
    SearchResult,
    UserConfig,
    NodeMetadata,
)
from runner import PipelineFactory
from composer.core.errors import NodeExecutionError
from utils.model_profile import get_model_profile_for_task
from utils.message import extract_message_text
from .base_agent import BaseAgent

if TYPE_CHECKING:
    from db.summary_storage import SummaryStorage
    from db.search_storage import SearchStorage


class BriefSummaryAgent(BaseAgent[str]):
    """
    Brief Summary Agent for concise, action-oriented summarization.

    Specializes in creating brief summaries that focus on consolidating 
    key decisions, conclusions, and actionable outcomes. Uses BriefSummary model profile.
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
        Initialize brief summary agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating summarization pipelines
            profile: Model profile for brief summary operations
            node_metadata: Node execution metadata for tracking
            summary_storage: Injected summary storage service
            search_storage: Injected search storage service
            user_config: User configuration object
        """
        super().__init__(pipeline_factory, profile, node_metadata, "BriefSummaryAgent")
        self.summary_storage = summary_storage
        self.search_storage = search_storage
        self.user_config = user_config

    async def summarize_text_brief(
        self,
        text: str,
        user_id: str,
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.CONCISE,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Create brief summary of input text focusing on key outcomes and decisions.

        Args:
            text: Text content to summarize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length (typically short for brief)
            style: Summary style (defaults to concise for brief summaries)
            tools: Optional tools available to the agent for enhanced capabilities
            grammar: Optional grammar constraints for structured output

        Returns:
            Brief summary focusing on key decisions and actionable outcomes
        """
        try:
            self.logger.info(
                "Generating brief text summary",
                user_id=user_id,
                text_length=len(text),
                max_length=max_length,
                style=style,
                has_tools=bool(tools),
            )

            # Brief summaries have stricter length constraints
            brief_max_length = min(max_length or 200, 300)

            prompt = await self._create_brief_text_prompt(text, style, brief_max_length)

            # Get specialized brief summary model profile
            profile = get_model_profile_for_task(
                ModelProfileType.BriefSummary, 
                user_id, 
                self.user_config
            )

            summary = await self._execute_summarization_with_profile(
                profile, prompt, user_id, tools, grammar
            )

            # Extract focused brief data - decisions and actions only
            decisions = await self._extract_brief_decisions(summary, user_id)
            action_items = await self._extract_brief_action_items(summary, user_id)
            conclusions = await self._extract_brief_conclusions(summary, user_id)

            # Store brief summary with action-oriented metadata
            await self._store_brief_summary(
                summary, user_id, text, SummaryType.BRIEF, style, 
                decisions, action_items, conclusions
            )

            self.logger.info(
                "Generated brief summary",
                user_id=user_id,
                summary_length=len(summary),
                decisions_count=len(decisions),
                action_items_count=len(action_items),
            )

            return summary

        except Exception as e:
            self.logger.error(
                "Failed to generate brief text summary",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Brief text summarization failed: {e}")

    async def summarize_search_results_brief(
        self,
        search_results: List[SearchResult],
        user_id: str,
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.CONCISE,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> SearchTopicSynthesis:
        """
        Create brief synthesis from search results focusing on actionable insights.

        Args:
            search_results: List of search results to synthesize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length
            style: Summary style preference
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            SearchTopicSynthesis with brief, action-oriented analysis
        """
        try:
            self.logger.info(
                "Generating brief search results synthesis",
                user_id=user_id,
                results_count=len(search_results),
                style=style,
            )

            # Combine search content with brief focus
            content = await self._combine_search_content_brief(search_results)
            brief_max_length = min(max_length or 250, 350)
            prompt = await self._create_brief_search_prompt(content, style, brief_max_length)

            # Get brief summary model profile
            profile = get_model_profile_for_task(
                ModelProfileType.BriefSummary,
                user_id,
                self.user_config
            )

            summary = await self._execute_summarization_with_profile(
                profile, prompt, user_id, tools, grammar
            )

            # Extract brief, action-focused structured data
            key_decisions = await self._extract_brief_decisions(summary, user_id)
            action_items = await self._extract_brief_action_items(summary, user_id)
            conclusions = await self._extract_brief_conclusions(summary, user_id)

            synthesis = SearchTopicSynthesis(
                summary=summary,
                key_topics=conclusions,  # For brief summaries, topics are conclusions
                key_points=key_decisions,  # Key points are decisions
                decisions=key_decisions,
                action_items=action_items,
                sources=[result.url for result in search_results if result.url][:5],  # Limit sources for brevity
                created_at=datetime.datetime.now(datetime.timezone.utc),
                synthesis_type="brief_actionable"
            )

            self.logger.info(
                "Generated brief search synthesis",
                user_id=user_id,
                summary_length=len(summary),
                decisions_count=len(key_decisions),
                actions_count=len(action_items),
            )

            return synthesis

        except Exception as e:
            self.logger.error(
                "Failed to generate brief search synthesis",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Brief search synthesis failed: {e}")

    async def summarize_conversation_brief(
        self,
        messages: List[Message],
        user_id: str,
        conversation_id: int,
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.CONCISE,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> Summary:
        """
        Create brief summary of conversation focusing on decisions and outcomes.

        Args:
            messages: Conversation messages to summarize
            user_id: User identifier for model profile retrieval
            conversation_id: ID of the conversation being summarized
            max_length: Optional maximum summary length
            style: Summary style preference
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            Brief conversation summary focused on actionable outcomes
        """
        try:
            self.logger.info(
                "Generating brief conversation summary",
                user_id=user_id,
                conversation_id=conversation_id,
                messages_count=len(messages),
                style=style,
            )

            # Create conversation text with brief focus on outcomes
            conversation_text = await self._create_brief_conversation_text(messages)
            brief_max_length = min(max_length or 300, 400)

            prompt = await self._create_brief_conversation_prompt(
                conversation_text, style, brief_max_length
            )

            # Get brief summary model profile
            profile = get_model_profile_for_task(
                ModelProfileType.BriefSummary,
                user_id,
                self.user_config
            )

            summary = await self._execute_summarization_with_profile(
                profile, prompt, user_id, tools, grammar
            )

            # Extract brief, action-focused insights
            decisions = await self._extract_brief_decisions(summary, user_id)
            action_items = await self._extract_brief_action_items(summary, user_id)
            conclusions = await self._extract_brief_conclusions(summary, user_id)

            # Create brief conversation summary
            brief_summary = Summary(
                id=f"brief_conv_{conversation_id}_{datetime.datetime.now().isoformat()}",
                user_id=user_id,
                conversation_id=conversation_id,
                content=summary,
                summary_type=SummaryType.BRIEF,
                style=style,
                key_points=decisions,  # Key points are decisions for brief summaries
                topics=conclusions,    # Topics are conclusions for brief summaries
                word_count=len(summary.split()),
                original_length=len(conversation_text.split()),
                compression_ratio=len(summary.split()) / len(conversation_text.split()),
                created_at=datetime.datetime.now(datetime.timezone.utc),
                metadata={
                    "agent_type": "brief",
                    "focus": "decisions_and_actions",
                    "summary_nature": "action_oriented",
                    "decisions": decisions,
                    "action_items": action_items,
                    "conclusions": conclusions,
                }
            )

            # Store the brief conversation summary
            await self.summary_storage.create_summary(brief_summary)

            self.logger.info(
                "Generated brief conversation summary",
                user_id=user_id,
                conversation_id=conversation_id,
                summary_id=brief_summary.id,
                content_length=len(summary),
                decisions_count=len(decisions),
                actions_count=len(action_items),
            )

            return brief_summary

        except Exception as e:
            self.logger.error(
                "Failed to generate brief conversation summary",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Brief conversation summarization failed: {e}")

    async def consolidate_summaries_brief(
        self,
        summaries: List[Summary],
        user_id: str,
        focus_area: str = "outcomes",
        max_length: Optional[int] = None,
        grammar: Optional[Any] = None,
    ) -> Summary:
        """
        Consolidate multiple summaries into a brief, action-oriented overview.

        Args:
            summaries: List of summaries to consolidate briefly
            user_id: User identifier for model profile retrieval
            focus_area: Area to focus on (outcomes, decisions, actions)
            max_length: Optional maximum summary length
            grammar: Optional grammar constraints for structured output

        Returns:
            Brief consolidated summary focusing on key outcomes
        """
        try:
            self.logger.info(
                "Consolidating summaries into brief overview",
                user_id=user_id,
                summaries_count=len(summaries),
                focus_area=focus_area,
            )

            prompt = await self._create_brief_consolidation_prompt(
                summaries, focus_area, max_length
            )

            # Get brief summary model profile
            profile = get_model_profile_for_task(
                ModelProfileType.BriefSummary,
                user_id,
                self.user_config
            )

            consolidated_content = await self._execute_summarization_with_profile(
                profile, prompt, user_id, grammar=grammar
            )

            # Extract and consolidate action-oriented data
            all_decisions = []
            all_actions = []
            for summary in summaries:
                if hasattr(summary, 'metadata') and summary.metadata:
                    all_decisions.extend(summary.metadata.get('decisions', []))
                    all_actions.extend(summary.metadata.get('action_items', []))

            # Consolidate briefly - only the most important items
            key_decisions = await self._consolidate_brief_decisions(all_decisions, user_id)
            key_actions = await self._consolidate_brief_actions(all_actions, user_id)
            
            # Create brief consolidated summary
            brief_consolidated = Summary(
                id=f"brief_consolidated_{focus_area}_{datetime.datetime.now().isoformat()}",
                user_id=user_id,
                content=consolidated_content,
                summary_type=SummaryType.BRIEF,
                style=SummaryStyle.CONCISE,
                key_points=key_decisions,
                topics=[focus_area],  # Single focus topic for brief summary
                word_count=len(consolidated_content.split()),
                created_at=datetime.datetime.now(datetime.timezone.utc),
                metadata={
                    "agent_type": "brief",
                    "consolidation_focus": focus_area,
                    "source_summaries_count": len(summaries),
                    "focus": "brief_actionable_outcomes",
                    "decisions": key_decisions,
                    "action_items": key_actions,
                }
            )

            # Store the brief consolidated summary
            await self.summary_storage.create_summary(brief_consolidated)

            self.logger.info(
                "Generated brief consolidated summary",
                user_id=user_id,
                summary_id=brief_consolidated.id,
                content_length=len(consolidated_content),
                key_decisions_count=len(key_decisions),
            )

            return brief_consolidated

        except Exception as e:
            self.logger.error(
                "Failed to consolidate summaries briefly",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Brief summary consolidation failed: {e}")

    async def _create_brief_text_prompt(
        self, text: str, style: SummaryStyle, max_length: int
    ) -> str:
        """Create specialized prompt for brief text summarization."""
        style_instruction = self._get_brief_style_instruction(style)
        
        return f"""As a decision-focused analyst, create a brief summary that consolidates key decisions, conclusions, and actionable outcomes.

BRIEF SUMMARY REQUIREMENTS:
- Focus on consolidating key decisions, conclusions, and actionable outcomes
- Highlight what was decided, concluded, or should be acted upon
- Eliminate background information and focus on results and next steps
- Keep it concise and action-oriented
- Maximum {max_length} words

CONTENT TO SUMMARIZE:
{text}

STYLE: {style_instruction}

Create a brief summary focusing on decisions, conclusions, and actionable outcomes only."""

    async def _create_brief_search_prompt(
        self, content: str, style: SummaryStyle, max_length: int
    ) -> str:
        """Create specialized prompt for brief search results synthesis."""
        style_instruction = self._get_brief_style_instruction(style)
        
        return f"""As a results-focused analyst, create a brief synthesis that consolidates key findings, decisions, and actionable insights from these search results.

BRIEF SEARCH SYNTHESIS REQUIREMENTS:
- Focus on actionable findings and clear conclusions
- Highlight key decisions or recommendations found in the sources
- Identify what actions should be taken based on the information
- Eliminate redundancy and focus on outcomes
- Maximum {max_length} words

SEARCH RESULTS TO SYNTHESIZE:
{content}

STYLE: {style_instruction}

Create a brief synthesis focusing on decisions, conclusions, and actionable outcomes from the search results."""

    async def _create_brief_conversation_prompt(
        self, conversation_text: str, style: SummaryStyle, max_length: int
    ) -> str:
        """Create specialized prompt for brief conversation summarization."""
        style_instruction = self._get_brief_style_instruction(style)
        
        return f"""As a outcomes-focused analyst, create a brief summary that consolidates the key decisions, conclusions, and actionable outcomes from this conversation.

BRIEF CONVERSATION SUMMARY REQUIREMENTS:
- Focus on what was decided, agreed upon, or concluded
- Highlight action items and next steps
- Capture key outcomes and resolutions
- Eliminate discussion process and focus on results
- Maximum {max_length} words

CONVERSATION TO SUMMARIZE:
{conversation_text}

STYLE: {style_instruction}

Create a brief summary focusing on decisions, conclusions, and actionable outcomes from the conversation."""

    async def _create_brief_consolidation_prompt(
        self, summaries: List[Summary], focus_area: str, max_length: Optional[int]
    ) -> str:
        """Create specialized prompt for brief summary consolidation."""
        length_constraint = f"Maximum {max_length} words." if max_length else "Keep it concise."
        
        summaries_text = "\n\n".join([
            f"Summary {i+1}:\n{s.content}"
            for i, s in enumerate(summaries)
        ])
        
        return f"""As a consolidation expert, create a brief overview that consolidates key decisions, conclusions, and actionable outcomes from these summaries.

BRIEF CONSOLIDATION REQUIREMENTS:
- Focus on consolidating key decisions, conclusions, and actionable outcomes
- Eliminate redundancy and focus on unique outcomes
- Highlight the most important decisions and actions
- Focus area: {focus_area}
- {length_constraint}

SUMMARIES TO CONSOLIDATE:
{summaries_text}

Create a brief consolidation focusing on decisions, conclusions, and actionable outcomes with emphasis on {focus_area}."""

    async def _create_brief_conversation_text(self, messages: List[Message]) -> str:
        """Create conversation text optimized for brief analysis."""
        # Focus on messages that contain decisions, conclusions, or actions
        decision_keywords = ['decide', 'conclude', 'agree', 'determine', 'resolve', 'action', 'next step', 'will', 'should', 'must']
        
        relevant_messages = []
        for msg in messages:
            text = extract_message_text(msg).lower()
            if any(keyword in text for keyword in decision_keywords):
                relevant_messages.append(f"{msg.role}: {extract_message_text(msg)}")
        
        # If no decision-focused messages found, include all but keep it concise
        if not relevant_messages:
            return "\n".join([f"{msg.role}: {extract_message_text(msg)}" for msg in messages[-10:]])  # Last 10 messages
        
        return "\n".join(relevant_messages)

    async def _combine_search_content_brief(self, search_results: List[SearchResult]) -> str:
        """Combine search results content optimized for brief analysis."""
        content_parts = []
        for i, result in enumerate(search_results[:3], 1):  # Limit to top 3 for brevity
            content_parts.append(f"Source {i}: {result.title or 'No title'}")
            # Focus on snippets for brief analysis
            if result.snippet:
                content_parts.append(f"Key finding: {result.snippet}")
            elif result.content:
                # Take first 100 words of content for brief analysis
                words = result.content.split()[:100]
                content_parts.append(f"Content: {' '.join(words)}")
            content_parts.append("---")
        
        return "\n".join(content_parts)

    def _get_brief_style_instruction(self, style: SummaryStyle) -> str:
        """Get style-specific instruction for brief summaries."""
        style_instructions = {
            SummaryStyle.CONCISE: "Use clear, direct language focusing on outcomes.",
            SummaryStyle.DETAILED: "Provide specific details about decisions and actions only.",
            SummaryStyle.BULLET_POINTS: "Use bullet points for decisions and action items.",
            SummaryStyle.NARRATIVE: "Create a brief narrative focused on outcomes and next steps.",
        }
        return style_instructions.get(style, "Use clear, action-oriented language.")

    async def _store_brief_summary(
        self,
        summary: str,
        user_id: str,
        original_text: str,
        summary_type: SummaryType,
        style: SummaryStyle,
        decisions: List[str],
        action_items: List[str],
        conclusions: List[str],
    ) -> None:
        """Store brief summary with action-oriented metadata."""
        try:
            summary_obj = Summary(
                id=f"brief_{datetime.datetime.now().isoformat()}_{user_id}",
                user_id=user_id,
                content=summary,
                summary_type=summary_type,
                style=style,
                key_points=decisions,
                topics=conclusions,
                word_count=len(summary.split()),
                original_length=len(original_text.split()),
                compression_ratio=len(summary.split()) / len(original_text.split()),
                created_at=datetime.datetime.now(datetime.timezone.utc),
                metadata={
                    "agent_type": "brief",
                    "focus": "decisions_and_actions",
                    "summary_nature": "action_oriented",
                    "decisions": decisions,
                    "action_items": action_items,
                    "conclusions": conclusions,
                }
            )
            
            await self.summary_storage.create_summary(summary_obj)
            
        except Exception as e:
            self.logger.warning(
                "Failed to store brief summary",
                user_id=user_id,
                error=str(e),
            )

    async def _execute_summarization_with_profile(
        self,
        profile: ModelProfile,
        prompt: str,
        user_id: str,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """Execute summarization using the brief summary profile."""
        try:
            # Use the base agent's streaming interface
            messages = [prompt]
            response_chunks = []
            
            async for chunk in self.stream(
                messages=messages,
                priority=PipelinePriority.NORMAL,
                tools=tools,
                grammar=grammar,
            ):
                if chunk.message and chunk.message.content:
                    for content in chunk.message.content:
                        if hasattr(content, 'text') and content.text:
                            response_chunks.append(content.text)
            
            return "".join(response_chunks).strip()
            
        except Exception as e:
            self.logger.error(
                "Brief summarization execution failed",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Failed to execute brief summarization: {e}")

    # Brief-specific extraction methods
    async def _extract_brief_decisions(self, summary: str, user_id: str) -> List[str]:
        """Extract decisions with brief, action-focused analysis."""
        decision_indicators = ['decided', 'determined', 'concluded', 'agreed', 'resolved', 'chosen', 'selected']
        sentences = summary.split('. ')
        decisions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in decision_indicators):
                # Keep it brief - truncate long sentences
                brief_decision = sentence.strip()[:150] + ('...' if len(sentence.strip()) > 150 else '')
                decisions.append(brief_decision + ('.' if not brief_decision.endswith('.') else ''))
        return decisions[:3]  # Max 3 for brief summaries

    async def _extract_brief_action_items(self, summary: str, user_id: str) -> List[str]:
        """Extract action items with brief, focused analysis."""
        action_indicators = ['will', 'should', 'must', 'need to', 'plan to', 'action required', 'next step']
        sentences = summary.split('. ')
        actions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in action_indicators):
                # Keep it brief and actionable
                brief_action = sentence.strip()[:150] + ('...' if len(sentence.strip()) > 150 else '')
                actions.append(brief_action + ('.' if not brief_action.endswith('.') else ''))
        return actions[:3]  # Max 3 for brief summaries

    async def _extract_brief_conclusions(self, summary: str, user_id: str) -> List[str]:
        """Extract conclusions with brief, outcome-focused analysis."""
        conclusion_indicators = ['conclude', 'result', 'outcome', 'finding', 'therefore', 'thus', 'finally', 'ultimately']
        sentences = summary.split('. ')
        conclusions = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in conclusion_indicators):
                # Keep it brief and outcome-focused
                brief_conclusion = sentence.strip()[:150] + ('...' if len(sentence.strip()) > 150 else '')
                conclusions.append(brief_conclusion + ('.' if not brief_conclusion.endswith('.') else ''))
        return conclusions[:3]  # Max 3 for brief summaries

    async def _consolidate_brief_decisions(self, all_decisions: List[str], user_id: str) -> List[str]:
        """Consolidate decisions for brief summary - only the most important."""
        if not all_decisions:
            return []
        
        # Deduplicate and prioritize by length (assuming longer = more detailed/important)
        unique_decisions = list(set(all_decisions))
        unique_decisions.sort(key=len, reverse=True)
        return unique_decisions[:2]  # Only top 2 for brief consolidation

    async def _consolidate_brief_actions(self, all_actions: List[str], user_id: str) -> List[str]:
        """Consolidate action items for brief summary - only the most critical."""
        if not all_actions:
            return []
        
        # Deduplicate and prioritize urgent actions
        urgent_keywords = ['urgent', 'immediate', 'asap', 'priority', 'critical', 'must']
        unique_actions = list(set(all_actions))
        
        # Sort by urgency first, then by length
        urgent_actions = [action for action in unique_actions 
                         if any(keyword in action.lower() for keyword in urgent_keywords)]
        other_actions = [action for action in unique_actions 
                        if not any(keyword in action.lower() for keyword in urgent_keywords)]
        
        # Return urgent actions first, then others, max 3 total
        consolidated = urgent_actions[:2] + other_actions[:1]
        return consolidated[:3]