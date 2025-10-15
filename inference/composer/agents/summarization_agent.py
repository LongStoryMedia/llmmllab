"""
Summarization Agent for content summarization and synthesis.
Provides core business logic for text summarization and content processing.
"""

import datetime
from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING

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


class SummarizationAgent(BaseAgent[str]):
    """
    Summarization Agent for content summarization with grammar-constrained output.

    Provides core business logic for summarizing text content, conversation history,
    and search results using configured summarization models.
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
        Initialize summarization agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating summarization pipelines
            profile: Model profile for summarization operations
            node_metadata: Node execution metadata for tracking
            summary_storage: Injected summary storage service
            search_storage: Injected search storage service
            user_config: User configuration object
        """
        super().__init__(pipeline_factory, profile, node_metadata, "SummarizationAgent")
        self.summary_storage = summary_storage
        self.search_storage = search_storage
        self.user_config = user_config

    async def summarize_text(
        self,
        text: str,
        user_id: str,
        summary_type: SummaryType = SummaryType.PRIMARY,
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.CONCISE,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Summarize input text using configured summarization model.

        Args:
            text: Text content to summarize
            user_id: User identifier for model profile retrieval
            summary_type: Type of summary (general, technical, creative, etc.)
            max_length: Optional maximum summary length
            style: Summary style (concise, detailed, bullet_points, etc.)
            tools: Optional tools available to the agent for enhanced capabilities
            grammar: Optional grammar constraints for structured output

        Returns:
            Summarized text content
        """
        try:
            self.logger.info(
                "Generating text summary",
                user_id=user_id,
                text_length=len(text),
                summary_type=summary_type,
                style=style,
                has_tools=bool(tools),
                has_grammar=bool(grammar),
            )

            # Use unified execution pipeline
            return await self._execute_with_unified_pipeline(
                user_id=user_id,
                summary_type=summary_type,
                prompt_creator=self._create_summarization_prompt,
                prompt_args={
                    "text": text,
                    "summary_type": summary_type,
                    "style": style,
                    "max_length": max_length,
                },
                tools=tools,
                grammar=grammar,
            )

        except Exception as e:
            self.logger.error(
                "Text summarization failed",
                user_id=user_id,
                error=str(e),
                text_length=len(text),
            )
            raise NodeExecutionError(f"Text summarization failed: {e}") from e

    async def summarize_search_results(
        self,
        search_results: List[SearchResult],
        user_id: str,
        conversation_id: int,
        query: str,
        max_length: int,
    ) -> SearchTopicSynthesis:
        """
        Summarize and synthesize web search results into coherent response.

        Args:
            search_results: List of search result dictionaries
            user_id: User identifier
            query: Original search query for context
            focus_areas: Optional list of specific areas to focus on

        Returns:
            Synthesized summary with metadata
        """
        try:
            self.logger.info(
                "Summarizing search results",
                user_id=user_id,
                result_count=len(search_results),
                query=query[:100],
            )

            if not search_results:
                raise NodeExecutionError("No search results available to summarize.")

            # Extract and combine content from search results
            combined_content = await self._combine_search_content(
                search_results,
                query,
                max_length,
            )

            # Generate comprehensive summary using unified pipeline
            summary = await self._execute_with_unified_pipeline(
                user_id=user_id,
                summary_type=SummaryType.PRIMARY,
                prompt_creator=self._create_search_summary_prompt,
                prompt_args={
                    "content": combined_content,
                    "query": query,
                },
            )

            # Extract key points and metadata
            key_points = await self._extract_key_points(summary, user_id)
            sources = []
            for result in search_results:
                if result.contents:
                    for content in result.contents:
                        if content.url and content.url not in sources:
                            sources.append(content.url)

            synth = SearchTopicSynthesis(
                synthesis=summary,
                topics=key_points,
                urls=sources,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                conversation_id=conversation_id,
            )

            # Use injected storage service
            search_svc = self.search_storage

            synth.id = await search_svc.create(synth)

            return synth

        except Exception as e:
            self.logger.error(
                "Search results summarization failed", user_id=user_id, error=str(e)
            )
            raise NodeExecutionError(f"Search results summarization failed: {e}") from e

    async def summarize_conversation(
        self,
        messages: List[Message],
        user_id: str,
        conversation_id: int,
        summary_type: SummaryType = SummaryType.MASTER,
    ) -> Summary:
        """
        Summarize conversation history highlighting key points and decisions.

        Args:
            messages: List of conversation messages
            user_id: User identifier
            focus: Focus area (key_decisions, topics, action_items, etc.)

        Returns:
            Conversation summary with structured output
        """
        try:
            self.logger.info(
                "Summarizing conversation",
                user_id=user_id,
                message_count=len(messages),
                summary_type=summary_type,
            )

            if not messages:
                raise NodeExecutionError("No messages available to summarize.")

            # Convert messages to text for summarization
            conversation_text = self._format_conversation_for_summary(messages)

            # Generate conversation summary using unified pipeline
            summary = await self._execute_with_unified_pipeline(
                user_id=user_id,
                summary_type=summary_type,
                prompt_creator=self._create_conversation_summary_prompt,
                prompt_args={
                    "conversation_text": conversation_text,
                    "summary_type": summary_type,
                },
            )

            # Use injected storage service
            summary_svc = self.summary_storage

            summ = Summary(
                content=summary,
                level=1,
                conversation_id=conversation_id,
                source_ids=[message.id for message in messages if message.id],
                id=-1,  # Placeholder, will be set on storage create
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )
            # Extract structured elements
            summary_id = await summary_svc.create_summary(summ)
            assert summary_id is not None
            summ.id = summary_id

            self.logger.info(
                "Conversation summarized successfully",
                user_id=user_id,
                summary_length=len(summary),
            )

            return summ

        except Exception as e:
            self.logger.error(
                "Conversation summarization failed", user_id=user_id, error=str(e)
            )
            raise NodeExecutionError(f"Conversation summarization failed: {e}") from e

    async def consolidate_summaries(
        self,
        summaries: List[Summary],
        user_id: str,
        conversation_id: int,
        summary_type: SummaryType = SummaryType.MASTER,
        level: int = 1,
        target_level: int = 2,
        grammar: Optional[Any] = None,
    ) -> Summary:
        """
        Consolidate multiple summaries into a higher-level summary.

        This method implements hierarchical summarization as per the context extension
        architecture, enabling multi-level compression of conversation history.

        Args:
            summaries: List of summary texts to consolidate
            user_id: User identifier for model profile retrieval
            level: Current level of summaries being consolidated
            target_level: Target level for the new consolidated summary
            focus: Focus area for consolidation (consolidation, key_themes, decisions)
            grammar: Optional grammar constraints for structured output

        Returns:
            Consolidated summary text
        """
        try:
            self.logger.info(
                "Consolidating summaries",
                user_id=user_id,
                summary_count=len(summaries),
                from_level=level,
                to_level=target_level,
                has_grammar=bool(grammar),
            )

            if not summaries:
                raise NodeExecutionError("No summaries available to consolidate.")

            if len(summaries) == 1:
                return summaries[0]

            # Prepare consolidation content
            consolidation_content = self._format_summaries_for_consolidation(
                summaries, level, target_level
            )

            # Execute consolidation using unified pipeline
            consolidated_summary = await self._execute_with_unified_pipeline(
                user_id=user_id,
                summary_type=summary_type,
                prompt_creator=self._create_consolidation_prompt,
                prompt_args={
                    "content": consolidation_content,
                    "level": level,
                    "target_level": target_level,
                    "summary_type": summary_type,
                },
                grammar=grammar,
            )

            self.logger.info(
                "Summaries consolidated successfully",
                user_id=user_id,
                input_summaries=len(summaries),
                output_length=len(consolidated_summary),
                target_level=target_level,
            )

            # Use injected storage service
            summary_svc = self.summary_storage

            summ = Summary(
                content=consolidated_summary,
                level=target_level,
                conversation_id=conversation_id,
                source_ids=[summary.id for summary in summaries if summary.id],
                id=-1,  # Placeholder, will be set on storage create
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )
            # Extract structured elements
            sumid = await summary_svc.create_summary(summ)
            assert sumid is not None
            summ.id = sumid

            self.logger.info(
                "Conversation summarized successfully",
                user_id=user_id,
                summary_length=len(consolidated_summary),
            )

            return summ

        except Exception as e:
            self.logger.error(
                "Summary consolidation failed",
                user_id=user_id,
                error=str(e),
                summary_count=len(summaries),
            )
            raise NodeExecutionError(f"Summary consolidation failed: {e}") from e

    async def _create_summarization_prompt(
        self,
        text: str,
        summary_type: SummaryType,
        style: SummaryStyle,
        model_profile: ModelProfile,
        max_length: Optional[int],
    ) -> str:
        """Create appropriate summarization prompt based on parameters."""

        length_instruction = (
            f" Keep the summary under {max_length} words." if max_length else ""
        )

        style_instructions = {
            "concise": "Provide a brief, concise summary focusing on the main points.",
            "detailed": "Provide a comprehensive summary with important details and context.",
            "bullet_points": "Summarize using clear bullet points for each main topic.",
            "structured": "Organize the summary with clear sections and headings.",
        }

        type_instructions = {
            "general": "Summarize the following content:",
            "technical": "Provide a technical summary highlighting key concepts, methods, and findings:",
            "creative": "Summarize the creative content focusing on themes, style, and key ideas:",
            "research": "Provide a research-focused summary highlighting methodology, findings, and implications:",
            "conversation": "Summarize this conversation highlighting key points, decisions, and outcomes:",
        }

        instruction = type_instructions.get(summary_type, type_instructions["general"])
        style_instruction = style_instructions.get(style, style_instructions["concise"])

        prompt = f"""{model_profile.system_prompt}

{instruction}

{style_instruction}{length_instruction}

Content to summarize:
{text}

Summary:"""

        return prompt

    async def _create_prompt_with_system_context(
        self,
        instruction: str,
        content: str,
        model_profile: ModelProfile,
        additional_instructions: Optional[List[str]] = None,
    ) -> str:
        """Create prompt with consistent system prompt integration.

        Args:
            instruction: Main instruction for the task
            content: Content to be processed
            model_profile: Model profile containing system prompt
            additional_instructions: Optional additional instruction lines

        Returns:
            Formatted prompt with system prompt integration
        """
        additional_text = ""
        if additional_instructions:
            additional_text = "\n" + "\n".join(additional_instructions)

        prompt = f"""{model_profile.system_prompt}

{instruction}{additional_text}

{content}"""

        return prompt

    async def _create_search_summary_prompt(
        self,
        content: str,
        query: str,
        model_profile: ModelProfile,
    ) -> str:
        """Create prompt for search results summarization with system prompt."""

        instruction = f'Based on the following search results for the query "{query}", provide a comprehensive summary that answers the user\'s question and synthesizes the key information found.'

        additional_instructions = [
            "Please provide:",
            "1. A clear answer to the query",
            "2. Key findings from the sources",
            "3. Important details and context",
            "4. Any conflicting information or limitations",
            "",
            "Summary:",
        ]

        formatted_content = f"Search Results Content:\n{content}"

        return await self._create_prompt_with_system_context(
            instruction, formatted_content, model_profile, additional_instructions
        )

    async def _create_conversation_summary_prompt(
        self, conversation_text: str, focus: str, model_profile: ModelProfile
    ) -> str:
        """Create prompt for conversation summarization with system prompt."""

        focus_instructions = {
            "key_decisions": "Focus on decisions made, conclusions reached, and action items identified.",
            "topics": "Focus on the main topics discussed and key points covered.",
            "action_items": "Focus on tasks, action items, and next steps identified.",
            "outcomes": "Focus on outcomes, results, and conclusions reached.",
        }

        focus_instruction = focus_instructions.get(focus, focus_instructions["topics"])
        instruction = f"Summarize the following conversation. {focus_instruction}"

        additional_instructions = [
            "Provide a structured summary including:",
            "1. Main topics discussed",
            "2. Key decisions or conclusions",
            "3. Action items or next steps",
            "4. Important details and context",
            "",
            "Summary:",
        ]

        formatted_content = f"Conversation:\n{conversation_text}"

        return await self._create_prompt_with_system_context(
            instruction, formatted_content, model_profile, additional_instructions
        )

    async def _combine_search_content(
        self, search_results: List[SearchResult], query: str, max_length: int
    ) -> str:
        """Combine search results into single text for summarization."""

        combined_parts = [f"Search Query: {query}\n"]

        for result in search_results:
            if result.contents:
                for content in result.contents:
                    title = content.title
                    snippet = content.content[:max_length]  # Truncate long content
                    url = content.url

                    part = f"\n--- {title} ---\nSource: {url}\nContent: {snippet}\n"
                    combined_parts.append(part)

        return "\n".join(combined_parts)

    def _format_conversation_for_summary(self, messages: List[Message]) -> str:
        """Format conversation messages for summarization."""

        formatted_parts = []
        for message in messages:
            if message.content:
                formatted_parts.append(
                    f"{message.role.title()}: {extract_message_text(message)}"
                )

        return "\n\n".join(formatted_parts)

    def _parse_bullet_points(self, text: str, max_items: int = 5) -> List[str]:
        """Parse bullet points from LLM response into clean list.

        Args:
            text: LLM response text containing bullet points
            max_items: Maximum number of items to return

        Returns:
            List of parsed bullet point strings
        """
        items = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if line and (
                line.startswith(("•", "-", "*"))
                or any(line.startswith(f"{i}.") for i in range(1, 10))
            ):
                # Clean up the bullet point formatting
                clean_item = line.lstrip("•-*0123456789. ").strip()
                if clean_item:
                    items.append(clean_item)

        return items[:max_items]

    async def _extract_with_llm_and_fallback(
        self,
        content: str,
        user_id: str,
        extraction_prompt: str,
        fallback_fn: Callable[[str, int], List[str]],
        max_items: int = 5,
        max_length: int = 200,
    ) -> List[str]:
        """Generic extraction method with LLM processing and fallback.

        Args:
            content: Content to extract from
            user_id: User identifier
            extraction_prompt: Prompt for LLM extraction
            fallback_fn: Fallback function if LLM extraction fails
            max_items: Maximum items to return
            max_length: Maximum length for LLM response

        Returns:
            List of extracted items
        """
        try:
            # Use LLM for extraction
            extracted_text = await self._execute_summarization(
                text=extraction_prompt,
                user_id=user_id,
                summary_type=SummaryType.KEY_POINTS,
                style=SummaryStyle.BULLET_POINTS,
                max_length=max_length,
            )

            # Parse the LLM response
            return self._parse_bullet_points(extracted_text, max_items)

        except Exception as e:
            self.logger.warning(
                "LLM extraction failed, using fallback method",
                user_id=user_id,
                error=str(e),
            )
            # Use fallback method
            return fallback_fn(content, max_items)

    async def _extract_key_points(self, summary: str, user_id: str) -> List[str]:
        """Extract key points from summary text using unified extraction method."""
        extraction_prompt = f"""Extract the key points from the following summary. List them as bullet points, one key point per line:

{summary}

Key points:"""

        def fallback_key_points(content: str, max_items: int) -> List[str]:
            """Fallback method for key point extraction."""
            return self._parse_bullet_points(content, max_items)

        return await self._extract_with_llm_and_fallback(
            content=summary,
            user_id=user_id,
            extraction_prompt=extraction_prompt,
            fallback_fn=fallback_key_points,
            max_items=5,
            max_length=200,
        )

    async def _extract_topics(self, summary: str, user_id: str) -> List[str]:
        """Extract main topics from summary using unified extraction method."""
        extraction_prompt = f"""Extract the main topics from the following summary. List them as bullet points, one topic per line:

{summary}

Main topics:"""

        def fallback_topics(content: str, max_items: int) -> List[str]:
            """Fallback method for topic extraction."""
            topics = []
            sentences = content.split(".")
            for sentence in sentences[:max_items]:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 100:
                    topics.append(sentence)
            return topics

        return await self._extract_with_llm_and_fallback(
            content=summary,
            user_id=user_id,
            extraction_prompt=extraction_prompt,
            fallback_fn=fallback_topics,
            max_items=5,
            max_length=150,
        )

    async def _extract_decisions(self, summary: str, user_id: str) -> List[str]:
        """Extract decisions from summary using unified extraction method."""
        extraction_prompt = f"""Extract any decisions, conclusions, or resolutions from the following summary. List them as bullet points:

{summary}

Decisions and conclusions:"""

        def fallback_decisions(content: str, max_items: int) -> List[str]:
            """Fallback method for decision extraction using keyword matching."""
            decision_keywords = [
                "decided",
                "concluded",
                "agreed",
                "determined",
                "resolved",
            ]
            decisions = []
            sentences = content.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in decision_keywords):
                    decisions.append(sentence)
                    if len(decisions) >= max_items:
                        break
            return decisions

        return await self._extract_with_llm_and_fallback(
            content=summary,
            user_id=user_id,
            extraction_prompt=extraction_prompt,
            fallback_fn=fallback_decisions,
            max_items=3,
            max_length=150,
        )

    async def _extract_action_items(self, summary: str, user_id: str) -> List[str]:
        """Extract action items from summary using unified extraction method."""
        extraction_prompt = f"""Extract any action items, tasks, or next steps from the following summary. List them as bullet points:

{summary}

Action items and next steps:"""

        def fallback_action_items(content: str, max_items: int) -> List[str]:
            """Fallback method for action item extraction using keyword matching."""
            action_keywords = [
                "will",
                "should",
                "need to",
                "must",
                "action",
                "task",
                "todo",
            ]
            action_items = []
            sentences = content.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in action_keywords):
                    action_items.append(sentence)
                    if len(action_items) >= max_items:
                        break
            return action_items

        return await self._extract_with_llm_and_fallback(
            content=summary,
            user_id=user_id,
            extraction_prompt=extraction_prompt,
            fallback_fn=fallback_action_items,
            max_items=5,
            max_length=200,
        )

    async def _execute_with_unified_pipeline(
        self,
        user_id: str,
        summary_type: SummaryType,
        prompt_creator: Callable,
        prompt_args: Dict[str, Any],
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Unified execution method that handles all database retrieval, model profile resolution, and pipeline execution.

        Args:
            user_id: User identifier for configuration retrieval
            summary_type: Type of summary for model profile selection
            prompt_creator: Callable that creates the prompt (method reference)
            prompt_args: Arguments to pass to the prompt creator
            tools: Optional tools for pipeline execution
            grammar: Optional grammar constraints

        Returns:
            Generated text from the pipeline
        """
        # Map summary types to model profile types
        profile_type_map = {
            SummaryType.PRIMARY: ModelProfileType.PrimarySummary,
            SummaryType.MASTER: ModelProfileType.MasterSummary,
            SummaryType.BRIEF: ModelProfileType.BriefSummary,
            SummaryType.KEY_POINTS: ModelProfileType.KeyPoints,
        }

        # Get model profile for the specified type
        model_profile = await get_model_profile_for_task(
            self.user_config.model_profiles, profile_type_map[summary_type], user_id
        )
        circuit_breaker = (
            model_profile.circuit_breaker or self.user_config.circuit_breaker
        )

        # Create prompt using the provided prompt creator with model profile
        prompt_args["model_profile"] = model_profile
        prompt = await prompt_creator(**prompt_args)

        # Execute with unified pipeline logic
        return await self._execute_summarization_with_profile(
            prompt=prompt,
            model_profile=model_profile,
            circuit_breaker=circuit_breaker,
            tools=tools,
            grammar=grammar,
        )

    async def _execute_summarization(
        self,
        text: str,
        user_id: str,
        summary_type: SummaryType,
        style: SummaryStyle = SummaryStyle.CONCISE,
        max_length: Optional[int] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Execute summarization using unified pipeline method.

        Maintained for backward compatibility with extraction methods.
        """
        return await self._execute_with_unified_pipeline(
            user_id=user_id,
            summary_type=summary_type,
            prompt_creator=self._create_summarization_prompt,
            prompt_args={
                "text": text,
                "summary_type": summary_type,
                "style": style,
                "max_length": max_length,
            },
            grammar=grammar,
        )

    async def _execute_summarization_with_profile(
        self,
        prompt: str,
        circuit_breaker: Any,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Execute summarization with given model profile and parameters.

        Extracted common pipeline execution logic to ensure consistent
        behavior across all summarization methods.
        """
        return await self.run(
            messages=prompt,
            circuit_breaker=circuit_breaker,
            tools=tools,
            grammar=grammar,
        )

    async def _execute_summarization_pipeline(
        self,
        prompt: str,
        model_profile: ModelProfile,
        circuit_breaker: Any,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """Internal executor for summarization pipeline operation."""
        with self.pipeline_factory.pipeline(
            model_profile, str, PipelinePriority.NORMAL, circuit_breaker
        ) as pipeline:
            res = await run_pipeline(prompt, pipeline, tools=tools, grammar=grammar)
            return extract_message_text(res.message) if res and res.message else ""

    def _format_summaries_for_consolidation(
        self, summaries: List[Summary], current_level: int, target_level: int
    ) -> str:
        """
        Format multiple summaries for consolidation processing.

        Args:
            summaries: List of summary texts to format
            current_level: Current summarization level
            target_level: Target consolidation level

        Returns:
            Formatted text ready for consolidation
        """
        formatted_parts = [
            f"Consolidating {len(summaries)} Level {current_level} summaries into Level {target_level}:\n"
        ]

        for i, summary in enumerate(summaries, 1):
            formatted_parts.append(f"--- Summary {i} ---\n{summary.content}\n")

        return "\n".join(formatted_parts)

    async def _create_consolidation_prompt(
        self,
        content: str,
        level: int,
        target_level: int,
        summary_type: SummaryType,
        model_profile: ModelProfile,
    ) -> str:
        """
        Create prompt for summary consolidation with system prompt.

        Args:
            content: Formatted summaries content
            level: Current level being consolidated
            target_level: Target consolidation level
            focus: Focus area for consolidation
            model_profile: Model profile containing system prompt

        Returns:
            Consolidation prompt text with system context
        """
        focus_instructions = {
            SummaryType.MASTER: "Focus on combining and synthesizing the key information from all summaries into a coherent, comprehensive overview.",
            SummaryType.KEY_POINTS: "Focus on identifying and consolidating the main themes, patterns, and recurring topics across the summaries.",
            SummaryType.BRIEF: "Focus on consolidating key decisions, conclusions, and actionable outcomes from the summaries.",
            SummaryType.PRIMARY: "Focus on the logical progression and evolution of topics and ideas across the summaries.",
        }

        focus_instruction = focus_instructions.get(
            summary_type, focus_instructions[summary_type]
        )
        instruction = f"Consolidate the following Level {level} summaries into a single Level {target_level} summary. {focus_instruction}"

        additional_instructions = [
            "The consolidated summary should:",
            "1. Preserve the most important information from each source summary",
            "2. Eliminate redundancy and overlap between summaries",
            "3. Maintain logical flow and coherence",
            "4. Be more concise than the combined source summaries while retaining essential details",
            "",
            f"Consolidated Level {target_level} Summary:",
        ]

        return await self._create_prompt_with_system_context(
            instruction, content, model_profile, additional_instructions
        )
