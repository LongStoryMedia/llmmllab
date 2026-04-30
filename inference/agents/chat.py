"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

import datetime
from typing import (
    Annotated,
    List,
    cast,
)
from pydantic import BaseModel, Field

from core.errors import NodeExecutionError
from agents.base import BaseAgent
from models import (
    MessageRole,
    Message,
    SearchResult,
    SearchTopicSynthesis,
    Summary,
)
from utils.grammar_generator import parse_structured_output
from utils.message_conversion import extract_text_from_message


class TitleResponse(BaseModel):
    title: str


class ChatAgent(BaseAgent):
    """
    Base class for all workflow agents providing common functionality.

    This base class provides:
    - Node metadata injection for workflow tracking
    - Consistent logging setup with component binding
    - Common error handling patterns
    - Shared initialization patterns
    - Generic typing for pipeline execution results

    All agent classes should inherit from this base class to ensure consistent
    behavior across the workflow system.
    """

    async def generate_title(
        self,
        messages: List[Message],
    ) -> str:
        """
        Generate a concise, descriptive title for a conversation based on its messages.

        Args:
            messages: List of conversation messages to analyze

        Returns:
            str: Generated conversation title (2-6 words)

        """

        try:
            # Only collect last 5 User/Assistant messages, and concatenate consecutive messages of the same role
            filtered = [
                m
                for m in messages
                if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
            ]
            last_msgs = filtered[-5:] if len(filtered) > 5 else filtered

            # Concatenate consecutive messages of the same role
            conversation_blocks = []
            current_role = None
            current_text = ""
            for msg in last_msgs:
                text = extract_text_from_message(msg)
                if not text.strip():
                    continue
                role = (
                    MessageRole.USER
                    if msg.role == MessageRole.USER
                    else MessageRole.ASSISTANT
                )
                if role == current_role:
                    current_text += f" {text}"  # Concatenate with space
                else:
                    if current_text and current_role:
                        conversation_blocks.append(
                            f"{current_role.value}: {current_text.strip()}"
                        )
                    current_role = role
                    current_text = text
            if current_text:
                conversation_blocks.append(f"{current_role}: {current_text.strip()}")

            conversation_text = "\n".join(conversation_blocks)

            if not conversation_text.strip():
                return "New Conversation"
            title_prompt = f"""
/no_think
Generate a concise, descriptive title for this conversation. The title should:
- Be 2-6 words maximum
- Capture the main topic or purpose
- Be clear and professional
- Not include quotes or special characters
- Be suitable as a conversation label

Conversation:
{conversation_text}
"""

            result = await self.run(
                title_prompt,
                grammar=TitleResponse,
            )

            txt = (
                extract_text_from_message(result.message)
                if result and result.message
                else ""
            )
            assert txt.strip(), "Empty title generation response"

            intents = parse_structured_output(txt, TitleResponse)
            return intents.title

        except Exception as e:
            self.logger.error(
                "Title generation failed", error=str(e), context="title_generation"
            )
            # Provide fallback title instead of raising error
            return "Conversation"

    async def summarize_conversation(
        self,
        messages: List[Message] | List[Summary],
        level: int = 1,
    ) -> Summary:
        """
        Create primary summary of conversation messages.

        Args:
            messages: Conversation messages to summarize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            Comprehensive primary conversation summary
        """
        try:
            self.logger.info(
                "Generating primary conversation summary",
                messages_count=len(messages),
            )

            prompt = ""

            if level == 1:
                messages = cast(List[Message], messages)
                prompt = await self._create_summary_prompt(
                    [
                        f"### [{msg.role}]:\n{extract_text_from_message(msg)}\n\n---\n\n"
                        for msg in messages
                    ],
                )
            else:
                messages = cast(List[Summary], messages)
                prompt = await self._create_summary_prompt(
                    [
                        f"### [From messages {', '.join(str(sid) for sid in msg.source_ids)} {msg.created_at}]:\n{msg.content}\n\n---\n\n"
                        for msg in messages
                    ],
                )

            res = await self.run(prompt)
            assert res.message is not None, "No message returned from summarization"
            summary_text = f"Here is a summary of the conversation to date:\n\n{extract_text_from_message(res.message)}"

            summary = Summary(
                id=-1,  # Placeholder ID; to be set by storage layer
                content=summary_text,
                level=level,
                source_ids=[
                    msg.id
                    for msg in messages
                    if hasattr(msg, "id") and msg.id is not None
                ],
                created_at=datetime.datetime.now(datetime.timezone.utc),
                conversation_id=getattr(messages[0], "conversation_id", -1) or -1,
            )

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
            raise RuntimeError(f"Primary conversation summarization failed: {e}") from e

    async def _create_summary_prompt(
        self,
        messages: List[str],
    ) -> str:
        """Create specialized prompt for primary conversation summarization."""
        return f"""<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step. Because of this, you must do your very best to extract and record all of the most important context from the conversation history.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.
- Trace the evolution of topics and ideas throughout the conversation
- Identify key decision points and their rationale
- Highlight agreements, disagreements, and resolution processes
- Capture the flow of reasoning and argumentation
- Focus on logical progression and development of concepts
</instructions>

The user will message you with the full message history you'll be extracting context from, to then replace. Carefully read over it all, and think deeply about what information is most important to your overall goal that should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>
"""

    async def summarize_search_results(
        self,
        search_results: List[SearchResult],
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
            )

            # Combine search content with primary focus
            content = await self._combine_search_content(search_results)
            prompt = await self._create_primary_search_prompt(content)

            class SearchSynthesisOutput(BaseModel):
                """
                Grammar for primary search synthesis output.
                """

                topics: Annotated[
                    List[str],
                    Field(
                        ...,
                        description="List of key topics discussed in the synthesis",
                        min_length=1,
                        max_length=10,
                    ),
                ]
                summary: Annotated[
                    str,
                    Field(
                        ..., description="Comprehensive synthesis of the search results"
                    ),
                ]

            response = await self.run(messages=prompt, grammar=SearchSynthesisOutput)
            assert response.message is not None
            summary = parse_structured_output(
                extract_text_from_message(response.message), SearchSynthesisOutput
            )

            # Collect URLs from search results
            urls = []
            for result in search_results:
                if result.contents:
                    for content in result.contents:
                        if hasattr(content, "url") and content.url:
                            urls.append(content.url)

            synthesis = SearchTopicSynthesis(
                urls=urls,
                topics=summary.topics,
                synthesis=summary.summary,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                conversation_id=0,
            )

            self.logger.info(
                "Generated primary search synthesis",
                sources_count=len(synthesis.urls),
            )

            return synthesis

        except Exception as e:
            self.logger.error(
                "Failed to generate primary search synthesis",
                error=str(e),
            )
            raise NodeExecutionError(f"Primary search synthesis failed: {e}") from e

    async def _create_primary_search_prompt(self, content: str) -> str:
        """Create specialized prompt for primary search results synthesis."""

        return f"""<role>
Research Synthesis Expert
</role>

<primary_objective>
Your objective in this task is to create a comprehensive primary analysis of these search results focusing on logical progression of information.
You will also provide a short list of key topics discussed in the synthesis.
</primary_objective>

<objective_information>
You are to synthesize the search results provided below, ensuring that you capture the logical progression and evolution of topics and ideas across all sources.
This synthesis should not merely summarize individual results, but rather weave them together into a coherent narrative that reflects the development of themes and concepts.
You should identify patterns, trends, and relationships between the findings of different sources, highlighting areas of consensus as well as contradictions.
</objective_information>

<instructions>
When synthesizing the search results, ensure you:
- Synthesize information across all sources with analytical depth
- Identify patterns, trends, and relationships between findings
- Highlight consensus and contradictions in the information
- Provide comprehensive coverage of key themes
- Focus on logical progression and evolution of topics
- Once complete, provide a short list of key topics discussed in the synthesis
</instructions>

<search_results>
Search results to synthesize:
{content}
</search_results>
"""

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
