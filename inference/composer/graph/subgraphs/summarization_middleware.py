"""Summarization middleware with background event loop support.

This implementation preserves the LangChain ``AgentMiddleware`` synchronous
``before_model`` interface while safely executing async summarization logic by
offloading it to a dedicated background asyncio event loop/thread.

Avoids: ``asyncio.run`` within a running loop (prior RuntimeError).
Pattern: ``asyncio.run_coroutine_threadsafe`` against a singleton loop.
"""

import uuid
import asyncio
import threading
from collections.abc import Callable, Iterable
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
from langgraph.runtime import Runtime


from composer.agents.base_agent import BaseAgent
from utils.logging import llmmllogger
from utils.message_conversion import extract_text_from_message

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

DEFAULT_SUMMARY_PROMPT = """<role>
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
</instructions>

The user will message you with the full message history you'll be extracting context from, to then replace. Carefully read over it all, and think deeply about what information is most important to your overall goal that should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501

SUMMARY_PREFIX = "## Previous conversation summary:"

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


_SUMMARY_LOOP: asyncio.AbstractEventLoop | None = None
_SUMMARY_THREAD: threading.Thread | None = None


def _ensure_summary_loop() -> asyncio.AbstractEventLoop:
    global _SUMMARY_LOOP, _SUMMARY_THREAD
    if _SUMMARY_LOOP is None:
        loop = asyncio.new_event_loop()
        _SUMMARY_LOOP = loop

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run_loop, name="summary-loop", daemon=True)
        _SUMMARY_THREAD = t
        t.start()
    return _SUMMARY_LOOP


from typing import Coroutine


def _run_coroutine_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = _ensure_summary_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


class SummarizationMiddleware(AgentMiddleware):
    """Middleware that summarizes conversation history when token limits are approached.

    Executes async summary generation on a background event loop to remain
    compatible with synchronous LangChain middleware contract.
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_tokens_before_summary: int | None = None,
        messages_to_keep: int = _DEFAULT_MESSAGES_TO_KEEP,
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        summary_prefix: str = SUMMARY_PREFIX,
    ) -> None:
        """Initialize the summarization middleware.

        Args:
            model: The language model to use for generating summaries.
            max_tokens_before_summary: Token threshold to trigger summarization.
                If `None`, summarization is disabled.
            messages_to_keep: Number of recent messages to preserve after summarization.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            summary_prefix: Prefix added to system message when including summary.
        """
        super().__init__()

        self.agent = agent
        self.max_tokens_before_summary = max_tokens_before_summary
        self.messages_to_keep = messages_to_keep
        self.token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.summary_prefix = summary_prefix
        self.logger = llmmllogger.bind(component="SummarizationMiddleware")

    def before_model(  # type: ignore[override]
        self,
        state: AgentState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if (
            self.max_tokens_before_summary is not None
            and total_tokens < self.max_tokens_before_summary
        ):
            return None

        self.logger.debug(
            f"Total tokens ({total_tokens}) exceed threshold "
            f"({self.max_tokens_before_summary}), summarizing..."
            f" Keeping last {self.messages_to_keep} messages."
            f" Total messages: {len(messages)}"
        )

        cutoff_index = self._find_safe_cutoff(messages)
        self.logger.debug(f"Determined safe cutoff index at: {cutoff_index}")
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(
            messages, cutoff_index
        )

        summary = _run_coroutine_sync(self._create_summary(messages_to_summarize))
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    def _build_new_messages(self, summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=f"Here is a summary of the conversation to date:\n\n{summary}"
            )
        ]

    def _ensure_message_ids(self, messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs for the add_messages reducer."""
        for msg in messages:
            # Only AI and Tool messages expose id in langchain schema we rely on for reducer behavior
            if hasattr(msg, "id") and getattr(msg, "id") is None:  # type: ignore[attr-defined]
                try:
                    setattr(msg, "id", str(uuid.uuid4()))  # type: ignore[attr-defined]
                except Exception:
                    # Ignore silently if message type is immutable or lacks id
                    continue

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]

        return messages_to_summarize, preserved_messages

    def _find_safe_cutoff(self, messages: list[AnyMessage]) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns 0 if no safe cutoff is found.
        """
        if len(messages) <= self.messages_to_keep:
            return 0

        target_cutoff = len(messages) - self.messages_to_keep

        for i in range(target_cutoff, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _is_safe_cutoff_point(
        self, messages: list[AnyMessage], cutoff_index: int
    ) -> bool:
        """Check if cutting at index would separate AI/Tool message pairs."""
        if cutoff_index >= len(messages):
            return True

        search_start = max(0, cutoff_index - _SEARCH_RANGE_FOR_TOOL_PAIRS)
        search_end = min(len(messages), cutoff_index + _SEARCH_RANGE_FOR_TOOL_PAIRS)

        for i in range(search_start, search_end):
            if not self._has_tool_calls(messages[i]):
                continue

            tool_call_ids = self._extract_tool_call_ids(cast("AIMessage", messages[i]))
            if self._cutoff_separates_tool_pair(
                messages, i, cutoff_index, tool_call_ids
            ):
                return False

        return True

    def _has_tool_calls(self, message: AnyMessage) -> bool:
        """Check if message is an AI message with tool calls."""
        return (
            isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls  # type: ignore[return-value]
        )

    def _extract_tool_call_ids(self, ai_message: AIMessage) -> set[str]:
        """Extract tool call IDs from an AI message."""
        tool_call_ids = set()
        for tc in ai_message.tool_calls:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if call_id is not None:
                tool_call_ids.add(call_id)
        return tool_call_ids

    def _cutoff_separates_tool_pair(
        self,
        messages: list[AnyMessage],
        ai_message_index: int,
        cutoff_index: int,
        tool_call_ids: set[str],
    ) -> bool:
        """Check if cutoff separates an AI message from its corresponding tool messages."""
        for j in range(ai_message_index + 1, len(messages)):
            message = messages[j]
            if (
                isinstance(message, ToolMessage)
                and message.tool_call_id in tool_call_ids
            ):
                ai_before_cutoff = ai_message_index < cutoff_index
                tool_before_cutoff = j < cutoff_index
                if ai_before_cutoff != tool_before_cutoff:
                    return True
        return False

    async def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        prompt = self.summary_prompt.format(messages=trimmed_messages)
        try:
            response = await self.agent.run(prompt)
            assert response.message is not None
            return extract_text_from_message(response.message)
        except Exception as e:  # noqa: BLE001
            return f"Error generating summary: {e!s}"

    def _trim_messages_for_summary(
        self, messages: list[AnyMessage]
    ) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            return trim_messages(
                messages,
                max_tokens=_DEFAULT_TRIM_TOKEN_LIMIT,
                token_counter=self.token_counter,
                start_on="human",
                strategy="last",
                allow_partial=True,
                include_system=True,
            )
        except Exception:  # noqa: BLE001
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
