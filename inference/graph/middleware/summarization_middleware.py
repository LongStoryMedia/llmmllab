"""Summarization middleware."""

import uuid
from collections.abc import Callable, Iterable
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


from utils.message_conversion import lc_messages_to_messages
from utils.logging import llmmllogger

from agents.chat import ChatAgent


TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]


_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


class SummarizationMiddleware(AgentMiddleware):
    """Middleware that summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.
    """

    def __init__(
        self,
        agent: ChatAgent,
        conversation_id: int,
        max_tokens_before_summary: int | None = None,
        percent_to_keep: int = 50,
        min_messages_to_keep: int = 2,
        token_counter: TokenCounter = count_tokens_approximately,
    ) -> None:
        """Initialize the summarization middleware.

        Args:
            agent: BaseAgent for generating conversation summaries.
            max_tokens_before_summary: Token threshold to trigger summarization.
                If `None`, summarization is disabled.
            percent_to_keep: Percentage of recent messages to preserve after summarization.
            token_counter: Function to count tokens in messages.
        """
        super().__init__()

        self.agent = agent
        self.max_tokens_before_summary = max_tokens_before_summary or 100000
        self.percent_to_keep = percent_to_keep
        self.token_counter = token_counter
        self.messages_to_keep = min_messages_to_keep
        self.conversation_id = conversation_id
        self.logger = llmmllogger.bind(component="SummarizationMiddleware")

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:  # noqa: ARG002
        """Process messages before model invocation, potentially triggering summarization."""
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
            f" Keeping last {self.percent_to_keep}% of messages."
            f" Total messages: {len(messages)}"
        )

        cutoff_index = self._find_safe_cutoff(messages)

        self.logger.debug(f"Determined safe cutoff index at: {cutoff_index}")

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(
            messages, cutoff_index
        )

        summary = await self._create_summary(messages_to_summarize)

        return {
            "messages": [
                summary,
                *preserved_messages,
            ]
        }

    def _ensure_message_ids(self, messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs for the add_messages reducer."""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

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
        """Find safe cutoff point that preserves AI/Tool message pairs using token-based binary search.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Uses binary search to find the largest number
        of messages (from beginning) that amount to less than percent_to_keep of max_tokens.
        """
        if not self.max_tokens_before_summary:
            return 0

        if len(messages) <= self.messages_to_keep:  # Keep at least some messages
            return 0

        # Calculate target tokens to preserve (from the end)
        target_tokens_to_keep = int(
            (self.percent_to_keep / 100) * self.max_tokens_before_summary
        )

        self.logger.debug(
            f"Target tokens to keep: {target_tokens_to_keep} "
            f"({self.percent_to_keep}% of {self.max_tokens_before_summary})"
        )

        # Use binary search to find the cutoff point
        cutoff_index = self._binary_search_token_cutoff(messages, target_tokens_to_keep)

        # Find a safe cutoff point that doesn't separate AI/Tool pairs
        for i in range(cutoff_index, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _binary_search_token_cutoff(
        self, messages: list[AnyMessage], target_tokens_to_keep: int
    ) -> int:
        """Use binary search to find the largest cutoff index where remaining messages
        have tokens <= target_tokens_to_keep.

        Args:
            messages: List of messages to search
            target_tokens_to_keep: Target number of tokens to preserve from the end

        Returns:
            Index where to cut messages (messages[cutoff_index:] will be preserved)
        """
        if not messages:
            return 0

        # Binary search for the optimal cutoff
        left, right = 0, len(messages)
        best_cutoff = len(messages)  # Default to keeping all messages

        while left <= right:
            mid = (left + right) // 2

            # Calculate tokens in messages that would be preserved (from mid to end)
            preserved_messages = messages[mid:]
            preserved_tokens = self.token_counter(preserved_messages)

            self.logger.debug(
                f"Binary search: mid={mid}, preserved_tokens={preserved_tokens}, "
                f"target={target_tokens_to_keep}"
            )

            if preserved_tokens <= target_tokens_to_keep:
                # We can afford to preserve more messages, try cutting earlier
                best_cutoff = mid
                right = mid - 1
            else:
                # Too many tokens, need to cut later (preserve fewer messages)
                left = mid + 1

        self.logger.debug(f"Binary search result: cutoff_index={best_cutoff}")
        return best_cutoff

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
        """Generate summary using PrimarySummaryAgent and store in database."""
        if not messages_to_summarize:
            # Return empty summary object instead of string
            return "No previous conversation history."

        try:
            # Use PrimarySummaryAgent's summarize_conversation method
            summary = await self.agent.summarize_conversation(
                messages=lc_messages_to_messages(
                    messages_to_summarize, self.conversation_id
                ),
                level=1,
            )

            self.logger.debug(
                f"Created summary with {len(messages_to_summarize)} messages"
            )
            return summary.content

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error generating summary: {e}", exc_info=True)
            return f"Error generating summary: {e!s}"
