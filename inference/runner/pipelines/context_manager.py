"""
Context management utilities for LLM pipelines to handle context window limitations.
"""

import logging
from typing import List, Tuple
from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context window limitations by intelligently truncating message history."""

    def __init__(self, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation - approximately 4 characters per token."""
        return len(text) // 4

    def get_message_tokens(self, message: BaseMessage) -> int:
        """Estimate tokens in a message."""
        content = ""
        if hasattr(message, "content") and message.content:
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                content = " ".join(str(item) for item in message.content)
        return self.estimate_tokens(content)

    def truncate_messages(
        self, messages: List[BaseMessage], target_tokens: int | None = None
    ) -> Tuple[List[BaseMessage], int]:
        """
        Truncate message history to fit within context window.

        Always keeps the last user message and preserves system messages.
        Returns truncated messages and estimated token count.
        """
        if target_tokens is None:
            target_tokens = int(
                self.max_context_tokens * 0.7
            )  # Leave room for response

        if not messages:
            return messages, 0

        # Separate system messages and conversation messages
        system_messages: List[BaseMessage] = [
            msg for msg in messages if isinstance(msg, SystemMessage)
        ]
        other_messages: List[BaseMessage] = [
            msg for msg in messages if not isinstance(msg, SystemMessage)
        ]

        # Always keep the last message (usually the current user input)
        if other_messages:
            last_message = other_messages[-1]
            remaining_messages = other_messages[:-1]
        else:
            last_message = None
            remaining_messages = []

        # Calculate tokens for fixed parts
        system_tokens = sum(self.get_message_tokens(msg) for msg in system_messages)
        last_message_tokens = (
            self.get_message_tokens(last_message) if last_message else 0
        )

        available_tokens = target_tokens - system_tokens - last_message_tokens

        # Select messages from the end of remaining_messages that fit in available space
        selected_messages: List[BaseMessage] = []
        used_tokens = 0

        for message in reversed(remaining_messages):
            msg_tokens = self.get_message_tokens(message)
            if used_tokens + msg_tokens <= available_tokens:
                selected_messages.insert(0, message)
                used_tokens += msg_tokens
            else:
                break

        # Combine all parts
        final_messages: List[BaseMessage] = system_messages + selected_messages
        if last_message:
            final_messages.append(last_message)

        total_tokens = system_tokens + used_tokens + last_message_tokens

        if len(final_messages) < len(messages):
            logger.info(
                f"Truncated {len(messages) - len(final_messages)} messages "
                f"to fit {total_tokens} tokens in context window"
            )

        return final_messages, total_tokens

    def handle_context_overflow(
        self, messages: List[BaseMessage], error_message: str
    ) -> List[BaseMessage]:
        """
        Handle context overflow by aggressively truncating and adding explanation.
        """
        # Extract token count from error if possible
        if "tokens" in error_message:
            try:
                # Try to find actual token count in error message
                import re

                match = re.search(
                    r"(\d+)\s*tokens.*exceed.*context.*?(\d+)", error_message
                )
                if match:
                    context_limit = int(match.group(2))
                    # Be very conservative - use only 60% of context limit
                    target_tokens = int(context_limit * 0.6)
                else:
                    target_tokens = int(self.max_context_tokens * 0.5)
            except Exception:
                target_tokens = int(self.max_context_tokens * 0.5)
        else:
            target_tokens = int(self.max_context_tokens * 0.5)

        truncated_messages, actual_tokens = self.truncate_messages(
            messages, target_tokens
        )

        # Add a system message explaining the truncation if we truncated significantly
        if len(truncated_messages) < len(messages) - 1:
            explanation = SystemMessage(
                content=(
                    f"[Note: This conversation was truncated to fit the context window. "
                    f"Some earlier messages were removed. Current context: {actual_tokens} tokens.]"
                )
            )
            truncated_messages.insert(
                -1, explanation
            )  # Insert before last user message

        return truncated_messages
