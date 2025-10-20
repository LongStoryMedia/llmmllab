"""
State building utilities for LangGraph and Workflow state objects.

This module provides functions to construct properly typed state objects
for use in LangGraph workflows and the composer system, as well as utilities
for assembling context messages from WorkflowState following the context
extension architecture patterns.
"""

from typing import Any, Dict, Iterable, List, Optional


from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    LangGraphState,
)
from composer.graph.state import WorkflowState
from .conversion import (
    convert_langchain_messages_to_messages,
)
from .langchain_compat import _coerce_to_langchain_message_dict


def build_langgraph_state(
    messages: Iterable[Any],
    user_input: str,
    *,
    error_count: int = 0,
    max_iterations: int = 10,
    current_iteration: int = 0,
    tools_used: Iterable[str] | None = None,
    intermediate_results: Dict[str, Any] | None = None,
) -> LangGraphState:
    """Construct a LangGraphState from heterogeneous message inputs safely."""
    msg_list = list(messages) if messages is not None else []
    coerced = [_coerce_to_langchain_message_dict(m) for m in msg_list]

    return LangGraphState(
        messages=coerced,  # type: ignore[arg-type]
        user_input=user_input or "",
        error_count=error_count,
        max_iterations=max_iterations,
        current_iteration=current_iteration,
        tools_used=list(tools_used or []),
        intermediate_results=dict(intermediate_results or {}),
    )


# =============================================================================
# CONTEXT ASSEMBLY UTILITIES
# =============================================================================


def _create_text_message_content(text: str) -> MessageContent:
    """Create a MessageContent object with text content."""
    return MessageContent(type=MessageContentType.TEXT, text=text, url=None)


def _text_to_message_content_list(text: str) -> List[MessageContent]:
    """Convert a text string to a list containing a single MessageContent object."""
    return [_create_text_message_content(text)]


def _memory_to_messages(memory, conversation_id: Optional[int] = None) -> List[Message]:
    """
    Convert a Memory object to a list of Message objects.

    Follows the context pairing logic from context_extension.md:
    - User messages are paired with assistant responses
    - Assistant messages are paired with user queries
    - Summaries are used directly

    Args:
        memory: Memory object from WorkflowState.retrieved_memories
        conversation_id: Optional conversation ID for the messages

    Returns:
        List of Message objects constructed from memory fragments
    """
    messages = []

    if not hasattr(memory, "fragments") or not memory.fragments:
        return messages

    for fragment in memory.fragments:
        if not hasattr(fragment, "content") or not hasattr(fragment, "role"):
            continue

        # Determine the role from the fragment
        role = MessageRole.USER  # Default
        if hasattr(fragment, "role") and fragment.role:
            role_str = str(fragment.role).lower()
            if role_str in ("assistant", "ai"):
                role = MessageRole.ASSISTANT
            elif role_str == "system":
                role = MessageRole.SYSTEM
            elif role_str in ("user", "human"):
                role = MessageRole.USER

        # Create message from fragment
        message = Message(
            content=_text_to_message_content_list(str(fragment.content)),
            role=role,
            conversation_id=conversation_id,
            created_at=getattr(memory, "created_at", None),
        )
        messages.append(message)

    return messages


def _summary_to_message(summary, conversation_id: Optional[int] = None) -> Message:
    """
    Convert a Summary object to a Message with SYSTEM role.

    Following context_extension.md guidance, summaries are integrated as system messages
    to provide hierarchical context without disrupting conversation flow.

    Args:
        summary: Summary object from WorkflowState.summaries
        conversation_id: Optional conversation ID for the message

    Returns:
        Message object with SYSTEM role containing summary content
    """
    content_text = f"[Summary Level {summary.level}]: {summary.content}"

    return Message(
        content=_text_to_message_content_list(content_text),
        role=MessageRole.SYSTEM,
        conversation_id=conversation_id,
        created_at=getattr(summary, "created_at", None),
    )


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    # Simple approximation: ~4 characters per token for most languages
    return max(1, len(text) // 4)


def _count_message_tokens(messages: List[Message]) -> int:
    """Count approximate tokens in messages."""
    total_tokens = 0
    for message in messages:
        if message.content:
            for content in message.content:
                if content.text:
                    total_tokens += _estimate_tokens(content.text)
    return total_tokens


def _trim_messages_to_context_window(
    messages: List[Message], max_tokens: int, reserve_tokens: int = 4096
) -> List[Message]:
    """
    Trim messages to fit within context window.

    Args:
        messages: List of messages to trim
        max_tokens: Maximum token count for context window
        reserve_tokens: Tokens to reserve for response generation

    Returns:
        Trimmed list of messages that fit within context window
    """
    available_tokens = max_tokens - reserve_tokens
    if available_tokens <= 0:
        return []

    # Always keep system messages
    system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
    other_messages = [msg for msg in messages if msg.role != MessageRole.SYSTEM]

    # Count system message tokens
    system_tokens = _count_message_tokens(system_messages)
    remaining_tokens = available_tokens - system_tokens

    if remaining_tokens <= 0:
        return system_messages

    # Add other messages from most recent, checking token limits
    trimmed_other = []
    current_tokens = 0

    for message in reversed(other_messages):
        message_tokens = _count_message_tokens([message])
        if current_tokens + message_tokens <= remaining_tokens:
            trimmed_other.insert(0, message)  # Insert at beginning to maintain order
            current_tokens += message_tokens
        else:
            break

    return system_messages + trimmed_other


def assemble_context_messages(
    state: WorkflowState, max_tokens: Optional[int] = None
) -> List[Message]:
    """
    Assemble a comprehensive list of Message objects from WorkflowState.

    Implements the context extension architecture from context_extension.md:
    1. Core conversation messages (highest priority)
    2. Retrieved memories (semantic relevance)
    3. Hierarchical summaries (context continuity)

    This function should be used every time messages are being sent to a pipeline
    to ensure consistent context assembly following the three-pronged approach.

    Args:
        state: WorkflowState containing messages, memories, and summaries
        max_tokens: Optional maximum token count for context window management

    Returns:
        List of Message objects assembled in context extension priority order,
        trimmed to fit within context window if max_tokens is provided
    """
    assembled_messages: List[Message] = []
    assert state.messages
    assert state.conversation_id

    # 1. CORE CONVERSATION MESSAGES (Highest Priority)
    # Convert LangChainMessage objects from state.messages to Message objects
    assembled_messages.extend(
        convert_langchain_messages_to_messages(state.messages, state.conversation_id)
    )

    # 2. RETRIEVED MEMORIES (Semantic Relevance Priority)
    # Following context_extension.md: "Memory search results ordered by similarity"
    if state.retrieved_memories:
        for memory in state.retrieved_memories:
            assembled_messages.extend(
                _memory_to_messages(memory, state.conversation_id)
            )

    # 3. HIERARCHICAL SUMMARIES (Context Continuity)
    # Following context_extension.md: "Hierarchical compression maintaining context"
    if state.summaries:
        assembled_messages.extend(
            [
                _summary_to_message(summary, state.conversation_id)
                for summary in state.summaries
            ]
        )

    final_messages = list(reversed(assembled_messages))

    # Apply context window trimming if max_tokens is provided
    if max_tokens:
        final_messages = _trim_messages_to_context_window(final_messages, max_tokens)

    return final_messages
