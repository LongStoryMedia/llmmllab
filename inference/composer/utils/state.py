"""
State building utilities for LangGraph and Workflow state objects.

This module provides functions to construct properly typed state objects
for use in LangGraph workflows and the composer system, as well as utilities
for assembling context messages from WorkflowState following the context
extension architecture patterns.
"""

import asyncio
from typing import Any, Dict, Iterable, List, Optional


from models import (
    Message,
    UserConfig,
    MessageRole,
    MessageContent,
    MessageContentType,
    LangGraphState,
)
from composer.graph.state import WorkflowState
from .conversion import (
    convert_messages_to_langchain,
    convert_langchain_messages_to_messages,
    message_to_langchain_message,
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


def assemble_context_messages(state: WorkflowState) -> List[Message]:
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

    Returns:
        List of Message objects assembled in context extension priority order
    """
    assembled_messages = []
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

    return reversed(assembled_messages)
