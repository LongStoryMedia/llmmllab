"""
GraphState Pydantic models with LangGraph reducers.
This is the centralized state schema that acts as the common interface
"""

import operator
from typing import Any, Dict, List, Optional, Annotated, Sequence, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from models import (
    Memory,
    MessageContent,
    MessageContentType,
    MessageRole,
    UserConfig,
    Summary,
    SearchTopicSynthesis,
    SearchResult,
    Message,
    Document,
    ToolCall,
)


@dataclass
class ServerToolEvent:
    """Emitted by the executor when ServerToolNode executes a server-side tool.

    Carries the original tool call and the execution result so the router
    can emit the correct SSE content blocks (server_tool_use, result).
    """

    tool_call: ToolCall
    result_text: str
    canonical_name: str  # "web_search", "web_fetch", etc.


class WorkflowState(BaseModel):
    """
    Unified LangGraph state schema with reducer functions.
    This state is shared across all nodes in composer workflows.
    """

    model_config = {
        "arbitrary_types_allowed": True,  # Allow LangChain message types
        "validate_assignment": True,  # Validate on field assignment
        "use_enum_values": True,  # Use enum values in serialization
        "extra": "forbid",  # Prevent extra fields for type safety
    }

    current_user_message: Annotated[
        Optional[Message], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Most recent user message in the conversation")

    workflow_type: Annotated[Optional[str], lambda x, y: y if y is not None else x] = (
        Field(default=None, description="Type of workflow (ide, dialog, etc.)")
    )

    things_to_remember: Annotated[
        Sequence[Union[Message, Summary, SearchTopicSynthesis, Document]],
        operator.add,
    ] = Field(
        default_factory=list, description="Key messages or information to remember"
    )

    title: Annotated[Optional[str], lambda x, y: y if y is not None else x] = Field(
        default=None, description="Title for the conversation or workflow"
    )

    # Conversation history and final outputs - essential for context and token streaming
    messages: Annotated[List[Message], lambda x, y: y if y is not None else x] = Field(
        default_factory=list, description="Conversation history and LLM outputs"
    )

    summaries: Annotated[List[Summary], operator.add] = Field(
        default_factory=list,
        description="All summaries relevant to this workflow execution",
    )

    # Memory retrieval results
    retrieved_memories: Annotated[List[Memory], operator.add] = Field(
        default_factory=list,
        description="Retrieved memories from similarity search",
    )

    created_memories: Annotated[List[Memory], operator.add] = Field(
        default_factory=list,
        description="Memories created during this workflow execution",
    )

    # search results
    web_search_results: Annotated[List[SearchResult], operator.add] = Field(
        default_factory=list,
        description="Web search results from integrated search engines",
    )

    search_syntheses: Annotated[List[SearchTopicSynthesis], operator.add] = Field(
        default_factory=list, description="Syntheses of web search results"
    )
    # Additional context fields
    conversation_id: Annotated[int, lambda x, y: y if y is not None else x] = Field(
        ...,
        description="Conversation identifier for memory and context management",
    )

    user_id: Annotated[str, lambda x, y: y if y is not None else x] = Field(
        ..., description="User identifier for personalization"
    )

    # User configuration - centralized to eliminate database fetch duplication
    user_config: Annotated[UserConfig, lambda x, y: y if y is not None else x] = Field(
        ..., description="User configuration for this workflow execution"
    )

    # Server tool execution events — populated by ServerToolNode so the
    # executor can yield them to the router for SSE emission.
    server_tool_events: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Server-side tool call/result events for streaming",
    )

    # Tracks how many times the Agent→ServerToolNode→Agent loop has run
    # so the routing function can enforce a maximum iteration count.
    server_tool_iterations: Annotated[int, lambda x, y: (x or 0) + (y or 0)] = Field(
        default=0,
        description="Counter for server tool loop iterations",
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
    assembled_messages.extend(state.messages)

    # 2. RETRIEVED MEMORIES (Semantic Relevance Priority)
    # Following context_extension.md: "Memory search results ordered by similarity"
    if state.retrieved_memories:
        for memory in state.retrieved_memories:
            assembled_messages.append(_memory_to_message(memory, state.conversation_id))

    # 3. HIERARCHICAL SUMMARIES (Context Continuity)
    # Following context_extension.md: "Hierarchical compression maintaining context"
    if state.summaries:
        for summary in state.summaries:
            assembled_messages.append(
                _summary_to_message(summary, state.conversation_id)
            )

    final_messages = list(assembled_messages)

    # Apply context window trimming if max_tokens is provided
    # if max_tokens:
    #     final_messages = _trim_messages_to_context_window(final_messages, max_tokens)

    return final_messages


def _memory_to_message(
    memory: Memory, conversation_id: Optional[int] = None
) -> Message:
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
    message = Message(
        content=[],
        role=MessageRole.SYSTEM,
        conversation_id=conversation_id,
        created_at=getattr(memory, "created_at", None),
    )

    txt = (
        f"MEMORY FROM {memory.created_at}, conversation ID {memory.conversation_id}:\n"
    )

    for fragment in memory.fragments:
        txt += f"{fragment.role.value.upper()}: {fragment.content}\n"

    message.content.append(
        MessageContent(
            type=MessageContentType.TEXT,
            text=txt,
            url=None,
        )
    )
    return message


def _summary_to_message(
    summary: Summary, conversation_id: Optional[int] = None
) -> Message:
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
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text=content_text,
                url=None,
            )
        ],
        role=MessageRole.SYSTEM,
        conversation_id=conversation_id,
        created_at=getattr(summary, "created_at", None),
    )
