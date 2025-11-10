"""
GraphState Pydantic models with LangGraph reducers.
This is the centralized state schema that acts as the common interface
"""

import operator
from typing import List, Dict, Any, Optional, Annotated, Set, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from models import (
    Memory,
    IntentAnalysis,
    MessageContent,
    MessageContentType,
    MessageRole,
    TodoItem,
    Tool,
    ToolConfig,
    WorkflowType,
    UserConfig,
    Summary,
    SearchTopicSynthesis,
    SearchResult,
    Message,
    NodeMetadata,
    ToolCall,
)


class ToolsState(BaseModel):
    """
    Minimal state for agent subgraph with chat_agent + tool_node workflow.

    Contains only essential data for the agent to operate efficiently while
    minimizing context window usage. The agent cycles between chat_agent and
    tool_node until completion, then returns results via Command.
    """

    # Message thread for agent conversation (using LangChain core messages for proper serialization)
    messages: Annotated[List[BaseMessage], add_messages]

    # Essential context for tool operations
    user_id: str
    conversation_id: int

    # User configuration (full object for tool access)
    user_config: UserConfig

    # Current operation tracking
    tool_call_count: int

    # Shared pipeline for tools to prevent duplicate server instances
    shared_pipeline: Optional[Any] = Field(
        default=None, description="Pipeline instance for tools to reuse"
    )


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

    current_date: Annotated[Optional[str], lambda x, y: y if y is not None else x] = (
        Field(
            default_factory=lambda: __import__("datetime").datetime.now().isoformat(),
            description="Current date in ISO format",
        )
    )

    current_user_message: Annotated[
        Optional[Message], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Most recent user message in the conversation")

    things_to_remember: Annotated[
        List[Union[Message, Summary, SearchTopicSynthesis]],
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

    # Structured output from Intent Agent - directs subsequent search and tool decisions
    intent_classification: Annotated[
        List[IntentAnalysis], lambda x, y: y if y is not None else x
    ] = Field(
        default_factory=list,
        description="Intent analysis results for routing decisions",
    )

    # Curated list of tools collected for current execution phase
    available_tools: Annotated[List[Tool], lambda x, y: y if y is not None else x] = (
        Field(
            default_factory=list,
            description="Dynamic and static tools selected for this workflow",
        )
    )

    dynamic_tools: Annotated[List[Tool], lambda x, y: y if y is not None else x] = (
        Field(
            default_factory=list,
            description="Dynamic tools created during this workflow",
        )
    )

    static_tools: Annotated[List[Tool], lambda x, y: y if y is not None else x] = Field(
        default_factory=list,
        description="Static tools available for this workflow",
    )

    # Stores search depth decision - drives conditional edge routing
    search_depth_config: Annotated[
        Optional[str], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Search complexity level: 'SHALLOW' or 'DEEP'")

    # User-defined signals for granular progress tracking
    progress_updates: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Progress signals during tool or crawl execution",
    )

    summaries: Annotated[List[Summary], operator.add] = Field(
        default_factory=list,
        description="All summaries relevant to this workflow execution",
    )

    # Ephemeral structured tool calls from the latest assistant message.
    # This is surfaced explicitly so streaming state events include a
    # 'tool_calls' key allowing external harnesses to detect tool usage
    # without parsing raw assistant content. Replaced (not concatenated)
    # each time a new assistant message is produced.
    tool_calls: Annotated[
        Optional[List[ToolCall]],
        lambda current, new: new if new is not None else current,
    ] = Field(
        default=None,
        description="Structured tool calls from the most recent assistant generation",
    )

    embedding: Annotated[
        Optional[List[float]],
        lambda current, new: new if new is not None else current,
    ] = Field(
        default=None,
        description="Embedding for comparison and retrieval tasks",
    )

    unranked_retrievals: Annotated[
        Optional[List[List[float]]],
        lambda current, new: new if new is not None else current,
    ] = Field(
        default=None,
        description="Unranked retrieval results from the most recent retrieval operation",
    )

    ranked_retrievals: Annotated[
        Optional[List[List[float]]],
        lambda current, new: new if new is not None else current,
    ] = Field(
        default=None,
        description="Ranked retrieval results from the most recent retrieval operation",
    )

    # Routing and execution control fields (referenced by builder.py)
    next_node: Annotated[Optional[str], lambda x, y: y if y is not None else x] = Field(
        default=None,
        description="Next node name for Command-based deterministic routing",
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

    search_query: Annotated[Optional[str], lambda x, y: y if y is not None else x] = (
        Field(default=None, description="Search query used for web search")
    )

    selected_workflows: Annotated[
        Set[WorkflowType], lambda x, y: y if y is not None else x
    ] = Field(
        default_factory=set, description="List of workflows selected for execution"
    )

    # Additional context fields
    user_id: Annotated[Optional[str], lambda x, y: y if y is not None else x] = Field(
        default=None, description="User identifier for personalization"
    )

    conversation_id: Annotated[
        Optional[int], lambda x, y: y if y is not None else x
    ] = Field(
        default=None,
        description="Conversation identifier for memory and context management",
    )

    # User configuration - centralized to eliminate database fetch duplication
    user_config: Annotated[
        Optional[UserConfig], lambda x, y: y if y is not None else x
    ] = Field(
        default=None, description="User configuration for this workflow execution"
    )

    workflow_type: Annotated[
        Optional[WorkflowType], lambda x, y: y if y is not None else x
    ] = Field(
        default=None,
        description="Type of workflow: CHAT, RESEARCH, MULTI_AGENT, CREATIVE",
    )

    # Circuit breaker and error tracking
    error_details: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Error details for circuit breaker and recovery",
    )

    # Node execution metadata tracking
    node_metadata: Annotated[
        Dict[str, NodeMetadata], lambda x, y: {**x, **y} if x and y else y or x or {}
    ] = Field(
        default_factory=dict,
        description="Strongly typed metadata from node executions keyed by node_id",
    )

    # Generated todos from planning middleware
    generated_todos: Annotated[
        List[TodoItem], lambda x, y: y if y is not None else x
    ] = Field(
        default_factory=list,
        description="Todos automatically generated by intent analysis planning middleware",
    )

    # Active todos from database (checkpointer integration)
    active_todos: Annotated[List[TodoItem], lambda x, y: y if y is not None else x] = (
        Field(
            default_factory=list,
            description="Active todos loaded from database for context continuity",
        )
    )

    # Planning context for multi-turn workflows
    planning_steps: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Sequence of planning steps taken in this workflow",
    )

    complexity_score: Annotated[
        Optional[int], lambda x, y: y if y is not None else x
    ] = Field(
        default=None,
        description="Complexity score from planning analysis (1-10 scale)",
    )

    # Checkpoint context for state persistence
    checkpoint_metadata: Annotated[
        Dict[str, Any], lambda x, y: {**x, **y} if x and y else y or x or {}
    ] = Field(
        default_factory=dict,
        description="Metadata for checkpoint persistence including turn tracking",
    )


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
