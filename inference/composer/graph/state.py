"""
GraphState Pydantic models with LangGraph reducers.
This is the centralized state schema that acts as the common interface
"""

import operator
from typing import List, Dict, Any, Optional, Annotated, Set, Union
from pydantic import BaseModel, Field

from models import (
    LangChainMessage,
    Memory,
    IntentAnalysis,
    Tool,
    WorkflowType,
    UserConfig,
    Summary,
    SearchTopicSynthesis,
    SearchResult,
    Message,
    NodeMetadata,
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
        Optional[LangChainMessage], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Most recent user message in the conversation")

    things_to_remember: Annotated[
        List[Union[Message, LangChainMessage, Summary, SearchTopicSynthesis]],
        operator.add,
    ] = Field(
        default_factory=list, description="Key messages or information to remember"
    )

    title: Annotated[Optional[str], lambda x, y: y if y is not None else x] = Field(
        default=None, description="Title for the conversation or workflow"
    )

    # Conversation history and final outputs - essential for context and token streaming
    messages: Annotated[
        List[LangChainMessage], lambda x, y: y if y is not None else x
    ] = Field(default_factory=list, description="Conversation history and LLM outputs")

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
        Optional[List[Dict[str, Any]]],
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
