"""
GraphState Pydantic models with LangGraph reducers.
This is the centralized state schema that acts as the common interface
"""

import operator
from enum import Enum

from re import L
from typing import List, Dict, Any, Optional, Annotated, Union
from pydantic import BaseModel, Field, field_validator

from models import (
    LangChainMessage,
    Memory,
    IntentAnalysis,
    Tool,
    WorkflowType,
    UserConfig,
    Summary,
    SearchTopicSynthesis,
    ResponseFormat,
    TechnicalDomain,
    SearchResult,
    Message,
)


class ExecutionMetadata(BaseModel):
    """
    Strongly typed execution metadata for workflow state.
    Provides type safety and validation for runtime metadata.
    """

    model_config = {
        "extra": "forbid",  # Prevent additional fields for type safety
        "validate_assignment": True,
        "use_enum_values": True,
    }

    # Core execution information
    created_at: Optional[float] = Field(
        default=None, description="Workflow creation timestamp"
    )
    composer_version: Optional[str] = Field(
        default=None, description="Composer service version"
    )
    streaming_enabled: Optional[bool] = Field(
        default=None, description="Whether streaming is enabled for this workflow"
    )

    # Tool orchestration metadata
    tool_orchestration: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool orchestration metadata"
    )
    tool_generation_errors: Optional[List[str]] = Field(
        default=None, description="Errors encountered during tool generation"
    )
    orchestration_success: Optional[bool] = Field(
        default=None, description="Whether tool orchestration was successful"
    )
    dynamic_tools_generated: Optional[int] = Field(
        default=None, description="Number of dynamic tools generated"
    )
    static_tools_collected: Optional[int] = Field(
        default=None, description="Number of static tools collected"
    )
    tool_orchestration_error: Optional[str] = Field(
        default=None, description="Tool orchestration error message"
    )

    # Search and research metadata (execution results, not configuration)
    web_search_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Web search results data"
    )
    search_completed: Optional[bool] = Field(
        default=None, description="Whether search operation completed"
    )
    web_search_error: Optional[str] = Field(
        default=None, description="Web search error message"
    )

    # Summary operation completion flags
    conversation_summary_completed: Optional[bool] = Field(
        default=None, description="Whether conversation summarization completed"
    )
    search_summary_completed: Optional[bool] = Field(
        default=None, description="Whether search summarization completed"
    )
    consolidation_completed: Optional[bool] = Field(
        default=None, description="Whether summary consolidation completed"
    )

    # Legacy fields for backward compatibility (will be deprecated)
    conversation_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Legacy conversation summary - use level_1_summaries"
    )
    search_synthesis: Optional[Dict[str, Any]] = Field(
        default=None, description="Legacy search synthesis - use level_1_summaries"
    )

    text_summary_completed: Optional[bool] = Field(
        default=None, description="Whether text summarization completed"
    )

    # Node-specific failure tracking
    conversationsummarynode_failed: Optional[bool] = Field(
        default=None, description="ConversationSummaryNode execution failed"
    )
    searchsummarynode_failed: Optional[bool] = Field(
        default=None, description="SearchSummaryNode execution failed"
    )
    consolidationnode_failed: Optional[bool] = Field(
        default=None, description="ConsolidationNode execution failed"
    )
    textsummarynode_failed: Optional[bool] = Field(
        default=None, description="TextSummaryNode execution failed"
    )
    has_memory_context: Optional[bool] = Field(
        default=None, description="Whether memory context is available"
    )
    memories_stored: Optional[bool] = Field(
        default=None, description="Whether memories were successfully stored"
    )
    ranked_similarities: Optional[List[tuple]] = Field(
        default=None, description="Ranked similarity results"
    )

    # Error tracking
    errors: Optional[List[str]] = Field(
        default_factory=list, description="General error messages"
    )

    def update_tool_orchestration(
        self,
        tool_metadata: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        dynamic_tools_count: int = 0,
        static_tools_count: int = 0,
    ) -> None:
        """Update tool orchestration metadata."""
        self.tool_orchestration = tool_metadata or {}
        self.tool_generation_errors = errors or []
        self.orchestration_success = not bool(errors)
        self.dynamic_tools_generated = dynamic_tools_count
        self.static_tools_collected = static_tools_count

    def update_search_metadata(
        self,
        results: Optional[Dict[str, Any]] = None,
        completed: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update search execution metadata."""
        if results is not None:
            self.web_search_results = results
        if completed is not None:
            self.search_completed = completed
        if error is not None:
            self.web_search_error = error

    def add_error(self, error: str) -> None:
        """Add an error to the error list."""
        if self.errors is None:
            self.errors = []
        self.errors.append(error)

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return bool(self.errors and len(self.errors) > 0)

    # Note: Summary management methods are in WorkflowState class
    # ExecutionMetadata tracks completion flags, WorkflowState manages active summaries


class ExecutionStrategy(str, Enum):
    """Workflow execution strategies for multi-graph coordination."""

    SINGLE = "single"
    PARALLEL = "parallel"
    SERIES = "series"
    HYBRID = "hybrid"


class RoutingDecision(str, Enum):
    """
    Valid routing decisions for subgraph selection.

    Note: Values are dynamically generated from available workflows.
    See composer.workflows.registry.WorkflowRegistry for source of truth.
    """

    # Core workflows (always available)
    CHAT = "chat"
    RESEARCH = "research"
    CREATIVE = "creative"
    MULTI_AGENT = "multi_agent"

    # Extended workflows (may not always be available)
    ENGINEERING = "engineering"
    MEMORY = "memory"
    EMBEDDING_ONLY = "embedding_only"

    # Special routing target
    COORDINATOR = "coordinator"


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
        Optional[LangChainMessage], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Most recent user message in the conversation")

    things_to_remember: Annotated[
        List[Union[Message, LangChainMessage, Summary, SearchTopicSynthesis]],
        operator.add,
    ] = Field(
        default_factory=list, description="Key messages or information to remember"
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

    # Routing and execution control fields (referenced by builder.py)
    next_node: Annotated[Optional[str], lambda x, y: y if y is not None else x] = Field(
        default=None,
        description="Next node name for Command-based deterministic routing",
    )

    routing_decision: Annotated[
        Optional[RoutingDecision], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Explicit routing decision from router node")

    execution_strategy: Annotated[
        ExecutionStrategy, lambda x, y: y if y is not None else x
    ] = Field(
        default=ExecutionStrategy.SINGLE,
        description="Strategy for executing multiple subgraphs",
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

    selected_workflows: Annotated[List[str], lambda x, y: y if y is not None else x] = (
        Field(
            default_factory=list, description="List of workflows selected for execution"
        )
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

    execution_metadata: Annotated[
        ExecutionMetadata, lambda x, y: y if y is not None else x
    ] = Field(
        default_factory=ExecutionMetadata,
        description="Runtime metadata and debugging information",
    )

    # Circuit breaker and error tracking
    error_details: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Error details for circuit breaker and recovery",
    )

    @field_validator("workflow_type", mode="before")
    @classmethod
    def validate_workflow_type(cls, v):
        """Ensure workflow_type is properly typed."""
        if isinstance(v, str):
            try:
                return WorkflowType(v.upper())
            except ValueError:
                return WorkflowType.CHAT  # Default fallback
        return v


class ChatWorkflowState(WorkflowState):
    """Specialized state for chat workflows."""

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    title_generated: bool = Field(
        default=False, description="Whether conversation title has been generated"
    )

    tool_calls_pending: bool = Field(
        default=False, description="Whether there are pending tool calls to execute"
    )


class ResearchWorkflowState(WorkflowState):
    """Specialized state for research workflows."""

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    query_expanded: bool = Field(
        default=False, description="Whether initial query has been expanded"
    )

    sources_gathered: List[str] = Field(
        default_factory=list, description="List of information sources discovered"
    )

    synthesis_complete: bool = Field(
        default=False, description="Whether research synthesis is complete"
    )


class MultiAgentWorkflowState(WorkflowState):
    """Specialized state for multi-agent orchestration."""

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    active_agent: Annotated[Optional[str], lambda x, y: y if y is not None else x] = (
        Field(default=None, description="Currently active agent identifier")
    )

    agent_handoff_reason: Annotated[
        Optional[str], lambda x, y: y if y is not None else x
    ] = Field(default=None, description="Reason for agent handoff")

    collaborative_context: Dict[str, Any] = Field(
        default_factory=dict, description="Shared context between agents"
    )
