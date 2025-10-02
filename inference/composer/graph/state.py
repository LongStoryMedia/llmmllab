"""
GraphState Pydantic models with LangGraph reducers.
This is the centralized state schema that acts as the common interface     @validator('workflow_type', pre=True)
    @classmethod
    def validate_workflow_type(cls, v):
        \"\"\"Ensure workflow_type is properly typed.\"\"\"
        if isinstance(v, str):
            try:
                return WorkflowType(v.upper())
            except ValueError:
                return WorkflowType.CHAT  # Default fallback
        return vodes.
"""

import operator
from datetime import datetime
from enum import Enum

from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field, validator

from langgraph.graph import add_messages

from models.lang_chain_message import LangChainMessage
from models.intent_analysis import IntentAnalysis
from models.available_tool import AvailableTool
from models.workflow_type import WorkflowType


class ExecutionStrategy(str, Enum):
    """Workflow execution strategies for multi-graph coordination."""
    SINGLE = "single"
    PARALLEL = "parallel" 
    SERIES = "series"
    HYBRID = "hybrid"


class RoutingDecision(str, Enum):
    """Valid routing decisions for subgraph selection."""
    CHAT = "chat"
    RESEARCH = "research"
    CREATIVE = "creative"
    MULTI_AGENT = "multi_agent"
    COORDINATOR = "coordinator"


class WorkflowState(BaseModel):
    """
    Unified LangGraph state schema with reducer functions.
    This state is shared across all nodes in composer workflows.
    """

    # Conversation history and final outputs - essential for context and token streaming
    messages: Annotated[List[LangChainMessage], add_messages] = Field(
        default_factory=list, description="Conversation history and LLM outputs"
    )

    # Structured output from Intent Agent - directs subsequent RAG and tool decisions
    intent_classification: Annotated[Optional[IntentAnalysis], operator.add] = Field(
        default=None, description="Intent analysis results for routing decisions"
    )

    # Curated list of tools collected for current execution phase
    required_tools: Annotated[List[AvailableTool], operator.add] = Field(
        default_factory=list,
        description="Dynamic and static tools selected for this workflow",
    )

    # Consolidated, synthesized output from RAG execution (shallow or deep)
    search_results: Annotated[Optional[str], operator.add] = Field(
        default=None, description="Results from RAG operations"
    )

    # Stores RAG depth decision - drives conditional edge routing
    rag_depth_config: Annotated[Optional[str], operator.add] = Field(
        default=None, description="RAG complexity level: 'SHALLOW' or 'DEEP'"
    )

    # User-defined signals for granular progress tracking
    progress_updates: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Progress signals during tool or crawl execution",
    )

    # Routing and execution control fields (referenced by builder.py)
    next_node: Optional[str] = Field(
        default=None, 
        description="Next node name for Command-based deterministic routing"
    )
    
    routing_decision: Optional[RoutingDecision] = Field(
        default=None,
        description="Explicit routing decision from router node"
    )
    
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.SINGLE,
        description="Strategy for executing multiple subgraphs"
    )
    
    selected_workflows: List[str] = Field(
        default_factory=list,
        description="List of workflows selected for execution"
    )

    # Additional context fields
    user_id: Optional[str] = Field(
        default=None, description="User identifier for personalization"
    )

    workflow_type: Optional[WorkflowType] = Field(
        default=None,
        description="Type of workflow: CHAT, RESEARCH, MULTI_AGENT, CREATIVE",
    )

    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Runtime metadata and debugging information"
    )
    
    # Circuit breaker and error tracking
    error_details: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Error details for circuit breaker and recovery"
    )
    
    # Timing and performance metadata
    execution_start_time: Optional[datetime] = Field(
        default=None,
        description="Workflow execution start timestamp"
    )
    
    node_execution_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution time tracking per node"
    )

    @validator('workflow_type', pre=True)
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

    title_generated: bool = Field(
        default=False, description="Whether conversation title has been generated"
    )

    tool_calls_pending: bool = Field(
        default=False, description="Whether there are pending tool calls to execute"
    )


class ResearchWorkflowState(WorkflowState):
    """Specialized state for research workflows."""

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

    active_agent: Optional[str] = Field(
        default=None, description="Currently active agent identifier"
    )

    agent_handoff_reason: Optional[str] = Field(
        default=None, description="Reason for agent handoff"
    )

    collaborative_context: Dict[str, Any] = Field(
        default_factory=dict, description="Shared context between agents"
    )
