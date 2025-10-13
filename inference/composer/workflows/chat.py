"""
Chat workflow implementation for composer.
Implements the standard chat workflow with adaptive search and tool orchestration.
"""

from typing import Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import ModelProfileType

from composer.graph.state import WorkflowState
from composer.nodes import PipelineNode, ToolExecutorNode
from composer.nodes.routing import IntentClassifierNode
from composer.nodes.agents import EngineeringAgentNode
from composer.nodes.research import (
    ResearchRouter,
    QuickResearchExecutor,
    ComprehensiveResearchExecutor,
)
from utils.logging import llmmllogger


async def build_chat_workflow(
    user_id: str, pipeline_factory: Any = None
) -> CompiledStateGraph:
    """
    Build standard chat workflow with adaptive search routing.

    Workflow includes:
    1. Intent classification to determine search depth
    2. Engineering agent for dynamic tool orchestration
    3. Conditional search routing (shallow vs deep)
    4. Primary chat agent with streaming support
    5. Tool execution with conditional routing

    Args:
        user_id: User identifier for configuration retrieval
        pipeline_factory: Factory for creating pipeline instances

    Returns:
        Compiled LangGraph workflow
    """
    llmmllogger.logger.info("Building chat workflow", extra={"user_id": user_id})

    # Create workflow graph
    workflow = StateGraph(WorkflowState)

    # Add workflow nodes
    workflow.add_node("classifier_agent", IntentClassifierNode())
    workflow.add_node("engineering_agent", EngineeringAgentNode(pipeline_factory))
    workflow.add_node("execute_quick_research", QuickResearchExecutor(user_id))
    workflow.add_node(
        "execute_comprehensive_research", ComprehensiveResearchExecutor(user_id)
    )

    # Primary chat agent with streaming enabled
    workflow.add_node(
        "chat_agent",
        PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=True),
    )

    # Tool execution node - tools populated at runtime by tool_collection_node
    workflow.add_node(
        "tool_executor", ToolExecutorNode([])
    )  # Tools populated at runtime

    # Set workflow entry point
    workflow.set_entry_point("classifier_agent")

    # Linear flow to engineering agent
    workflow.add_edge("classifier_agent", "engineering_agent")

    # Conditional search routing based on intent analysis
    def route_search_depth(state: WorkflowState) -> str:
        """Route to appropriate search implementation based on intent classification."""
        router = ResearchRouter()
        return router.route_research_depth(state)

    workflow.add_conditional_edges(
        "engineering_agent",
        route_search_depth,
        {
            "execute_quick_research": "execute_quick_research",
            "execute_comprehensive_research": "execute_comprehensive_research",
        },
    )

    # Both research paths flow to chat agent
    workflow.add_edge("execute_quick_research", "chat_agent")
    workflow.add_edge("execute_comprehensive_research", "chat_agent")

    # Conditional routing after chat agent
    def route_after_agent(state: WorkflowState) -> str:
        """Route to tools if tool calls present, otherwise end."""
        if (
            state.messages
            and hasattr(state.messages[-1], "tool_calls")
            and state.messages[-1].tool_calls
            and hasattr(state, "required_tools")
            and state.available_tools
        ):
            return "tool_executor"
        return END

    workflow.add_conditional_edges("chat_agent", route_after_agent)
    workflow.add_edge("tool_executor", "chat_agent")  # Tools loop back to agent

    # Compile and return workflow
    compiled_workflow = workflow.compile()

    llmmllogger.logger.info(
        "Chat workflow built successfully",
        extra={"user_id": user_id, "node_count": len(workflow.nodes)},
    )

    return compiled_workflow


def get_chat_workflow_config(user_id: str) -> Dict[str, Any]:
    """
    Get configuration specific to chat workflows.

    Args:
        user_id: User identifier

    Returns:
        Chat workflow configuration dictionary
    """
    return {
        "workflow_type": "chat",
        "streaming_enabled": True,
        "adaptive_search": True,
        "tool_orchestration": True,
        "intent_classification": True,
        "user_id": user_id,
    }
