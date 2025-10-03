"""
Multi-agent workflow implementation for composer.
Implements multi-agent orchestration with specialist coordination.
"""

from typing import List, Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import AvailableTool, ModelProfileType

from composer.graph.state import WorkflowState
from composer.nodes.standard import PipelineNode, ToolExecutorNode
from composer.nodes.routing import IntentClassifierNode
from composer.nodes.agents import EngineeringAgentNode
from composer.monitoring.logging import composer_logger


async def build_multi_agent_workflow(
    user_id: str, tools: List[AvailableTool], pipeline_factory: Any = None
) -> CompiledStateGraph:
    """
    Build multi-agent orchestration workflow with specialist coordination.

    Workflow includes:
    1. Intent classification for task analysis
    2. Agent router for specialist selection
    3. Specialized agent execution:
       - Analysis Agent: Technical analysis, research, data processing, summarization
       - Content Generation Agent: Creative writing, content creation, general tasks
    4. Coordination and synthesis
    5. Final response generation

    Args:
        user_id: User identifier for configuration retrieval
        tools: Available tools for this workflow
        pipeline_factory: Factory for creating pipeline instances

    Returns:
        Compiled LangGraph workflow
    """
    composer_logger.logger.info(
        "Building multi-agent workflow",
        extra={"user_id": user_id, "tool_count": len(tools)},
    )

    # Create workflow graph
    workflow = StateGraph(WorkflowState)

    # Add workflow nodes
    workflow.add_node("intent_classifier", IntentClassifierNode())
    workflow.add_node("engineering_agent", EngineeringAgentNode(pipeline_factory))

    # Agent router for specialist selection
    workflow.add_node(
        "agent_router",
        PipelineNode(pipeline_factory, ModelProfileType.Analysis, stream=False),
    )

    # Specialized agents
    workflow.add_node(
        "analysis_agent",
        PipelineNode(pipeline_factory, ModelProfileType.Analysis, stream=False),
    )

    workflow.add_node(
        "content_generation_agent",
        PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=False),
    )

    # Coordination agent
    workflow.add_node(
        "coordination",
        PipelineNode(pipeline_factory, ModelProfileType.Analysis, stream=False),
    )

    # Final response with streaming
    workflow.add_node(
        "final_response",
        PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=True),
    )

    # Tool execution node (if tools available)
    if tools:
        workflow.add_node("tool_executor", ToolExecutorNode([]))

    # Set workflow entry point
    workflow.set_entry_point("intent_classifier")

    # Linear flow for multi-agent coordination
    workflow.add_edge("intent_classifier", "engineering_agent")
    workflow.add_edge("engineering_agent", "agent_router")

    # Conditional routing to specialists based on task type
    def route_to_specialists(state: WorkflowState) -> str:
        """Route to appropriate specialist agents based on task analysis."""
        # Enhanced routing logic based on intent analysis and task type
        task_type = getattr(state, "task_type", "general")

        if task_type in [
            "technical",
            "analysis",
            "research",
            "data_processing",
            "summarization",
        ]:
            return "analysis_agent"
        else:
            return "content_generation_agent"

    workflow.add_conditional_edges(
        "agent_router",
        route_to_specialists,
        {
            "analysis_agent": "analysis_agent",
            "content_generation_agent": "content_generation_agent",
        },
    )

    # Both specialist paths flow to coordination
    workflow.add_edge("analysis_agent", "coordination")
    workflow.add_edge("content_generation_agent", "coordination")

    # Coordination flows to final response
    workflow.add_edge("coordination", "final_response")

    # Conditional routing after final response
    if tools:

        def route_after_response(state: WorkflowState) -> str:
            """Route to tools if tool calls present, otherwise end."""
            if (
                state.messages
                and hasattr(state.messages[-1], "tool_calls")
                and state.messages[-1].tool_calls
            ):
                return "tool_executor"
            return END

        workflow.add_conditional_edges("final_response", route_after_response)
        workflow.add_edge("tool_executor", "final_response")  # Tools loop back
    else:
        workflow.add_edge("final_response", END)

    # Compile and return workflow
    compiled_workflow = workflow.compile()

    composer_logger.logger.info(
        "Multi-agent workflow built successfully",
        extra={"user_id": user_id, "node_count": len(workflow.nodes)},
    )

    return compiled_workflow


def get_multi_agent_workflow_config(user_id: str) -> Dict[str, Any]:
    """
    Get configuration specific to multi-agent workflows.

    Args:
        user_id: User identifier

    Returns:
        Multi-agent workflow configuration dictionary
    """
    return {
        "workflow_type": "multi_agent",
        "specialist_coordination": True,
        "agent_orchestration": True,
        "task_distribution": True,
        "synthesis_enabled": True,
        "streaming_response": True,
        "user_id": user_id,
    }
