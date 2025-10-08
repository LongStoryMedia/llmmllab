"""
Multi-agent workflow implementation for composer.
Implements multi-agent orchestration with specialist coordination.
"""

from typing import Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import ModelProfileType
from runner import PipelineFactory

from composer.graph.state import WorkflowState
from composer.nodes import PipelineNode, ToolExecutorNode
from composer.nodes.routing import IntentClassifierNode
from composer.nodes.agents import EngineeringAgentNode
from composer.monitoring.logging import composer_logger
from models.required_capability import RequiredCapability


async def build_multi_agent_workflow(
    user_id: str, pipeline_factory: PipelineFactory
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
        extra={"user_id": user_id},
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

    # workflow.add_node(
    #     "image_generation_agent",
    #     PipelineNode(pipeline_factory, ModelProfileType.ImageGeneration, stream=False),
    # )

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

    # Tool execution node - tools populated at runtime
    workflow.add_node(
        "tool_executor", ToolExecutorNode([])
    )  # Tools populated at runtime

    # Set workflow entry point
    workflow.set_entry_point("intent_classifier")

    # Linear flow for multi-agent coordination
    workflow.add_edge("intent_classifier", "engineering_agent")
    workflow.add_edge("engineering_agent", "agent_router")

    # Conditional routing to specialists based on intent analysis
    def route_to_specialists(state: WorkflowState) -> str:
        """Route to appropriate specialist agents based on intent analysis."""

        # Use actual intent classification from state
        if not state.intent_classification:
            # Default to content generation if no intent analysis available
            return "content_generation"

        intent = state.intent_classification
        # Route based on primary intent patterns
        analysis_intents = {
            "research",
            "analyze",
            "investigate",
            "summarize",
            "compare",
            "evaluate",
            "calculate",
            "process",
        }

        # Check primary intent (handle list case)
        primary_intent_text = ""
        if isinstance(intent, list):
            if intent and hasattr(intent[0], "primary_intent"):
                primary_intent_text = intent[0].primary_intent.lower()
        elif hasattr(intent, "primary_intent"):
            primary_intent_text = intent.primary_intent.lower()

        if any(keyword in primary_intent_text for keyword in analysis_intents):
            return "analysis"

        # Default to content generation for creative, general, and conversational tasks
        return "content_generation"

    workflow.add_conditional_edges(
        "agent_router",
        route_to_specialists,
        {
            "analysis": "analysis_agent",
            "content_generation": "content_generation_agent",
        },
    )

    # Both specialist paths flow to coordination
    workflow.add_edge("analysis_agent", "coordination")
    workflow.add_edge("content_generation_agent", "coordination")

    # Coordination flows to final response
    workflow.add_edge("coordination", "final_response")

    # Conditional routing after final response
    def route_after_response(state: WorkflowState) -> str:
        """Route to tools if tool calls present, otherwise end."""
        if (
            state.messages
            and hasattr(state.messages[-1], "tool_calls")
            and state.messages[-1].tool_calls
            and state.available_tools
        ):
            return "tool_executor"
        return END

    workflow.add_conditional_edges("final_response", route_after_response)
    workflow.add_edge("tool_executor", "final_response")  # Tools loop back

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
