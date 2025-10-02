"""
Creative workflow implementation for composer.
Implements creative content generation with planning and refinement.
"""

from typing import List, Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import AvailableTool, ModelProfileType

from composer.graph.state import WorkflowState
from composer.nodes.standard import PipelineNode, ToolExecutorNode
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.engineering_agent import EngineeringAgentNode
from composer.monitoring.logging import composer_logger


async def build_creative_workflow(
    user_id: str, 
    tools: List[AvailableTool],
    pipeline_factory: Any = None
) -> CompiledStateGraph:
    """
    Build creative content generation workflow with planning and refinement.
    
    Workflow includes:
    1. Intent classification for creative task analysis
    2. Creative planning and ideation
    3. Content generation
    4. Refinement and enhancement
    5. Output formatting
    
    Args:
        user_id: User identifier for configuration retrieval
        tools: Available tools for this workflow
        pipeline_factory: Factory for creating pipeline instances
        
    Returns:
        Compiled LangGraph workflow
    """
    composer_logger.logger.info(
        "Building creative workflow",
        extra={"user_id": user_id, "tool_count": len(tools)}
    )

    # Create workflow graph
    workflow = StateGraph(WorkflowState)

    # Add workflow nodes
    workflow.add_node("intent_classifier", IntentClassifierNode())
    workflow.add_node("engineering_agent", EngineeringAgentNode(pipeline_factory))
    
    # Creative planning phase
    workflow.add_node("creative_planning", PipelineNode(
        pipeline_factory,
        ModelProfileType.Analysis,
        stream=False
    ))
    
    # Content generation with creativity-focused model
    workflow.add_node("content_generation", PipelineNode(
        pipeline_factory,
        ModelProfileType.Primary,
        stream=True
    ))
    
    # Refinement and enhancement
    workflow.add_node("refinement", PipelineNode(
        pipeline_factory,
        ModelProfileType.Secondary,
        stream=False
    ))
    
    # Output formatting and finalization
    workflow.add_node("output_formatting", PipelineNode(
        pipeline_factory,
        ModelProfileType.Primary,
        stream=True
    ))
    
    # Tool execution node (if tools available)
    if tools:
        workflow.add_node("tool_executor", ToolExecutorNode([]))

    # Set workflow entry point
    workflow.set_entry_point("intent_classifier")
    
    # Linear creative workflow progression
    workflow.add_edge("intent_classifier", "engineering_agent")
    workflow.add_edge("engineering_agent", "creative_planning")
    workflow.add_edge("creative_planning", "content_generation")
    
    # Conditional routing after content generation
    def route_after_generation(state: WorkflowState) -> str:
        """Route to refinement or directly to output based on quality assessment."""
        # Simple routing logic - could be enhanced with quality metrics
        content_quality = getattr(state, 'content_quality', 'good')
        
        if content_quality in ['poor', 'needs_improvement']:
            return "refinement"
        else:
            return "output_formatting"

    workflow.add_conditional_edges(
        "content_generation", 
        route_after_generation,
        {
            "refinement": "refinement",
            "output_formatting": "output_formatting"
        }
    )
    
    # Refinement flows to output formatting
    workflow.add_edge("refinement", "output_formatting")

    # Conditional routing after output formatting
    if tools:
        def route_after_output(state: WorkflowState) -> str:
            """Route to tools if tool calls present, otherwise end."""
            if (state.messages and 
                hasattr(state.messages[-1], 'tool_calls') and 
                state.messages[-1].tool_calls):
                return "tool_executor"
            return END

        workflow.add_conditional_edges("output_formatting", route_after_output)
        workflow.add_edge("tool_executor", "output_formatting")  # Tools loop back
    else:
        workflow.add_edge("output_formatting", END)

    # Compile and return workflow
    compiled_workflow = workflow.compile()
    
    composer_logger.logger.info(
        "Creative workflow built successfully",
        extra={"user_id": user_id, "node_count": len(workflow.nodes)}
    )
    
    return compiled_workflow


def get_creative_workflow_config(user_id: str) -> Dict[str, Any]:
    """
    Get configuration specific to creative workflows.
    
    Args:
        user_id: User identifier
        
    Returns:
        Creative workflow configuration dictionary
    """
    return {
        "workflow_type": "creative",
        "creative_planning": True,
        "content_generation": True,
        "refinement_enabled": True,
        "output_formatting": True,
        "streaming_generation": True,
        "quality_assessment": True,
        "user_id": user_id
    }