"""
Chat workflow implementation for composer.
Implements the standard chat workflow with adaptive search and tool orchestration.
"""

from typing import List, Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import AvailableTool, ModelProfileType

from composer.graph.state import WorkflowState
from composer.nodes.standard import PipelineNode, ToolExecutorNode
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.engineering_agent import EngineeringAgentNode
from composer.nodes.search.router import SearchDepthRouter, ShallowSearchExecutor, DeepSearchExecutor
from composer.monitoring.logging import composer_logger


async def build_chat_workflow(
    user_id: str, 
    tools: List[AvailableTool],
    pipeline_factory: Any = None
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
        tools: Available tools for this workflow
        pipeline_factory: Factory for creating pipeline instances
        
    Returns:
        Compiled LangGraph workflow
    """
    composer_logger.logger.info(
        "Building chat workflow",
        extra={"user_id": user_id, "tool_count": len(tools)}
    )

    # Create workflow graph
    workflow = StateGraph(WorkflowState)

    # Add workflow nodes
    workflow.add_node("intent_classifier", IntentClassifierNode())
    workflow.add_node("engineering_agent", EngineeringAgentNode(pipeline_factory))
    workflow.add_node("execute_shallow_search", ShallowSearchExecutor(user_id))
    workflow.add_node("execute_deep_crawl_and_synthesize", DeepSearchExecutor(user_id))
    
    # Primary chat agent with streaming enabled
    workflow.add_node("chat_agent", PipelineNode(
        pipeline_factory, 
        ModelProfileType.Primary, 
        stream=True
    ))
    
    # Tool execution node (if tools available)
    if tools:
        workflow.add_node("tool_executor", ToolExecutorNode([]))  # Tools populated at runtime

    # Set workflow entry point
    workflow.set_entry_point("intent_classifier")
    
    # Linear flow to engineering agent
    workflow.add_edge("intent_classifier", "engineering_agent")

    # Conditional search routing based on intent analysis
    def route_search_depth(state: WorkflowState) -> str:
        """Route to appropriate search implementation based on intent classification."""
        router = SearchDepthRouter()
        return router.route_search_depth(state)

    workflow.add_conditional_edges(
        "engineering_agent", 
        route_search_depth,
        {
            "execute_shallow_search": "execute_shallow_search",
            "execute_deep_crawl_and_synthesize": "execute_deep_crawl_and_synthesize"
        }
    )

    # Both search paths flow to chat agent
    workflow.add_edge("execute_shallow_search", "chat_agent")
    workflow.add_edge("execute_deep_crawl_and_synthesize", "chat_agent")

    # Conditional routing after chat agent
    if tools:
        def route_after_agent(state: WorkflowState) -> str:
            """Route to tools if tool calls present, otherwise end."""
            if (state.messages and 
                hasattr(state.messages[-1], 'tool_calls') and 
                state.messages[-1].tool_calls):
                return "tool_executor"
            return END

        workflow.add_conditional_edges("chat_agent", route_after_agent)
        workflow.add_edge("tool_executor", "chat_agent")  # Tools loop back to agent
    else:
        workflow.add_edge("chat_agent", END)

    # Compile and return workflow
    compiled_workflow = workflow.compile()
    
    composer_logger.logger.info(
        "Chat workflow built successfully",
        extra={"user_id": user_id, "node_count": len(workflow.nodes)}
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
        "user_id": user_id
    }