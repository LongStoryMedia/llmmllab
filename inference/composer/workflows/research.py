"""
Research workflow implementation for composer.
Implements research workflow with deep search and comprehensive synthesis.
"""

from typing import List, Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import AvailableTool, ModelProfileType

from composer.graph.state import ResearchWorkflowState
from composer.nodes.standard import PipelineNode
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.search.router import DeepSearchExecutor
from composer.monitoring.logging import composer_logger


async def build_research_workflow(
    user_id: str, tools: List[AvailableTool], pipeline_factory: Any = None
) -> CompiledStateGraph:
    """
    Build research workflow with deep RAG and synthesis capabilities.

    Workflow emphasizes:
    1. Comprehensive information gathering
    2. Multi-source analysis
    3. Enhanced RAG with external sources
    4. Detailed synthesis and summarization

    Args:
        user_id: User identifier for configuration retrieval
        tools: Available tools for this workflow
        pipeline_factory: Factory for creating pipeline instances

    Returns:
        Compiled LangGraph research workflow
    """
    composer_logger.logger.info(
        "Building research workflow", 
        extra={"user_id": user_id}
    )

    # Create research workflow graph
    workflow = StateGraph(ResearchWorkflowState)

    # Add research-specific nodes
    workflow.add_node("intent_classifier", IntentClassifierNode())

    # Query expansion for comprehensive research
    workflow.add_node(
        "query_expansion",
        PipelineNode(
            pipeline_factory,
            ModelProfileType.Analysis,
            stream=False,  # Analysis doesn't need streaming
        ),
    )

    # Enhanced RAG with multi-source capabilities
    workflow.add_node("deep_search", DeepSearchExecutor(user_id))

    # Synthesis agent for comprehensive results
    workflow.add_node(
        "synthesis_agent",
        PipelineNode(
            pipeline_factory,
            ModelProfileType.Primary,
            stream=True,  # Final synthesis can stream
        ),
    )

    # Set research workflow flow - linear progression for thorough analysis
    workflow.set_entry_point("intent_classifier")
    workflow.add_edge("intent_classifier", "query_expansion")
    workflow.add_edge("query_expansion", "deep_search")
    workflow.add_edge("deep_search", "synthesis_agent")
    workflow.add_edge("synthesis_agent", END)

    # Compile and return workflow
    compiled_workflow = workflow.compile()

    composer_logger.logger.info(
        "Research workflow built successfully",
        extra={"user_id": user_id, "node_count": len(workflow.nodes)}
    )

    return compiled_workflow


def get_research_workflow_config(user_id: str) -> Dict[str, Any]:
    """
    Get configuration specific to research workflows.

    Args:
        user_id: User identifier

    Returns:
        Research workflow configuration dictionary
    """
    return {
        "workflow_type": "research",
        "deep_rag_enabled": True,
        "multi_source_search": True,
        "synthesis_required": True,
        "streaming_synthesis": True,
        "comprehensive_analysis": True,
        "user_id": user_id,
    }
