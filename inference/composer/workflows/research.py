"""
Research workflow implementation for composer.
Implements research workflow with deep search and comprehensive synthesis.
"""

from typing import Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import ModelProfileType

from composer.graph.state import ResearchWorkflowState
from composer.nodes import PipelineNode
from composer.nodes.routing import IntentClassifierNode
from composer.nodes.research import ComprehensiveResearchExecutor
from utils.logging import llmmllogger


async def build_research_workflow(
    user_id: str, pipeline_factory: Any = None, use_search_orchestration: bool = True
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
        pipeline_factory: Factory for creating pipeline instances

    Returns:
        Compiled LangGraph research workflow
    """
    llmmllogger.logger.info("Building research workflow", extra={"user_id": user_id})

    # Create research workflow graph
    workflow = StateGraph(ResearchWorkflowState)

    # Add initial configuration node to retrieve and set web search config
    workflow.add_node("config_setup", _create_config_setup_node(user_id))

    # Add research-specific nodes
    workflow.add_node("classifier_agent", IntentClassifierNode())

    # Query expansion for comprehensive research
    workflow.add_node(
        "query_expansion",
        PipelineNode(
            pipeline_factory,
            ModelProfileType.Analysis,
            stream=False,  # Analysis doesn't need streaming
        ),
    )

    # Enhanced RAG with multi-source capabilities - optionally use new orchestration
    if use_search_orchestration:
        from composer.nodes.processing import (
            WebSearchOrchestrationNode,
        )  # pylint: disable=import-outside-toplevel

        workflow.add_node("web_search_orchestration", WebSearchOrchestrationNode())
        workflow.add_node(
            "comprehensive_research", ComprehensiveResearchExecutor(user_id)
        )

        # Insert orchestration in the flow
        workflow.add_edge("query_expansion", "web_search_orchestration")
        workflow.add_edge("web_search_orchestration", "comprehensive_research")
    else:
        workflow.add_node(
            "comprehensive_research", ComprehensiveResearchExecutor(user_id)
        )
        workflow.add_edge("query_expansion", "comprehensive_research")

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
    workflow.set_entry_point("config_setup")
    workflow.add_edge("config_setup", "classifier_agent")
    workflow.add_edge("classifier_agent", "query_expansion")
    # Query expansion edges are handled above based on orchestration choice
    workflow.add_edge("comprehensive_research", "synthesis_agent")
    workflow.add_edge("synthesis_agent", END)

    # Compile and return workflow
    compiled_workflow = workflow.compile()

    llmmllogger.logger.info(
        "Research workflow built successfully",
        extra={"user_id": user_id, "node_count": len(workflow.nodes)},
    )

    return compiled_workflow


def _create_config_setup_node(user_id: str):
    """
    Create a node that retrieves and sets up WebSearchConfig in the workflow state.

    Args:
        user_id: User identifier for configuration retrieval

    Returns:
        Callable node function
    """

    async def config_setup(state: ResearchWorkflowState) -> ResearchWorkflowState:
        """
        Retrieve WebSearchConfig from shared data layer and store in state.
        """
        try:
            from db import storage  # pylint: disable=import-outside-toplevel
            from models.web_search_config import (
                WebSearchConfig,
            )  # pylint: disable=import-outside-toplevel

            # Get user configuration from shared data layer
            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)

            if user_config and hasattr(user_config, "web_search"):
                web_search_config = user_config.web_search
            else:
                # Use default config if no user-specific config found
                llmmllogger.logger.info(
                    f"Using default web search config for research workflow, user {user_id}"
                )
                web_search_config = WebSearchConfig(
                    max_results=10,  # More results for research
                    max_urls_deep=5,  # Deep content extraction for research
                    engines=[
                        "google",
                        "bing",
                        "duckduckgo",
                        "arxiv",
                        "wikipedia",
                    ],  # Research-focused engines
                    categories=[
                        "general",
                        "science",
                        "news",
                    ],  # Research-relevant categories
                    include_results=True,
                    enable_caching=True,
                )

            # Store config in workflow state metadata
            if not state.execution_metadata:
                state.execution_metadata = {}
            state.execution_metadata["web_search_config"] = web_search_config

            # Set research-specific search parameters
            state.execution_metadata["search_depth"] = (
                "DEEP"  # Always use deep search for research
            )
            state.execution_metadata["max_search_results"] = (
                web_search_config.max_results
            )

            llmmllogger.logger.info(
                "Research workflow config setup completed",
                extra={
                    "user_id": user_id,
                    "web_search_enabled": web_search_config.enabled,
                    "max_results": web_search_config.max_results,
                    "engines": len(web_search_config.engines),
                },
            )

        except Exception as e:
            llmmllogger.logger.warning(
                f"Failed to set up research workflow config for user {user_id}: {e}"
            )
            # Continue with default settings - don't fail the workflow
            if not state.execution_metadata:
                state.execution_metadata = {}
            state.execution_metadata["search_depth"] = "DEEP"
            state.execution_metadata["max_search_results"] = 10

        return state

    return config_setup


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
