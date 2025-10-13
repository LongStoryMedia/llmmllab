"""
Enhanced workflow example demonstrating tool orchestration subgraph integration.
Shows how workflows can use the sophisticated tool generation capabilities.
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from composer.graph.state import WorkflowState
from composer.graph.tools.tool_orchestration import (
    ToolOrchestrationState,
    create_tool_orchestration_subgraph,
)
from composer.nodes.routing import IntentClassifierNode
from composer.nodes.agents import EngineeringAgentNode
from composer.nodes.agents.response_format_analysis import ResponseFormatAnalysisNode
from utils.logging import llmmllogger
from composer.tools.registry import ToolRegistry
from composer.utils.extraction import get_most_recent_user_message_content
from runner import PipelineFactory


async def build_enhanced_engineering_workflow(
    pipeline_factory: PipelineFactory, user_id: str
) -> CompiledStateGraph:
    """
    Build an enhanced engineering workflow that demonstrates sophisticated tool orchestration.

    This workflow shows how to integrate the Tool Orchestration Subgraph with other nodes
    for complex engineering tasks that may require dynamic tool generation.

    Args:
        pipeline_factory: Factory for creating LLM pipelines
        user_id: User identifier for configuration

    Returns:
        Compiled workflow with tool orchestration integration
    """
    try:
        llmmllogger.logger.info(
            "Building enhanced engineering workflow", extra={"user_id": user_id}
        )

        # Create main workflow graph
        workflow = StateGraph(WorkflowState)

        # Add intent analysis node
        workflow.add_node("intent_analysis", IntentClassifierNode())

        # Add response format analysis node (determines format and domain based on sophisticated analysis)
        workflow.add_node(
            "response_format_analysis", ResponseFormatAnalysisNode(pipeline_factory)
        )

        # Add tool orchestration as a subgraph
        tool_orchestration_subgraph = await create_tool_orchestration_subgraph(
            pipeline_factory=pipeline_factory,
            tool_registry=ToolRegistry(pipeline_factory),
        )

        # Create a bridge node to convert between state types
        workflow.add_node(
            "tool_orchestration",
            _create_tool_orchestration_bridge_node(
                tool_orchestration_subgraph, user_id
            ),
        )

        # Add engineering agent node that uses the orchestrated tools
        workflow.add_node(
            "engineering_response", EngineeringAgentNode(pipeline_factory)
        )

        # Define the enhanced workflow flow with response format analysis
        workflow.set_entry_point("intent_analysis")
        workflow.add_edge("intent_analysis", "response_format_analysis")
        workflow.add_edge("response_format_analysis", "tool_orchestration")
        workflow.add_edge("tool_orchestration", "engineering_response")
        workflow.add_edge("engineering_response", END)

        compiled_workflow = workflow.compile()

        llmmllogger.logger.info(
            "Enhanced engineering workflow built successfully",
            extra={"user_id": user_id},
        )

        return compiled_workflow

    except Exception as e:
        llmmllogger.logger.error(
            f"Failed to build enhanced engineering workflow: {e}",
            extra={"user_id": user_id},
        )
        raise


def _create_tool_orchestration_bridge_node(
    tool_orchestration_subgraph: CompiledStateGraph, user_id: str
):
    """
    Create a bridge node that converts WorkflowState to ToolOrchestrationState and back.

    This demonstrates how to integrate subgraphs with different state types into workflows.
    """

    async def tool_orchestration_bridge(state: WorkflowState) -> WorkflowState:
        """
        Bridge node that:
        1. Converts WorkflowState to ToolOrchestrationState
        2. Runs the tool orchestration subgraph
        3. Converts results back to WorkflowState
        """
        try:
            if not state.intent_classification:
                llmmllogger.logger.warning(
                    "No intent classification found in state; skipping tool orchestration",
                    extra={"user_id": user_id},
                )
                state.available_tools = []
                return state

            llmmllogger.logger.info(
                "Executing tool orchestration subgraph",
                extra={
                    "user_id": user_id,
                    "has_intent_analysis": bool(state.intent_classification),
                    "messages_count": len(state.messages or []),
                },
            )

            # Extract user query from messages using langgraph utility
            user_query = get_most_recent_user_message_content(state.messages or [])

            # Create ToolOrchestrationState from WorkflowState as Pydantic model
            # ToolOrchestrationState expects a single IntentAnalysis, take the most recent
            intent_obj = state.intent_classification[-1]
            tool_state = ToolOrchestrationState(
                user_id=user_id,
                user_query=user_query,
                intent_analysis=intent_obj,
            )

            # Execute the tool orchestration subgraph
            result = await tool_orchestration_subgraph.ainvoke(tool_state)

            # Update WorkflowState with orchestration results
            orchestrated = result.get("orchestrated_tools", [])
            state.available_tools = orchestrated
            # Maintain separate tracking if provided
            state.dynamic_tools = result.get("dynamic_tools", [])
            state.static_tools = result.get("static_tools", [])

            # Add tool metadata to state execution metadata using strongly typed methods
            state.execution_metadata.update_tool_orchestration(
                tool_metadata=result.get("tool_metadata") or {},
                errors=result.get("errors") or [],
                dynamic_tools_count=len(result.get("dynamic_tools") or []),
                static_tools_count=len(result.get("static_tools") or []),
            )

            llmmllogger.logger.info(
                "Tool orchestration completed",
                extra={
                    "user_id": user_id,
                    "total_tools": len(state.available_tools),
                    "orchestration_success": state.execution_metadata.orchestration_success
                    or False,
                },
            )

            return state

        except Exception as e:
            llmmllogger.logger.error(
                f"Tool orchestration bridge failed: {e}", extra={"user_id": user_id}
            )

            # Fallback - ensure state has empty tools list
            state.available_tools = state.available_tools or []

            # Add error to strongly typed metadata
            state.execution_metadata.tool_orchestration_error = str(e)

            return state

    return tool_orchestration_bridge


async def build_simple_tool_orchestration_workflow(
    pipeline_factory: PipelineFactory, user_id: str
) -> CompiledStateGraph:
    """
    Build a simple workflow focused purely on tool orchestration.

    This is useful for testing tool generation capabilities in isolation.
    """
    try:
        # Create a minimal workflow that just does tool orchestration
        workflow = StateGraph(WorkflowState)

        # Add intent analysis
        workflow.add_node("intent_analysis", IntentClassifierNode())

        # Add tool orchestration bridge
        tool_orchestration_subgraph = await create_tool_orchestration_subgraph(
            pipeline_factory=pipeline_factory,
            tool_registry=ToolRegistry(pipeline_factory),
        )

        workflow.add_node(
            "tool_orchestration",
            _create_tool_orchestration_bridge_node(
                tool_orchestration_subgraph, user_id
            ),
        )

        # Set up simple flow
        workflow.set_entry_point("intent_analysis")
        workflow.add_edge("intent_analysis", "tool_orchestration")
        workflow.add_edge("tool_orchestration", END)

        return workflow.compile()

    except Exception as e:
        llmmllogger.logger.error(
            f"Failed to build simple tool orchestration workflow: {e}"
        )
        raise
