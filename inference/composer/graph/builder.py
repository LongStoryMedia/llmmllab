"""
Simplified GraphBuilder - Focused coordinator using composition.
Uses clean factories and strategies instead of monolithic implementation.
"""

from typing import Dict, Any
from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START

from models import WorkflowType, ModelProfileType
from runner import PipelineFactory

from composer.nodes.routing import IntentClassifierNode
from composer.nodes.tools import (
    StaticToolCollectionNode,
    DynamicToolCreationNode,
    ToolComposerNode,
    ToolExecutorNode,
)
from composer.nodes.infrastructure import PipelineNode
from composer.nodes.memory import (
    MemorySearchNode,
    MemoryCreationNode,
    MemoryStorageNode,
)
# Summary nodes not used in simplified builder

from composer.tools.registry import ToolRegistry

from composer.monitoring.logging import composer_logger
from composer.core.errors import WorkflowConstructionError
from .cached_workflow_factory import CachedWorkflowFactory
from .state import WorkflowState
import datetime


class GraphBuilder:
    """
    Clean, focused GraphBuilder using composition over inheritance.

    Responsibilities:
    - Coordinate workflow creation using factories
    - Provide simple public interface
    - Handle errors gracefully

    Does NOT handle:
    - Caching (delegated to CachedWorkflowFactory)
    - Complex routing (handled by dedicated routers)
    - Circuit breaking (separate concern)
    - Tool orchestration (separate nodes)
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        if not pipeline_factory:
            raise ValueError("pipeline_factory is required")

        self.workflow_factory = CachedWorkflowFactory(pipeline_factory)
        # Keep direct reference for node construction and registry usage
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger.bind(component="GraphBuilder")

    async def build_workflow(
    self,
    user_id: str,
    use_cache: bool = True,  # noqa: ARG002
    ) -> CompiledStateGraph:
        """
        Build a workflow of the specified type.

        Simple delegation to workflow factory with error handling.

        Args:
            workflow_type: Type of workflow to build
            user_id: User identifier
            use_cache: Whether to use caching
            **kwargs: Additional workflow parameters

        Returns:
            Compiled workflow ready for execution
        """
        try:
            self.logger.info("Building workflow", user_id=user_id)
            # Create master workflow graph
            workflow = StateGraph(WorkflowState)
            tool_registry = ToolRegistry(self.pipeline_factory)

            # Add intent analysis node (always present for context enrichment)
            workflow.add_node("intent_analysis", IntentClassifierNode())

            # Memory
            workflow.add_node("memory_search", MemorySearchNode(self.pipeline_factory))
            workflow.add_node("memory_creation", MemoryCreationNode(self.pipeline_factory))
            workflow.add_node("memory_storage", MemoryStorageNode())

            # Tools
            workflow.add_node(
                "static_tool_collection", StaticToolCollectionNode(tool_registry)
            )
            workflow.add_node(
                "dynamic_tool_collection",
                DynamicToolCreationNode(tool_registry, self.pipeline_factory),
            )
            workflow.add_node("tool_composer", ToolComposerNode())
            workflow.add_node("tool_executor", ToolExecutorNode())

            # Primary chat agent with streaming enabled
            workflow.add_node(
                "chat_agent",
                PipelineNode(self.pipeline_factory, ModelProfileType.Primary, stream=True),
            )

            workflow.add_edge(START, "intent_analysis")
            workflow.add_edge(START, "memory_search")
            workflow.add_edge("intent_analysis", "static_tool_collection")
            workflow.add_edge("intent_analysis", "dynamic_tool_collection")
            workflow.add_edge("static_tool_collection", "tool_composer")
            workflow.add_edge("dynamic_tool_collection", "tool_composer")
            workflow.add_edge("tool_composer", "tool_executor")
            workflow.add_edge("tool_executor", "chat_agent")
            workflow.add_edge("chat_agent", "memory_creation")
            workflow.add_edge("memory_creation", "memory_storage")
            workflow.add_edge("memory_storage", END)

            csg = workflow.compile()

            # if os.getenv("DEBUG_MODE", "0") == "1":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/app/workflow_graph_{timestamp}.png"
            bts = csg.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(bts)
            self.logger.info("Workflow graph saved", path=output_path)

            return csg
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            # Try to create fallback chat workflow
            return await self._create_fallback_workflow(user_id)

    async def _create_fallback_workflow(self, user_id: str) -> CompiledStateGraph:
        """Create minimal fallback workflow when everything else fails."""
        try:
            self.logger.warning("Creating fallback workflow", user_id=user_id)
            return await self.workflow_factory.create_workflow(
                WorkflowType.CHAT, user_id, use_cache=False
            )
        except Exception as e:
            self.logger.error(
                "Fallback workflow creation failed", user_id=user_id, error=str(e)
            )
            raise WorkflowConstructionError(
                f"Cannot create any workflow, including fallback: {e}"
            ) from e

    # Cache management methods (delegate to factory)
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.workflow_factory.get_cache_stats()

    async def clear_cache(self) -> None:
        """Clear workflow cache."""
        await self.workflow_factory.invalidate_cache()

    async def invalidate_user_workflows(self, user_id: str) -> None:
        """Invalidate workflows for specific user."""
        await self.workflow_factory.invalidate_cache(user_id)

    async def close(self) -> None:
        """Clean up resources."""
        await self.workflow_factory.close()
        self.logger.info("GraphBuilder closed")
