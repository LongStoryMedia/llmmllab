"""
Simplified GraphBuilder - Focused coordinator using composition.
Uses clean factories and strategies instead of monolithic implementation.
"""

# Datetime not used in this module
from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START

from models import ModelProfileType
from runner import PipelineFactory

from composer.nodes.routing import IntentClassifierNode
from composer.nodes.routing.router import WorkflowRouter
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
from composer.nodes.agents import TitleGenerationNode
from composer.nodes.agents.engineering import EngineeringAgentNode

from composer.tools.registry import ToolRegistry

from composer.monitoring.logging import composer_logger

from .state import WorkflowState


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
        # Keep direct reference for node construction and registry usage
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger.bind(component="GraphBuilder")

    async def build_workflow(
        self,
        user_id: str,
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

            # Intent analysis -> router -> (optional specialized agents) pattern
            workflow.add_node("intent_analysis", IntentClassifierNode())
            workflow.add_node("workflow_router", WorkflowRouter(user_id))
            # Engineering agent (invoked only when routing selects engineering)
            workflow.add_node(
                "engineering_agent", EngineeringAgentNode(self.pipeline_factory)
            )

            # Title generation (if no title exists)
            workflow.add_node(
                "title_generation", TitleGenerationNode(self.pipeline_factory)
            )

            # Memory
            workflow.add_node("memory_search", MemorySearchNode(self.pipeline_factory))
            workflow.add_node(
                "memory_creation", MemoryCreationNode(self.pipeline_factory)
            )
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
            workflow.add_node("tool_executor", ToolExecutorNode(tool_registry))

            # Primary chat agent with streaming enabled
            workflow.add_node(
                "chat_agent",
                PipelineNode(
                    self.pipeline_factory, ModelProfileType.Primary, stream=True
                ),
            )

            # Build a logical workflow graph structure:
            # 1. Start -> Intent Analysis
            workflow.add_edge(START, "intent_analysis")

            # 2. Intent Analysis -> Sequential tool collection and memory search
            workflow.add_edge("intent_analysis", "static_tool_collection")
            workflow.add_edge("static_tool_collection", "dynamic_tool_collection")
            workflow.add_edge("dynamic_tool_collection", "tool_composer")
            workflow.add_edge("tool_composer", "memory_search")

            # 3. Memory search -> Router for workflow selection
            workflow.add_edge("memory_search", "workflow_router")

            # 5. Conditional routing: router decides next step based on complexity
            def route_post_router(state: WorkflowState):
                # If engineering workflow selected, use specialized agent first
                if (
                    state.selected_workflows
                    and "engineering" in state.selected_workflows
                ):
                    return "engineering_agent"
                # Otherwise go straight to primary chat agent
                return "chat_agent"

            workflow.add_conditional_edges(
                "workflow_router",
                route_post_router,
                {
                    "engineering_agent": "engineering_agent",
                    "chat_agent": "chat_agent",
                },
            )

            # 6. Engineering agent -> Chat agent (for final response)
            workflow.add_edge("engineering_agent", "chat_agent")

            # 7. Conditional routing from chat agent based on tool calls
            def should_execute_tools(state: WorkflowState):
                if not state.messages:
                    return "memory_creation"

                last_message = state.messages[-1]

                # If last message is a tool result, continue to memory creation
                if hasattr(last_message, "type") and last_message.type == "tool":
                    return "memory_creation"

                # If last message has tool calls, execute tools
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    return "tool_executor"

                # No tool calls, go to memory creation
                return "memory_creation"

            workflow.add_conditional_edges(
                "chat_agent",
                should_execute_tools,
                {
                    "tool_executor": "tool_executor",
                    "memory_creation": "memory_creation",
                },
            )

            # 8. Tool executor -> Chat agent (for final response with tool results)
            workflow.add_edge("tool_executor", "chat_agent")

            # 9. Memory and title generation happen after final response
            workflow.add_edge("memory_creation", "title_generation")

            # 10. Title generation -> Memory storage -> End
            workflow.add_edge("title_generation", "memory_storage")
            workflow.add_edge("memory_storage", END)

            return workflow.compile()
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            # Try to create fallback chat workflow
            raise
