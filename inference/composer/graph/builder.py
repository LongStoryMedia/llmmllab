"""
Simplified GraphBuilder with Dependency Injection - Focused coordinator using composition.
Uses clean factories and strategies with proper dependency injection pattern.
All agents, storage services, and model profiles are instantiated upfront and injected.
"""

from typing import TYPE_CHECKING

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START

from models import ModelProfileType, UserConfig, WorkflowType
from runner import PipelineFactory

from utils.model_profile import get_model_profile_for_task

# Import all agents
from composer.agents.classifier_agent import ClassifierAgent
from composer.agents.engineering_agent import EngineeringAgent
from composer.agents.memory_agent import MemoryAgent
from composer.agents.embedding_agent import EmbeddingAgent
from composer.agents.summarization_agent import SummarizationAgent

# Import all nodes
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
from composer.nodes.summary import ConsolidationNode, SearchSummaryNode

from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger

from .state import WorkflowState

if TYPE_CHECKING:
    from db import Storage
    from db.userconfig_storage import UserConfigStorage
    from db.conversation_storage import ConversationStorage
    from db.message_storage import MessageStorage
    from db.model_profile_storage import ModelProfileStorage
    from db.memory_storage import MemoryStorage
    from db.summary_storage import SummaryStorage
    from db.search_storage import SearchStorage
    from db.dynamic_tool_storage import DynamicToolStorage


class GraphBuilder:
    """
    Clean, focused GraphBuilder using dependency injection and composition.

    Responsibilities:
    - Create all agent and storage service instances upfront
    - Inject dependencies into nodes for proper separation of concerns
    - Coordinate workflow creation using factories
    - Provide simple public interface
    - Handle errors gracefully

    Does NOT handle:
    - Caching (delegated to CachedWorkflowFactory)
    - Complex routing (handled by dedicated routers)
    - Circuit breaking (separate concern)
    - Tool orchestration (separate nodes)
    """

    def __init__(
        self,
        storage: "Storage",
        pipeline_factory: PipelineFactory,
        user_config: UserConfig,
    ):
        """
        Initialize GraphBuilder with dependency injection.

        Args:
            storage: Storage instance for dependency injection
            pipeline_factory: PipelineFactory
        """
        # Core dependencies
        self.pipeline_factory = pipeline_factory
        self.user_config = user_config
        self.logger = llmmllogger.logger.bind(component="GraphBuilder")

        # Use storage.get_service for type safety and linter warnings avoidance
        self.user_config_storage: "UserConfigStorage" = storage.get_service(
            storage.user_config
        )
        self.conversation_storage: "ConversationStorage" = storage.get_service(
            storage.conversation
        )
        self.message_storage: "MessageStorage" = storage.get_service(storage.message)
        self.model_profile_storage: "ModelProfileStorage" = storage.get_service(
            storage.model_profile
        )
        self.memory_storage: "MemoryStorage" = storage.get_service(storage.memory)
        self.summary_storage: "SummaryStorage" = storage.get_service(storage.summary)
        self.search_storage: "SearchStorage" = storage.get_service(storage.search)
        self.dynamic_tool_storage: "DynamicToolStorage" = storage.get_service(
            storage.dynamic_tool
        )

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
            primary_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.Primary,
                self.user_config.user_id,
            )
            analysis_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.Analysis,
                self.user_config.user_id,
            )
            engineering_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.Engineering,
                self.user_config.user_id,
            )
            embedding_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.Embedding,
                self.user_config.user_id,
            )

            # Create agents with injected dependencies
            classifier_agent = ClassifierAgent(
                self.pipeline_factory,
                analysis_profile,
            )
            engineering_agent = EngineeringAgent(
                self.pipeline_factory,
                engineering_profile,
            )
            memory_agent = MemoryAgent(memory_storage=self.memory_storage)
            embedding_agent = EmbeddingAgent(
                self.pipeline_factory,
                embedding_profile,
            )
            summarization_agent = SummarizationAgent(
                pipeline_factory=self.pipeline_factory,
                summary_storage=self.summary_storage,
                search_storage=self.search_storage,
                user_config=self.user_config,
            )
            # Create tool registry (also depends on embedding agent)
            tool_registry = ToolRegistry(self.pipeline_factory, embedding_agent)

            self.logger.info(
                "Building workflow with dependency injection", user_id=user_id
            )
            # Create master workflow graph
            workflow = StateGraph(WorkflowState)

            # Create nodes with injected dependencies
            # Intent analysis -> router -> (optional specialized agents) pattern
            workflow.add_node(
                "intent_analysis",
                IntentClassifierNode(classifier_agent),
            )
            workflow.add_node("workflow_router", WorkflowRouter(user_id))

            # Engineering agent (invoked only when routing selects engineering)
            workflow.add_node(
                "engineering_agent",
                EngineeringAgentNode(engineering_agent),
            )

            # Title generation (if no title exists)
            workflow.add_node(
                "title_generation",
                TitleGenerationNode(self.pipeline_factory, classifier_agent),
            )

            # Memory nodes with injected agents and storage
            workflow.add_node(
                "memory_search",
                MemorySearchNode(
                    memory_agent,
                    embedding_agent,
                ),
            )
            workflow.add_node(
                "memory_creation",
                MemoryCreationNode(embedding_agent),
            )
            workflow.add_node(
                "memory_storage",
                MemoryStorageNode(memory_agent),
            )

            # Tool nodes with injected dependencies
            workflow.add_node(
                "static_tool_collection", StaticToolCollectionNode(tool_registry)
            )
            workflow.add_node(
                "dynamic_tool_collection",
                DynamicToolCreationNode(
                    tool_registry,
                    self.pipeline_factory,
                ),
            )
            workflow.add_node("tool_composer", ToolComposerNode())
            workflow.add_node("tool_executor", ToolExecutorNode(tool_registry))

            workflow.add_node("chat_summary", ConsolidationNode(summarization_agent))
            workflow.add_node("search_summary", SearchSummaryNode(summarization_agent))

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
            workflow.add_edge(START, "memory_search")

            # 2. Intent Analysis -> Sequential tool collection and memory search
            workflow.add_edge("intent_analysis", "static_tool_collection")
            workflow.add_edge("intent_analysis", "dynamic_tool_collection")
            workflow.add_edge("dynamic_tool_collection", "tool_composer")
            workflow.add_edge("static_tool_collection", "tool_composer")

            # 3. Memory search -> Router for workflow selection
            workflow.add_edge("tool_composer", "workflow_router")
            workflow.add_edge("memory_search", "workflow_router")

            # 5. Conditional routing: router decides next step based on complexity
            def route_post_router(state: WorkflowState):
                # If engineering workflow selected, use specialized agent first
                # add more workflows here as needed
                if WorkflowType.ENGINEERING in state.selected_workflows:
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

                # If last message is from assistant and has tool calls, execute tools
                if (
                    hasattr(last_message, "type")
                    and last_message.type == "ai"
                    and hasattr(last_message, "tool_calls")
                    and last_message.tool_calls
                ):
                    return "tool_executor"

                # Otherwise, proceed to chat summary
                return "chat_summary"

            workflow.add_conditional_edges(
                "chat_agent",
                should_execute_tools,
                {
                    "tool_executor": "tool_executor",
                    "memory_creation": "memory_creation",
                    "chat_summary": "chat_summary",
                },
            )

            # 8. Conditional routing from tool executor - check if web search added results to state
            def should_synthesize_search_results(state: WorkflowState):
                # Check if any search results were added to state (by Command from web search tool)
                if state.web_search_results:
                    return "search_summary"
                return "chat_agent"

            workflow.add_conditional_edges(
                "tool_executor",
                should_synthesize_search_results,
                {
                    "search_summary": "search_summary",
                    "chat_agent": "chat_agent",
                },
            )

            # 8b. Search summary -> Chat agent (for final response with synthesized search results)
            workflow.add_edge("search_summary", "chat_agent")

            workflow.add_edge("chat_summary", "title_generation")

            # 9. Memory and title generation happen after final response
            workflow.add_edge("title_generation", "memory_creation")

            # 10. Title generation -> Memory storage -> End
            workflow.add_edge("memory_creation", "memory_storage")
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
