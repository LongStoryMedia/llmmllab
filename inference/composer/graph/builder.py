"""
Simplified GraphBuilder with Dependency Injection - Focused coordinator using composition.
Uses clean factories and strategies with proper dependency injection pattern.
All agents, storage services, and model profiles are instantiated upfront and injected.
"""

from typing import TYPE_CHECKING
import uuid

from composer.agents.chat_agent import ChatAgent
from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START

from models import (
    ModelProfileType,
    PipelinePriority,
    UserConfig,
    WorkflowType,
    NodeMetadata,
)
from runner import PipelineFactory

from utils.model_profile import get_model_profile_for_task
from utils.logging import llmmllogger

# Import all agents
from composer.agents.classifier_agent import ClassifierAgent
from composer.agents.engineering_agent import EngineeringAgent
from composer.agents.memory_agent import MemoryAgent
from composer.agents.embedding_agent import EmbeddingAgent
from composer.agents.primary_summary_agent import PrimarySummaryAgent
from composer.agents.master_summary_agent import MasterSummaryAgent

# Import all nodes
from composer.nodes.routing import IntentClassifierNode
from composer.nodes.routing.router import WorkflowRouter
from composer.nodes.tools import (
    ToolCollectionNode,
    ToolComposerNode,
    StaticToolLoadingNode,
)
from composer.nodes.memory import (
    MemorySearchNode,
    MemoryCreationNode,
    MemoryStorageNode,
)
from composer.nodes.agents import TitleGenerationNode
from composer.nodes.agents.engineering import EngineeringAgentNode
from composer.nodes.summary import ConsolidationNode, SearchSummaryNode
from composer.tools.registry import ToolRegistry

from composer.graph.state import WorkflowState
from composer.graph.subgraphs import ToolsAgentSubgraph

# Checkpoint integration handled through CheckpointStorage service

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
    from db.checkpoint_storage import CheckpointStorage


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
        self.checkpoint_storage: "CheckpointStorage" = storage.get_service(
            storage.checkpoint
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
                ModelProfileType.PrimarySummary,
                self.user_config.user_id,
            )
            analysis_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.Analysis,
                self.user_config.user_id,
            )
            memory_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.MemoryRetrieval,
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
            summarization_profile = await get_model_profile_for_task(
                self.user_config.model_profiles,
                ModelProfileType.PrimarySummary,
                self.user_config.user_id,
            )

            # Node metadata for logging and tracing
            primary_node_metadata = NodeMetadata(
                node_name="PrimaryChatAgent",
                node_id=uuid.uuid4().hex,
                node_type="ChatNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )
            classifier_node_metadata = NodeMetadata(
                node_name="IntentClassifier",
                node_id=uuid.uuid4().hex,
                node_type="IntentClassifierNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )
            engineering_node_metadata = NodeMetadata(
                node_name="EngineeringAgent",
                node_id=uuid.uuid4().hex,
                node_type="EngineeringAgentNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )
            memory_node_metadata = NodeMetadata(
                node_name="MemoryAgent",
                node_id=uuid.uuid4().hex,
                node_type="MemoryAgentNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )
            embedding_node_metadata = NodeMetadata(
                node_name="EmbeddingAgent",
                node_id=uuid.uuid4().hex,
                node_type="EmbeddingAgentNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )
            primary_summary_node_metadata = NodeMetadata(
                node_name="PrimarySummaryAgent",
                node_id=uuid.uuid4().hex,
                node_type="PrimarySummaryAgentNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )
            master_summary_node_metadata = NodeMetadata(
                node_name="MasterSummaryAgent",
                node_id=uuid.uuid4().hex,
                node_type="MasterSummaryAgentNode",
                execution_time=None,
                user_id=user_id,
                conversation_id=None,
                profile_type=None,
                streaming=None,
                is_cached=None,
                cache_key=None,
                tool_count=None,
            )

            # Create agents with injected dependencies
            primary_agent = ChatAgent(
                pipeline_factory=self.pipeline_factory,
                profile=primary_profile,
                node_metadata=primary_node_metadata,
                priority=PipelinePriority.HIGH,
            )
            classifier_agent = ClassifierAgent(
                self.pipeline_factory,
                analysis_profile,
                classifier_node_metadata,
            )
            engineering_agent = EngineeringAgent(
                self.pipeline_factory,
                engineering_profile,
                engineering_node_metadata,
                self.dynamic_tool_storage,
            )
            memory_agent = MemoryAgent(
                self.pipeline_factory,
                memory_profile,
                memory_node_metadata,
                self.memory_storage,
            )
            embedding_agent = EmbeddingAgent(
                self.pipeline_factory,
                embedding_profile,
                embedding_node_metadata,
            )
            primary_summary_agent = PrimarySummaryAgent(
                self.pipeline_factory,
                summarization_profile,
                primary_summary_node_metadata,
                self.summary_storage,
                self.search_storage,
                self.user_config,
            )
            master_summary_agent = MasterSummaryAgent(
                self.pipeline_factory,
                summarization_profile,
                master_summary_node_metadata,
                self.summary_storage,
                self.search_storage,
                self.user_config,
            )

            # Create tool registry (also depends on embedding agent)
            tool_registry = ToolRegistry(self.pipeline_factory)

            # Create nodes with injected agents and storage
            classifier_node = IntentClassifierNode(classifier_agent)
            engineering_node = EngineeringAgentNode(engineering_agent)
            memory_creation_node = MemoryCreationNode(embedding_agent)
            memory_search_node = MemorySearchNode(
                memory_agent,
                embedding_agent,
            )
            memory_storage_node = MemoryStorageNode(memory_agent)
            title_generation_node = TitleGenerationNode(
                self.pipeline_factory,
                classifier_agent,
            )
            # Import here to avoid linting issues
            static_tool_loading_node = StaticToolLoadingNode(
                tool_registry,
                self.dynamic_tool_storage,
            )
            tool_collection_node = ToolCollectionNode(
                tool_registry,
                engineering_agent,
            )
            tool_composer_node = ToolComposerNode()

            # ConsolidationNode needs both primary (for conversation summaries) and master (for consolidation)
            chat_summary_node = ConsolidationNode(
                primary_summary_agent, master_summary_agent
            )
            # SearchSummaryNode uses primary summaries by default
            search_summary_node = SearchSummaryNode(primary_summary_agent)

            router_node = WorkflowRouter(user_id)

            self.logger.info(
                "Building workflow with dependency injection", user_id=user_id
            )

            # Create master workflow graph
            workflow = StateGraph(WorkflowState)

            # Create nodes with injected dependencies
            # Intent analysis -> router -> (optional specialized agents) pattern
            workflow.add_node("intent_analysis", classifier_node)
            workflow.add_node("workflow_router", router_node)

            # Engineering agent (invoked only when routing selects engineering)
            workflow.add_node("engineering_agent", engineering_node)

            # Title generation (if no title exists)
            workflow.add_node("title_generation", title_generation_node)

            # Memory nodes with injected agents and storage
            workflow.add_node("memory_search", memory_search_node)
            workflow.add_node("memory_creation", memory_creation_node)
            workflow.add_node("memory_storage", memory_storage_node)

            # Static tool loading node - loads static tools and previous dynamic tools early
            workflow.add_node("static_tool_loading", static_tool_loading_node)

            # Tool collection node with injected dependencies
            workflow.add_node("tool_collection", tool_collection_node)
            workflow.add_node("tool_composer", tool_composer_node)

            workflow.add_node("chat_summary", chat_summary_node)
            workflow.add_node("search_summary", search_summary_node)

            tools_agent_subgraph = ToolsAgentSubgraph(
                tool_registry=tool_registry,
                chat_agent=primary_agent,
            )

            # Create wrapper for subgraph execution
            async def tools_agent_node(state: WorkflowState) -> WorkflowState:
                """Execute the intelligent tools agent subgraph and return updated state."""
                command = await tools_agent_subgraph.execute(state)
                if command and command.update:
                    for key, value in command.update.items():
                        setattr(state, key, value)
                return state

            workflow.add_node("tools_agent", tools_agent_node)

            # Build a logical workflow graph structure:
            # 1. Start -> Static tool loading (loads static tools + previous dynamic tools)
            workflow.add_edge(START, "memory_search")
            workflow.add_edge("memory_search", "static_tool_loading")

            # 2. Static tool loading -> Intent Analysis (classifier can now see available tools)
            workflow.add_edge("static_tool_loading", "intent_analysis")

            # 3. Intent Analysis -> Tool collection (filters static tools + creates dynamic tools)
            workflow.add_edge("intent_analysis", "tool_collection")
            workflow.add_edge("tool_collection", "tool_composer")

            # 4. Tool composer -> Router for workflow selection
            workflow.add_edge("tool_composer", "workflow_router")

            # 5. Conditional routing: router decides next step based on complexity
            def route_post_router(state: WorkflowState):
                # If engineering workflow selected, use specialized agent first
                # add more workflows here as needed
                if WorkflowType.ENGINEERING in state.selected_workflows:
                    return "engineering_agent"
                # Otherwise go to intelligent tools agent subgraph (handles chat + tools + cycling)
                return "tools_agent"

            workflow.add_conditional_edges(
                "workflow_router",
                route_post_router,
                {
                    "engineering_agent": "engineering_agent",
                    "tools_agent": "tools_agent",
                },
            )

            # 6. Engineering agent -> Tools agent (subgraph handles intelligent agent cycling)
            workflow.add_edge("engineering_agent", "tools_agent")

            # 7. Simple routing: tools_agent -> search_summary (if web search) or chat_summary
            def route_after_tools_agent(state: WorkflowState):
                """Route after intelligent tools agent completes."""
                # Check if web search was performed and needs summarization
                if hasattr(state, "web_search_results") and state.web_search_results:
                    self.logger.info(
                        "🔀 Tools agent completed with web search results - routing to search_summary"
                    )
                    return "search_summary"

                # Otherwise proceed to chat summary for consolidation
                self.logger.info("🔀 Tools agent completed - routing to chat_summary")
                return "chat_summary"

            workflow.add_conditional_edges(
                "tools_agent",
                route_after_tools_agent,
                {
                    "search_summary": "search_summary",
                    "chat_summary": "chat_summary",
                },
            )

            # 9. Linear flow after agent completion with conditional title generation
            def route_after_chat_summary(state: WorkflowState):
                """Route after chat summary - conditionally generate title."""
                # Check if title already exists
                if hasattr(state, "title") and state.title and state.title.strip():
                    self.logger.info(
                        "🔀 Title already exists - skipping title generation"
                    )
                    return "memory_creation"
                else:
                    self.logger.info("🔀 No title exists - routing to title generation")
                    return "title_generation"

            workflow.add_edge("search_summary", "chat_summary")
            workflow.add_conditional_edges(
                "chat_summary",
                route_after_chat_summary,
                {
                    "title_generation": "title_generation",
                    "memory_creation": "memory_creation",
                },
            )
            workflow.add_edge("title_generation", "memory_creation")

            # 10. Memory storage -> End (both from dual-loop exit and normal flow)
            workflow.add_edge("memory_creation", "memory_storage")
            workflow.add_edge("memory_storage", END)

            # Configure checkpointer at compilation time for parent graph
            # Per LangGraph docs: "you only need to provide the checkpointer when compiling 
            # the parent graph. LangGraph will automatically propagate the checkpointer to child subgraphs"
            try:
                if self.checkpoint_storage.is_initialized():
                    # Use LangGraph's standard production pattern
                    async with self.checkpoint_storage.create_checkpointer() as checkpointer:
                        self.logger.info(
                            "✅ Compiling workflow with checkpointer - will auto-propagate to subgraphs"
                        )
                        # Compile with checkpointer - automatically propagates to all subgraphs
                        compiled_workflow = workflow.compile(checkpointer=checkpointer)
                        return compiled_workflow
                else:
                    self.logger.info(
                        "ℹ️  Checkpoint storage not initialized - compiling without persistence"
                    )
                    return workflow.compile()

            except Exception as e:
                self.logger.warning(f"⚠️  Checkpointer setup failed, compiling without persistence: {e}")
                # Fallback to compilation without checkpointer
                return workflow.compile()
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            # Try to create fallback chat workflow
            raise
