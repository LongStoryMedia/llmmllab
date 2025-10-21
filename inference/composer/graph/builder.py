"""
Simplified GraphBuilder with Dependency Injection - Focused coordinator using composition.
Uses clean factories and strategies with proper dependency injection pattern.
All agents, storage services, and model profiles are instantiated upfront and injected.
"""

from typing import TYPE_CHECKING
import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START

from models import ModelProfileType, UserConfig, WorkflowType, NodeMetadata
from runner import PipelineFactory

from utils.model_profile import get_model_profile_for_task
from utils.logging import llmmllogger

# Import all agents
from composer.agents.chat_agent import ChatAgent
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
    ToolExecutorNode,
    StaticToolLoadingNode,
)
from composer.nodes.memory import (
    MemorySearchNode,
    MemoryCreationNode,
    MemoryStorageNode,
)
from composer.nodes.agents.chat_node import ChatNode
from composer.nodes.agents import TitleGenerationNode
from composer.nodes.agents.engineering import EngineeringAgentNode
from composer.nodes.summary import ConsolidationNode, SearchSummaryNode
from composer.tools.registry import ToolRegistry

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
            chat_node_metadata = NodeMetadata(
                node_name="PrimaryChatAgent",
                node_id=uuid.uuid4().hex,
                node_type="ChatNode",
                user_id=user_id,
            )
            classifier_node_metadata = NodeMetadata(
                node_name="IntentClassifier",
                node_id=uuid.uuid4().hex,
                node_type="IntentClassifierNode",
                user_id=user_id,
            )
            engineering_node_metadata = NodeMetadata(
                node_name="EngineeringAgent",
                node_id=uuid.uuid4().hex,
                node_type="EngineeringAgentNode",
                user_id=user_id,
            )
            memory_node_metadata = NodeMetadata(
                node_name="MemoryAgent",
                node_id=uuid.uuid4().hex,
                node_type="MemoryAgentNode",
                user_id=user_id,
            )
            embedding_node_metadata = NodeMetadata(
                node_name="EmbeddingAgent",
                node_id=uuid.uuid4().hex,
                node_type="EmbeddingAgentNode",
                user_id=user_id,
            )
            primary_summary_node_metadata = NodeMetadata(
                node_name="PrimarySummaryAgent",
                node_id=uuid.uuid4().hex,
                node_type="PrimarySummaryAgentNode",
                user_id=user_id,
            )
            master_summary_node_metadata = NodeMetadata(
                node_name="MasterSummaryAgent",
                node_id=uuid.uuid4().hex,
                node_type="MasterSummaryAgentNode",
                user_id=user_id,
            )

            # Create agents with injected dependencies
            chat_agent = ChatAgent(
                self.pipeline_factory,
                profile=primary_profile,
                node_metadata=chat_node_metadata,
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
            chat_node = ChatNode(
                self.pipeline_factory,
                chat_agent,
                tool_registry,
            )
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
            tool_executor_node = ToolExecutorNode(tool_registry)
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
            workflow.add_node("tool_executor", tool_executor_node)

            workflow.add_node("chat_summary", chat_summary_node)
            workflow.add_node("search_summary", search_summary_node)

            # Primary chat agent with streaming enabled
            workflow.add_node("chat_agent", chat_node)

            # Build a logical workflow graph structure:
            # 1. Start -> Static tool loading (loads static tools + previous dynamic tools)
            workflow.add_edge(START, "static_tool_loading")
            workflow.add_edge(START, "memory_search")

            # 2. Static tool loading -> Intent Analysis (classifier can now see available tools)
            workflow.add_edge("static_tool_loading", "intent_analysis")

            # 3. Intent Analysis -> Tool collection (filters static tools + creates dynamic tools)
            workflow.add_edge("intent_analysis", "tool_collection")
            workflow.add_edge("tool_collection", "tool_composer")

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

                # Check if we have any tool results in recent messages (indicating we came back from tool execution)
                has_recent_tool_results = False
                for msg in state.messages[-5:]:  # Check last 5 messages for tool results
                    if hasattr(msg, "type") and msg.type == "tool":
                        has_recent_tool_results = True
                        break

                # If we have tool results but no new tool calls, proceed to memory creation
                # This handles the case where the agent is done with tool iterations
                if has_recent_tool_results:
                    return "memory_creation"

                # Otherwise, proceed to chat summary (initial flow)
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

            # 8. Agent pattern: tool_executor routes back to chat_agent or search processing
            def should_continue_agent_loop(state: WorkflowState):
                """
                Route after tool execution - either back to chat_agent for more iterations
                or to search_summary if web search results need processing.
                """
                # Check if web search was performed and needs summarization
                if state.web_search_results:
                    return "search_summary"
                
                # Otherwise, always return to chat_agent for tool result processing
                # This enables the LangGraph agent pattern for iterative tool calling
                return "chat_agent"

            workflow.add_conditional_edges(
                "tool_executor",  
                should_continue_agent_loop,
                {
                    "chat_agent": "chat_agent",    # Cycle back to LLM for more tool calls
                    "search_summary": "search_summary",  # Process web search results
                },
            )

            # 8b. Search summary -> Memory creation (skip second chat_agent call)
            workflow.add_edge("search_summary", "memory_creation")

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
