"""
Simplified GraphBuilder with Dependency Injection - Focused coordinator using composition.
Uses clean factories and strategies with proper dependency injection pattern.
All agents, storage services, and model profiles are instantiated upfront and injected.
"""

from typing import TYPE_CHECKING, Optional

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START

from models import ModelProfileType
from runner import PipelineFactory

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

# Import all agents
from composer.agents.intent_classifier import IntentClassifierAgent
from composer.agents.engineering_agent import EngineeringAgent
from composer.agents.memory_agent import MemoryAgent
from composer.agents.embedding_agent import EmbeddingAgent
from composer.agents.summarization_agent import SummarizationAgent
from composer.agents.single_source_agent import SingleSourceAgent

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

from composer.tools.registry import ToolRegistry
from composer.monitoring.logging import composer_logger

from .state import WorkflowState


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

    def __init__(self, storage: 'Storage', pipeline_factory: Optional[PipelineFactory] = None):
        """
        Initialize GraphBuilder with dependency injection.
        
        Args:
            storage: Storage instance for dependency injection
            pipeline_factory: Pipeline factory (optional, will import if None)
        """
        # Import pipeline_factory if not provided to maintain backward compatibility
        if pipeline_factory is None:
            from runner import pipeline_factory as default_pipeline_factory  # pylint: disable=import-outside-toplevel
            pipeline_factory = default_pipeline_factory
        
        # Core dependencies
        self.pipeline_factory = pipeline_factory
        self.storage = storage
        self.logger = composer_logger.logger.bind(component="GraphBuilder")
        
        # Create all agents upfront for dependency injection
        self._create_agents()

    def _create_agents(self):
        """Create all agent instances for dependency injection."""
        # Get storage services needed by agents
        self._create_storage_services()
        
        # Create agents with injected dependencies
        self.intent_classifier_agent = IntentClassifierAgent(
            user_config_storage=self.user_config_storage
        )
        self.engineering_agent = EngineeringAgent(
            pipeline_factory=self.pipeline_factory,
            user_config_storage=self.user_config_storage
        )
        self.memory_agent = MemoryAgent(
            memory_storage=self.memory_storage
        )
        self.embedding_agent = EmbeddingAgent(
            pipeline_factory=self.pipeline_factory,
            user_config_storage=self.user_config_storage
        )
        self.summarization_agent = SummarizationAgent(
            pipeline_factory=self.pipeline_factory,
            summary_storage=self.summary_storage,
            search_storage=self.search_storage,
            user_config_storage=self.user_config_storage
        )
        self.single_source_agent = SingleSourceAgent()
        
        # Create tool registry (also depends on embedding agent)
        self.tool_registry = ToolRegistry(self.pipeline_factory, self.user_config_storage)

    def _create_storage_services(self) -> None:
        """Create storage service instances for dependency injection."""
        # Extract specific storage services that agents and nodes need
        # Use storage.get_service for type safety when storage is initialized, 
        # otherwise fall back to direct access for test environments
        try:
            # Use storage.get_service for type safety and linter warnings avoidance
            self.user_config_storage: 'UserConfigStorage' = self.storage.get_service(self.storage.user_config)
            self.conversation_storage: 'ConversationStorage' = self.storage.get_service(self.storage.conversation)
            self.message_storage: 'MessageStorage' = self.storage.get_service(self.storage.message)
            self.model_profile_storage: 'ModelProfileStorage' = self.storage.get_service(self.storage.model_profile)
            self.memory_storage: 'MemoryStorage' = self.storage.get_service(self.storage.memory)
            self.summary_storage: 'SummaryStorage' = self.storage.get_service(self.storage.summary)
            self.search_storage: 'SearchStorage' = self.storage.get_service(self.storage.search)
            self.dynamic_tool_storage: 'DynamicToolStorage' = self.storage.get_service(self.storage.dynamic_tool)
        except ValueError as e:
            if "Storage not initialized" in str(e):
                # Fallback for test environments where storage may not be initialized
                # Check that storage services are available, otherwise raise an exception
                if self.storage.user_config is None:
                    raise ValueError("UserConfigStorage is required but not available")
                if self.storage.conversation is None:
                    raise ValueError("ConversationStorage is required but not available") 
                if self.storage.message is None:
                    raise ValueError("MessageStorage is required but not available")
                if self.storage.model_profile is None:
                    raise ValueError("ModelProfileStorage is required but not available")
                if self.storage.memory is None:
                    raise ValueError("MemoryStorage is required but not available")
                if self.storage.summary is None:
                    raise ValueError("SummaryStorage is required but not available")
                if self.storage.search is None:
                    raise ValueError("SearchStorage is required but not available")
                if self.storage.dynamic_tool is None:
                    raise ValueError("DynamicToolStorage is required but not available")
                    
                self.user_config_storage = self.storage.user_config
                self.conversation_storage = self.storage.conversation
                self.message_storage = self.storage.message
                self.model_profile_storage = self.storage.model_profile
                self.memory_storage = self.storage.memory
                self.summary_storage = self.storage.summary
                self.search_storage = self.storage.search
                self.dynamic_tool_storage = self.storage.dynamic_tool
            else:
                raise

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
            self.logger.info("Building workflow with dependency injection", user_id=user_id)
            # Create master workflow graph
            workflow = StateGraph(WorkflowState)

            # Create nodes with injected dependencies
            # Intent analysis -> router -> (optional specialized agents) pattern
            workflow.add_node(
                "intent_analysis", 
                IntentClassifierNode(intent_classifier_agent=self.intent_classifier_agent)
            )
            workflow.add_node("workflow_router", WorkflowRouter(user_id))
            
            # Engineering agent (invoked only when routing selects engineering)
            workflow.add_node(
                "engineering_agent", 
                EngineeringAgentNode(engineering_agent=self.engineering_agent)
            )

            # Title generation (if no title exists)
            workflow.add_node(
                "title_generation", 
                TitleGenerationNode(self.pipeline_factory, summarization_agent=self.summarization_agent)
            )

            # Memory nodes with injected agents and storage
            workflow.add_node(
                "memory_search", 
                MemorySearchNode(
                    pipeline_factory=self.pipeline_factory,
                    memory_agent=self.memory_agent,
                    embedding_agent=self.embedding_agent,
                    storage=self.storage
                )
            )
            workflow.add_node(
                "memory_creation", 
                MemoryCreationNode(
                    pipeline_factory=self.pipeline_factory,
                    embedding_agent=self.embedding_agent,
                    storage=self.storage
                )
            )
            workflow.add_node(
                "memory_storage", 
                MemoryStorageNode(
                    memory_agent=self.memory_agent,
                    storage=self.storage
                )
            )

            # Tool nodes with injected dependencies
            workflow.add_node(
                "static_tool_collection", 
                StaticToolCollectionNode(self.tool_registry)
            )
            workflow.add_node(
                "dynamic_tool_collection",
                DynamicToolCreationNode(
                    tool_registry=self.tool_registry, 
                    pipeline_factory=self.pipeline_factory,
                    storage=self.storage
                ),
            )
            workflow.add_node("tool_composer", ToolComposerNode())
            workflow.add_node(
                "tool_executor", 
                ToolExecutorNode(self.tool_registry)
            )

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

                # If last message is from assistant and has tool calls, execute tools
                if (
                    hasattr(last_message, "type")
                    and last_message.type == "ai"
                    and hasattr(last_message, "tool_calls")
                    and last_message.tool_calls
                ):
                    return "tool_executor"

                # Otherwise, proceed to memory creation (this includes tool results)
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
