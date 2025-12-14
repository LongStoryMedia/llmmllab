"""
Simplified GraphBuilder with Dependency Injection - Focused coordinator using composition.
Uses clean factories and strategies with proper dependency injection pattern.
All agents, storage services, and model profiles are instantiated upfront and injected.
"""

from typing import TYPE_CHECKING, Optional, Type, cast
import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from pydantic import BaseModel

from composer.constants import (
    AGENT_NODE_NAME,
    MEMORY_CREATE_NODE_NAME,
    MEMORY_SEARCH_NODE_NAME,
    MEMORY_STORE_NODE_NAME,
    TOOL_NODE_NAME,
)
from models import (
    ModelProfileType,
    UserConfig,
    NodeMetadata,
)
from runner import pipeline_factory

from utils.model_profile import get_model_profile_for_task
from utils.logging import llmmllogger

# Import all agents
# from composer.agents.classifier_agent import ClassifierAgent
from composer.agents.chat import ChatAgent
from composer.agents.engineering_agent import EngineeringAgent
from composer.agents.embed import EmbeddingAgent
from composer.graph.nodes.agent import AgentNode
from composer.graph.nodes.memory import (
    MemorySearchNode,
    MemoryCreationNode,
    MemoryStorageNode,
)
from composer.tools.registry import registry_manager
from composer.graph.state import WorkflowState, assemble_context_messages

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


def should_continue_tool_calls(state: WorkflowState) -> str:
    """Determine if the agent should continue making tool calls based on the last message."""
    # Get the last message from state
    if not state.messages:
        return "end"

    last_message = state.messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


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
        user_config: UserConfig,
    ):
        """
        Initialize GraphBuilder with dependency injection.

        Args:
            storage: Storage instance for dependency injection
            pipeline_factory: PipelineFactory
        """
        # Core dependencies
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
        response_format: Optional[Type[BaseModel]] = None,
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

            primary_model = pipeline_factory.get_pipeline(profile=primary_profile)
            embedding_model = pipeline_factory.get_pipeline(profile=embedding_profile)
            # engineering_model = pipeline_factory.get_pipeline(
            #     profile=engineering_profile
            # )

            primary_agent = ChatAgent(
                model=cast(BaseChatModel, primary_model),
                profile=primary_profile,
                component_name="PrimaryChatAgent",
            )
            engineering_agent = EngineeringAgent(
                model=cast(BaseChatModel, primary_model),
                profile=engineering_profile,
                tool_storage=self.dynamic_tool_storage,
            )
            embedding_agent = EmbeddingAgent(
                model=cast(Embeddings, embedding_model),
                profile=embedding_profile,
                component_name="EmbeddingAgent",
            )

            # Create nodes with injected agents and storage
            memory_creation_node = MemoryCreationNode(
                embedding_agent,
                NodeMetadata(
                    node_name="MemoryCreationNode",
                    node_id=uuid.uuid4().hex,
                    node_type=ModelProfileType(embedding_agent.profile.type).name,
                    user_id=user_id,
                ),
            )
            memory_search_node = MemorySearchNode(
                embedding_agent,
                self.memory_storage,
            )
            memory_storage_node = MemoryStorageNode(self.memory_storage)

            # create tool registry
            tool_registry = await registry_manager.get_user_registry(
                user_id, engineering_agent
            )
            tools = tool_registry.get_all_executable_tools()

            tool_node = ToolNode(tools)

            # Create master workflow graph
            workflow = StateGraph(WorkflowState)

            # create nodes with injected dependencies
            chat_node = AgentNode(
                agent=primary_agent,
                tool_registry=tool_registry,
                node_metadata=NodeMetadata(
                    node_name=AGENT_NODE_NAME,
                    node_id=uuid.uuid4().hex,
                    node_type=ModelProfileType(primary_agent.profile.type).name,
                    user_id=user_id,
                ),
                grammar=response_format,
            )

            async def context_node(state: WorkflowState) -> WorkflowState:
                """Execute the context assembly subgraph and return updated state."""
                state.messages = assemble_context_messages(state)
                return state

            workflow.add_node("context_assembly", context_node)

            # Memory nodes with injected agents and storage
            workflow.add_node(MEMORY_SEARCH_NODE_NAME, memory_search_node)
            workflow.add_node(MEMORY_CREATE_NODE_NAME, memory_creation_node)
            workflow.add_node(MEMORY_STORE_NODE_NAME, memory_storage_node)
            workflow.add_node(AGENT_NODE_NAME, chat_node)
            workflow.add_node(TOOL_NODE_NAME, tool_node)
            # Build a simplified workflow graph structure:
            workflow.add_edge(START, MEMORY_SEARCH_NODE_NAME)
            workflow.add_edge(START, "context_assembly")

            workflow.add_edge("context_assembly", AGENT_NODE_NAME)
            workflow.add_edge(MEMORY_SEARCH_NODE_NAME, AGENT_NODE_NAME)
            # create conditional tool call loop
            workflow.add_conditional_edges(
                AGENT_NODE_NAME,
                should_continue_tool_calls,
                {
                    "tools": TOOL_NODE_NAME,
                    "end": MEMORY_CREATE_NODE_NAME,
                },
            )

            workflow.add_edge(AGENT_NODE_NAME, MEMORY_CREATE_NODE_NAME)
            workflow.add_edge(MEMORY_CREATE_NODE_NAME, MEMORY_STORE_NODE_NAME)
            workflow.add_edge(MEMORY_STORE_NODE_NAME, END)

            # TEMPORARILY DISABLED: Checkpointer causes connection issues
            # The PostgreSQL checkpointer creates a connection during compilation
            # but the connection gets closed before workflow execution, causing failures.
            # NOTE: Checkpointer lifecycle management currently disabled
            self.logger.info(
                "ℹ️  Checkpointer temporarily disabled - compiling without persistence"
            )
            return workflow.compile()

            # # Configure checkpointer at compilation time for parent graph
            # # Per LangGraph docs: "you only need to provide the checkpointer when compiling
            # # the parent graph. LangGraph will automatically propagate the checkpointer to child subgraphs"
            # try:
            #     if self.checkpoint_storage.is_initialized():
            #         # Use LangGraph's standard production pattern
            #         async with (
            #             self.checkpoint_storage.create_checkpointer() as checkpointer
            #         ):
            #             self.logger.info(
            #                 "✅ Compiling workflow with checkpointer - will auto-propagate to subgraphs"
            #             )
            #             # Compile with checkpointer - automatically propagates to all subgraphs
            #             compiled_workflow = workflow.compile(checkpointer=checkpointer)
            #             return compiled_workflow
            #     else:
            #         self.logger.info(
            #             "ℹ️  Checkpoint storage not initialized - compiling without persistence"
            #         )
            #         return workflow.compile()

            # except Exception as e:
            #     self.logger.warning(
            #         f"⚠️  Checkpointer setup failed, compiling without persistence: {e}"
            #     )
            #     # Fallback to compilation without checkpointer
            #     return workflow.compile()
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            # Try to create fallback chat workflow
            raise
