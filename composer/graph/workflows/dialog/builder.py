"""
Simplified GraphBuilder with Dependency Injection - Focused coordinator using composition.
Uses clean factories and strategies with proper dependency injection pattern.
All agents, storage services, and model profiles are instantiated upfront and injected.
"""

from typing import TYPE_CHECKING, Optional, Type, cast, Sequence, Any
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
    TITLE_GENERATION_NODE_NAME,
    TOOL_NODE_NAME,
)
from composer.models import (
    ModelProfileType,
    UserConfig as UserConfigModel,
    NodeMetadata,
    MessageRole,
    Message as MessageModel,
    MessageContent,
    MessageContentType,
    Conversation as ConversationModel,
)
from composer.utils.model_profile import get_model_profile_for_task

# Runner imports for pipeline_factory - use runner's ModelProfile for pipeline
from runner.models import ModelProfile
from runner.pipeline_factory import pipeline_factory

if TYPE_CHECKING:
    from composer.server.interface import ServerInterface
    from composer.server.conversation import Conversation as ConversationService
    from composer.server.summary import Summary as SummaryService
from composer.utils.logging import llmmllogger

from composer.agents.chat import ChatAgent
from composer.agents.engineering_agent import EngineeringAgent
from composer.agents.embed import EmbeddingAgent
from composer.graph.workflows.base import (
    GraphBuilder,
    should_continue_tool_calls,
    should_generate_title,
)
from composer.graph.nodes.agent import AgentNode
from composer.graph.nodes.memory import (
    MemorySearchNode,
    MemoryCreationNode,
    MemoryStorageNode,
)
from composer.tools.registry import registry_manager
from composer.graph.state import WorkflowState, assemble_context_messages

# Server types for TYPE_CHECKING (to avoid circular imports at runtime)
if TYPE_CHECKING:
    from composer.server.conversation import Conversation as ConversationService
    from composer.server.message import Message as MessageService
    from composer.server.userconfig import UserConfig as UserConfigService
    from composer.server.memory import Memory as MemoryService
    from composer.server.summary import Summary as SummaryService
    from composer.server.dynamic_tool import DynamicToolStorage as DynamicToolService
else:
    # Runtime type alias for user_config parameter - use the protocol
    from composer.server.interface import UserConfigService


class DialogGraphBuilder(GraphBuilder):
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
        server: "ServerInterface",
        user_config: UserConfigService,
    ):
        """
        Initialize GraphBuilder with dependency injection.

        Args:
            server: Server instance for database access
            user_config: UserConfig object (passed from server layer)
        """
        # Core dependencies
        self.user_config = user_config
        self.logger = llmmllogger.logger.bind(component="GraphBuilder")

        # Use server services for database access
        self.user_config_service = server.user_config
        self.conversation_service = server.conversation
        self.message_service = server.message
        self.model_profile_service = server.model_profile
        self.memory_service = server.memory
        self.summary_service = server.summary
        self.dynamic_tool_service = server.dynamic_tool

    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        server: Optional["ServerInterface"] = None,
        **kwargs,
    ) -> CompiledStateGraph:
        """
        Build a workflow of the specified type.

        Simple delegation to workflow factory with error handling.

        Args:
            workflow_type: Type of workflow to build
            user_id: User identifier
            server: Server interface for model profile retrieval
            use_cache: Whether to use caching
            **kwargs: Additional workflow parameters

        Returns:
            Compiled workflow ready for execution
        """
        try:
            # Get server interface (required for model profile retrieval)
            if server is None:
                raise ValueError("Server interface is required for build_workflow")

            # Get user config model from service (self.user_config is the service, not the model)
            user_config_model = await self.user_config_service.get_user_config(user_id)

            primary_profile = await get_model_profile_for_task(
                server,
                user_config_model.model_profiles,
                ModelProfileType.Primary,
                user_id,
            )
            engineering_profile = await get_model_profile_for_task(
                server,
                user_config_model.model_profiles,
                ModelProfileType.Engineering,
                user_id,
            )
            embedding_profile = await get_model_profile_for_task(
                server,
                user_config_model.model_profiles,
                ModelProfileType.Embedding,
                user_id,
            )

            # Use runner's ModelProfile for pipeline_factory - cast to avoid type mismatch
            primary_model = pipeline_factory.get_pipeline(profile=primary_profile)  # type: ignore
            embedding_model = pipeline_factory.get_pipeline(profile=embedding_profile)  # type: ignore

            primary_agent = ChatAgent(
                model=cast(BaseChatModel, primary_model),
                profile=primary_profile,
                component_name="PrimaryChatAgent",
            )
            engineering_agent = EngineeringAgent(
                model=cast(BaseChatModel, primary_model),
                profile=engineering_profile,
                tool_service=cast(Any, self.dynamic_tool_service),  # type: ignore
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
                cast(Any, self.memory_service),  # type: ignore
            )
            memory_storage_node = MemoryStorageNode(cast(Any, self.memory_service))  # type: ignore

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

            async def title_generation_node(state: WorkflowState) -> WorkflowState:
                """Generate and update conversation title if needed."""
                try:
                    # Check if we need to generate a title
                    if state.title and not state.title.startswith("New conversation"):
                        self.logger.debug(
                            f"Skipping title generation - conversation already has title: {state.title}"
                        )
                        return state

                    # Need at least 2 messages (user + assistant) for meaningful title
                    if not state.messages or len(state.messages) < 2:
                        self.logger.debug("Not enough messages for title generation")
                        return state

                    # Generate title using primary agent
                    self.logger.info(
                        f"Generating title for conversation {state.conversation_id}"
                    )
                    title = await primary_agent.generate_title(state.messages)

                    if title and title != "New Conversation":
                        # Update the state
                        state.title = title

                        # Persist to database
                        await self.conversation_service.update_conversation_title(
                            title=title,
                            conversation_id=state.conversation_id,
                            user_id=state.user_id,
                        )
                        self.logger.info(
                            f"✓ Generated and saved title for conversation {state.conversation_id}: {title}"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to generate valid title for conversation {state.conversation_id}"
                        )

                except Exception as e:
                    # Don't fail the workflow if title generation fails
                    self.logger.error(
                        f"Error generating title for conversation {state.conversation_id}: {e}",
                        exc_info=True,
                    )

                return state

            workflow.add_node("context_assembly", context_node)
            workflow.add_node(TITLE_GENERATION_NODE_NAME, title_generation_node)

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
            # Tool results flow back to agent for further processing
            workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            workflow.add_edge(MEMORY_CREATE_NODE_NAME, MEMORY_STORE_NODE_NAME)

            # Add conditional title generation after memory storage
            workflow.add_conditional_edges(
                MEMORY_STORE_NODE_NAME,
                should_generate_title,
                {
                    "generate_title": TITLE_GENERATION_NODE_NAME,
                    "skip_title": END,
                },
            )
            workflow.add_edge(TITLE_GENERATION_NODE_NAME, END)

            return workflow.compile()
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            # Try to create fallback chat workflow
            raise

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        user_config: Optional[UserConfigModel] = None,
        messages: Optional[Sequence[MessageModel]] = None,
        conversation: Optional[ConversationModel] = None,
        summaries: Optional[Sequence["SummaryService"]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from messages.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            user_config: UserConfig object (optional, retrieved from db if not provided)
            messages: List of Message objects (optional, retrieved from db if not provided)
            conversation: Conversation object (optional, retrieved from db if not provided)
            summaries: List of Summary objects (optional, retrieved from db if not provided)
        """

        # Get data from db if not provided
        if (
            user_config is None
            or messages is None
            or conversation is None
            or summaries is None
        ):
            if user_config is None:
                user_config = await self.user_config_service.get_user_config(user_id)
            if messages is None:
                # Get messages as server types (we'll use them directly - they have same structure)
                messages = await self.message_service.get_conversation_history(
                    conversation_id
                )
            if conversation is None:
                conversation = await self.conversation_service.get_conversation(
                    conversation_id, user_id
                )
            if summaries is None:
                summaries = await self.summary_service.get_summaries_for_conversation(
                    conversation_id
                )  # type: ignore

        # WorkflowState expects Message objects, not BaseMessage objects
        # So we use the messages directly without LangChain conversion

        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            MessageModel(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        # Create the state with centralized user configuration and todo context
        state = WorkflowState(
            title=(
                conversation.title
                if (
                    conversation
                    and not conversation.title.startswith("New conversation")
                )
                else None
            ),
            messages=messages,  # type: ignore  # Use Message objects directly
            summaries=summaries,  # type: ignore
            current_user_message=current_user_message,  # Use Message object directly
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],  # type: ignore
        )

        return state
