"""
IDE GraphBuilder with Dependency Injection.
Supports two tool modes:
  - Proxy mode: client_tools are bound to the LLM via bind_tools() so it generates
    tool_calls that the client executes. No ToolNode in the graph.
  - Server-side mode: server_tools are added with a ToolNode and feedback loop.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, cast

import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from composer.constants import (
    AGENT_NODE_NAME,
    TOOL_NODE_NAME,
)
from models.default_configs import (
    create_default_user_config,
    DEFAULT_MODEL_PROFILE_CONFIG,
    DEFAULT_PARAMETER_OPTIMIZATION_CONFIG,
)
from models.default_model_profiles import (
    DEFAULT_TEXT_TO_TEXT_MODEL,
)
from models import (
    ModelProfileType,
    UserConfig,
    NodeMetadata,
    MessageRole,
    Message,
    MessageContent,
    MessageContentType,
    ModelProfile,
    ModelParameters,
)
from runner import pipeline_factory

from utils.logging import llmmllogger

from composer.agents.chat import ChatAgent
from composer.graph.workflows.base import GraphBuilder, should_continue_tool_calls
from composer.graph.nodes.agent import AgentNode
from composer.graph.state import WorkflowState

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


IDE_PRIMARY_SYSTEM_PROMPT = """You are a helpful AI coding assistant.

RULES:
- Be concise and direct. Do not repeat yourself.
- Do not wrap your response in thinking tags or reasoning blocks.
- Never output <think> or </think> tags.
- Do not narrate what you are about to do. Just do it.
- If you have tools available, use them via structured tool_calls when needed.
- When you need information you don't have, use the appropriate tool.
- Respond with your final answer directly.

TOOL CALLING:
- When tools are bound, call them using the structured tool_call format.
- Do NOT emit tool calls as XML, JSON, or markdown in your text response.
- You may call multiple tools in a single response.
- After receiving tool results, incorporate them into your response.
- If a tool call fails, try again with corrected arguments.
- Only use tools that are available to you."""


IDE_PRIMARY_PROFILE = ModelProfile(
    id=DEFAULT_MODEL_PROFILE_CONFIG.primary_profile_id,
    user_id="system",
    name="Primary (Default)",
    type=ModelProfileType.Primary.value,
    description="Primary model profile for general chat and reasoning.",
    model_name="glm-4.7-flash",
    parameters=ModelParameters(
        num_ctx=131072,
        repeat_last_n=-1,
        repeat_penalty=0,
        temperature=0.7,
        seed=-1,
        num_predict=-1,
        top_k=20,
        top_p=0.95,
        min_p=0.01,
        max_tokens=-1,
        n_parts=-1,
        batch_size=8192,
        micro_batch_size=1024,
        n_gpu_layers=-1,
        stop=[],
        think=False,
    ),
    system_prompt=IDE_PRIMARY_SYSTEM_PROMPT,
    parameter_optimization=DEFAULT_PARAMETER_OPTIMIZATION_CONFIG,
    created_at=None,
    updated_at=None,
)


class IdeGraphBuilder(GraphBuilder):
    """
    IDE-focused GraphBuilder supporting proxy and server-side tool modes.

    Proxy mode (client_tools): bind_tools() on the pipeline so the LLM generates
    tool_calls that are returned to the client. Graph: START -> Agent -> END.

    Server-side mode (server_tools): adds ToolNode + feedback loop.
    Graph: START -> Agent -> (tools? -> ToolNode -> Agent) | END.
    """

    def __init__(
        self,
        storage: "Storage",
        user_config: UserConfig,
    ):
        self.user_config = user_config
        self.logger = llmmllogger.logger.bind(component="IdeGraphBuilder")

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
        client_tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
        server_tools: Optional[List[BaseTool]] = None,
        tool_choice: Optional[str] = None,
    ) -> CompiledStateGraph:
        """
        Build IDE workflow with optional tool support.

        Args:
            user_id: User identifier
            response_format: Optional response format constraint
            client_tools: Tools for proxy mode.  Accepts OpenAI-format dicts
                (passed straight through to bind_tools, no lossy conversion)
                or LangChain BaseTool instances.
            server_tools: Tools for server-side execution (adds ToolNode + loop)
            tool_choice: Optional tool_choice parameter for bind_tools

        Returns:
            Compiled workflow ready for execution
        """
        try:
            primary_pipeline = pipeline_factory.get_pipeline(
                profile=IDE_PRIMARY_PROFILE
            )
            # Keep a strong reference to the original pipeline throughout build_workflow
            # so GC cannot collect it when bind_tools returns a RunnableBinding wrapper
            primary_model = primary_pipeline

            # Bind client tools to the pipeline so the LLM can generate tool_calls
            if client_tools:
                bind_kwargs: dict = {}
                if tool_choice:
                    bind_kwargs["tool_choice"] = tool_choice
                primary_model = primary_model.bind_tools(client_tools, **bind_kwargs)  # type: ignore[union-attr]

            primary_agent = ChatAgent(
                model=cast(BaseChatModel, primary_model),
                profile=IDE_PRIMARY_PROFILE,
                component_name="PrimaryCodingAgent",
            )

            workflow = StateGraph(WorkflowState)

            chat_node = AgentNode(
                agent=primary_agent,
                node_metadata=NodeMetadata(
                    node_name=AGENT_NODE_NAME,
                    node_id=uuid.uuid4().hex,
                    node_type=ModelProfileType(primary_agent.profile.type).name,
                    user_id=user_id,
                ),
                grammar=response_format,
            )

            workflow.add_node(AGENT_NODE_NAME, chat_node)
            workflow.add_edge(START, AGENT_NODE_NAME)

            if server_tools:
                # Server-side tool execution mode: Agent -> ToolNode -> Agent loop
                tool_node = ToolNode(server_tools)
                workflow.add_node(TOOL_NODE_NAME, tool_node)
                workflow.add_conditional_edges(
                    AGENT_NODE_NAME,
                    should_continue_tool_calls,
                    {
                        "tools": TOOL_NODE_NAME,
                        "end": END,
                    },
                )
                workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            else:
                # Proxy mode or no tools: Agent -> END
                workflow.add_edge(AGENT_NODE_NAME, END)

            return workflow.compile()
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            raise

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        messages: Optional[List[Message]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""
        assert messages is not None, "Messages must be provided to create initial state"
        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            Message(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        state = WorkflowState(
            messages=messages,
            current_user_message=current_user_message,
            user_id=user_id,
            user_config=create_default_user_config(user_id),
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],
        )

        return state
