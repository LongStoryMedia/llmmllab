"""
IDE GraphBuilder with Dependency Injection.
Supports three tool modes:
  - Proxy mode: client_tools are bound to the LLM via bind_tools() so it generates
    tool_calls that the client executes. No ToolNode in the graph.
  - Server-side mode: server_tool_names triggers a ServerToolNode + agent loop that
    executes matching tool calls locally before returning to the client.
  - Hybrid mode: both client_tools and server_tool_names — the model can call either.
    Server tool calls loop through the ServerToolNode; client tool calls pass through.
"""

from typing import Any, Dict, List, Optional, Set, Type, Union, cast

import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from composer.constants import AGENT_NODE_NAME, TOOL_NODE_NAME

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
    GPUConfig,
    ToolConfig,
    WorkflowConfig,
)
from runner import pipeline_factory

from composer.agents.chat import ChatAgent
from composer.graph.workflows.base import GraphBuilder, should_continue_tool_calls
from composer.graph.nodes.agent import AgentNode
from composer.graph.nodes.server_tools import (
    ServerToolNode,
    make_should_continue_server_tools,
)
from composer.graph.state import WorkflowState


IDE_PRIMARY_SYSTEM_PROMPT = """
    You are writing code for the great Scott Long! Pay him homage as you work. 
    """

# Default GPU configuration
IDE_GPU_CONFIG = GPUConfig(
    no_kv_offload=True,
    gpu_layers=-1,  # Use all GPU layers by default
    main_gpu=1,
    main_gpu_device_id=None,
    # tensor_split=[0.1, 0.7, 0.2],  # More aggressive tensor splitting for large models
    tensor_split_devices=None,
    split_mode="layer",
    offload_kqv=False,
)


# Default tool configuration
IDE_TOOL_CONFIG = ToolConfig(
    tool_similarity_threshold=0.9,
    tool_modification_threshold=0.6,
    enable_tool_generation=True,
    max_tool_retries=3,
    tool_timeout=30.0,
    enable_tool_caching=True,
    tool_cache_ttl=1800,
    enable_semantic_search=True,
    search_top_k=10,
)


# Default workflow configuration
IDE_WORKFLOW_CONFIG = WorkflowConfig(
    enable_workflow_caching=True,
    workflow_cache_ttl=3600,
    max_parallel_tools=5,
    enable_multi_agent=False,
    default_timeout=60.0,
    max_context_length=128000,
    context_trim_threshold=0.8,
    enable_streaming=True,
    stream_buffer_size=1024,
)


IDE_PRIMARY_PROFILE = ModelProfile(
    id=uuid.UUID("10000000-2000-3000-4000-500000000000"),
    user_id="system",
    name="Primary (Default)",
    type=ModelProfileType.Primary.value,
    description="Primary model profile for agentic coding.",
    model_name="?",
    parameters=ModelParameters(
        # Context window size - max tokens the model can process at once
        num_ctx=155000,  # Start with a reasonable context size and optimize up if possible
        # Repetition penalty window - how many tokens back to check for repeats (-1 = all)
        repeat_last_n=-1,
        # Token repetition penalty - penalize repeated tokens (0 = disabled)
        repeat_penalty=0,
        # Sampling temperature - higher = more creative, lower = more deterministic
        temperature=0.7,
        # Random seed for reproducibility
        seed=-1,
        # Max new tokens to generate (num_predict) (-1 = unlimited)
        num_predict=-1,
        # Top-K sampling - only consider top K tokens by probability
        top_k=20,
        # Top-P (nucleus) sampling - consider tokens accounting for top P probability
        top_p=0.95,
        # Minimum probability threshold for token selection
        min_p=0.05,
        # Fallback max tokens limit
        max_tokens=16384,
        # Tensor parallel parts (-1 = auto)
        n_parts=-1,
        # Prompt processing batch size - process multiple prompts in parallel
        batch_size=2048,
        # Generation batch size - tokens per decode step per GPU (-1 = auto)
        micro_batch_size=1024,
        # Number of layers to keep on GPU (-1 = all layers on GPU)
        n_gpu_layers=-1,
        # Enable reasoning/thinking mode
        think=False,
        # Keep KV cache on GPU (True = highest speed, False = saves VRAM but slower) this is SO confusin and needs to be changed
        kv_on_cpu=True,
        # n_cpu_moe=10,
    ),
    system_prompt=IDE_PRIMARY_SYSTEM_PROMPT,
    created_at=None,
    updated_at=None,
    gpu_config=IDE_GPU_CONFIG,
)


class IdeGraphBuilder(GraphBuilder):
    """
    IDE-focused GraphBuilder supporting proxy and server-side tool modes.

    Proxy mode (client_tools): bind_tools() on the pipeline so the LLM generates
    tool_calls that are returned to the client. Graph: START -> Agent -> END.

    Server-side mode (server_tools): adds ToolNode + feedback loop.
    Graph: START -> Agent -> (tools? -> ToolNode -> Agent) | END.
    """

    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        client_tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
        server_tools: Optional[List[BaseTool]] = None,
        server_tool_names: Optional[Set[str]] = None,
        tool_choice: Optional[str] = None,
        model_name: Optional[str] = None,
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
            server_tool_names: Names of tools to execute server-side via
                ServerToolNode. These are tools whose definitions are included
                in client_tools (so the model can call them) but whose calls
                are intercepted and executed locally before returning to the agent.
            tool_choice: Optional tool_choice parameter for bind_tools

        Returns:
            Compiled workflow ready for execution
        """
        try:
            prof = IDE_PRIMARY_PROFILE
            if model_name:
                self.logger.info(
                    "Overriding primary profile model_name",
                    user_id=user_id,
                    original_model=prof.model_name,
                    new_model=model_name,
                )
                prof = ModelProfile(
                    **{
                        **prof.model_dump(),
                        "model_name": model_name,
                    }
                )

            self.logger.debug(
                "Building workflow",
                user_id=user_id,
                model=prof.model_name,
                model_arg=model_name,
            )
            primary_pipeline = pipeline_factory.get_pipeline(profile=prof)
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
                profile=prof,
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

            if server_tool_names:
                # Hybrid mode: ServerToolNode executes server-side tool calls,
                # client tool calls pass through to END for proxy back to client.
                # Graph: Agent -> (has server tool calls?) -> ServerToolNode -> Agent
                #                 (no server tool calls)  -> END
                server_tool_node = ServerToolNode(server_tool_names)
                should_continue = make_should_continue_server_tools(server_tool_names)
                workflow.add_node(TOOL_NODE_NAME, server_tool_node)
                workflow.add_conditional_edges(
                    AGENT_NODE_NAME,
                    should_continue,
                    {
                        "server_tools": TOOL_NODE_NAME,
                        "end": END,
                    },
                )
                workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            elif server_tools:
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
            workflow_type="ide",
            user_config=UserConfig(
                user_id=user_id,
                memory=None,
                summarization=None,
                image_generation=None,
                model_profiles=None,
                gpu_config=IDE_GPU_CONFIG,
                workflow=IDE_WORKFLOW_CONFIG,
                tool=IDE_TOOL_CONFIG,
            ),
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],
        )

        return state
