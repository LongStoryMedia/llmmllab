"""
IDE GraphBuilder with Dependency Injection.
Supports two tool modes:
  - Proxy mode: client_tools are bound to the LLM via bind_tools() so it generates
    tool_calls that the client executes. No ToolNode in the graph.
  - Server-side mode: server_tools are added with a ToolNode and feedback loop.
"""

from datetime import datetime
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

from composer.models import (
    ModelProfileType,
    UserConfig as UserConfigModel,
    NodeMetadata,
    MessageRole,
    Message as MessageModel,
    MessageContent,
    MessageContentType,
    ModelProfile as ModelProfileModel,
    ModelParameters as ComposerModelParameters,
    GPUConfig as ComposerGPUConfig,
    ParameterOptimizationConfig as ComposerParameterOptimizationConfig,
    PerformanceParameter,
    ParameterTuningStrategy,
    CrashPrevention,
    ToolConfig,
    WorkflowConfig,
    CircuitBreakerConfig,
)
from runner.pipeline_factory import pipeline_factory
from runner.models import ModelProfile as RunnerModelProfile, ModelParameters as RunnerModelParameters, GPUConfig as RunnerGPUConfig, ParameterOptimizationConfig as RunnerParameterOptimizationConfig

from composer.utils.model_profile import get_model_profile_for_task
from composer.utils.logging import llmmllogger

from composer.agents.chat import ChatAgent
from composer.graph.workflows.base import GraphBuilder, should_continue_tool_calls
from composer.graph.nodes.agent import AgentNode
from composer.graph.state import WorkflowState

if TYPE_CHECKING:
    from composer.server.interface import ServerInterface


# Type alias for the server's UserConfig (at runtime, it's the composer type)
UserConfig = UserConfigModel  # Use composer's UserConfigModel as the runtime type


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


# Default parameter optimization configuration (disabled by default)
IDE_PARAMETER_OPTIMIZATION_CONFIG = ComposerParameterOptimizationConfig(
    enabled=False,
    parameters=[
        PerformanceParameter(
            parameter_name="n_ctx",
            priority=1,
            tuning_strategy=ParameterTuningStrategy.BINARY_SEARCH,
            max_search_attempts=15,
            floor=65536,  # Start with current profile setting and push higher
            operator="*",
            modifier=2,  # More aggressive scaling
            max_value=262144,  # Push to model's trained context limit
        ),
        PerformanceParameter(
            parameter_name="n_gpu_layers",
            priority=2,
            tuning_strategy=ParameterTuningStrategy.BINARY_SEARCH,
            max_search_attempts=10,
            floor=1,  # Start low and find the maximum that works
            operator="+",
            modifier=10,  # Smaller increments for precise optimization
            max_value=999,  # Very high limit (effectively unlimited GPU layers)
        ),
        PerformanceParameter(
            parameter_name="n_batch",
            priority=3,
            tuning_strategy=ParameterTuningStrategy.BINARY_SEARCH,
            max_search_attempts=15,
            floor=128,  # Start with profile setting and push higher
            operator="*",
            modifier=2,  # More aggressive scaling for throughput
            max_value=16384,  # Allow much larger batches for high-memory systems
        ),
        PerformanceParameter(
            parameter_name="n_ubatch",
            priority=4,
            tuning_strategy=ParameterTuningStrategy.BINARY_SEARCH,
            max_search_attempts=15,
            floor=128,  # Start with profile setting and push higher
            operator="*",
            modifier=2,  # More aggressive scaling
            max_value=16384,  # Allow much larger ubatch for throughput
        ),
    ],
    crash_prevention=CrashPrevention(
        enable_preallocation_test=False,
        memory_buffer_mb=4096,
        timeout_seconds=120,
        enable_graceful_degradation=False,
    ),
)

# Default GPU configuration
IDE_GPU_CONFIG = ComposerGPUConfig(
    no_kv_offload=False,
    gpu_layers=-1,  # Use all GPU layers by default
    main_gpu=0,
    main_gpu_device_id=None,
    # tensor_split=[0.5, 0.25, 0.25],
    tensor_split_devices=None,
    split_mode="row",
    offload_kqv=True,
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


# Default circuit breaker configuration
IDE_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    base_timeout=60.0,
    deep_research_timeout=120.0,
    max_retries=2,
    cooldown_period=30.0,
    enable_perplexity_guard=False,  # Disabled by default to prevent cutting off web search formatting
    perplexity_window=40,
    perplexity_threshold=10.0,
    avg_logprob_floor=-6.0,
    enable_repetition_detection=False,  # Disabled by default to reduce false positives
    repetition_ngram=6,
    repetition_threshold=6,
    min_tokens_for_eval=20,
    perplexity_log_interval_tokens=20,
    log_repetition_events=True,
    tool_gen_repetition_ngram=4,
    tool_gen_repetition_threshold=3,
)

IDE_PRIMARY_PROFILE = RunnerModelProfile(
    id=uuid.UUID("10000000-2000-3000-4000-500000000000"),
    user_id="system",
    name="Primary (Default)",
    type=ModelProfileType.Primary.value,
    description="Primary model profile for general chat and reasoning.",
    model_name="qwen3-coder-next-iq4-xs",
    parameters=RunnerModelParameters(
        # Context window size - max tokens the model can process at once
        num_ctx=100000,
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
        min_p=0.01,
        # Fallback max tokens limit
        max_tokens=-1,
        # Tensor parallel parts (-1 = auto)
        n_parts=-1,
        # Prompt processing batch size - process multiple prompts in parallel
        batch_size=4096,
        # Generation batch size - tokens per decode step per GPU (-1 = auto)
        micro_batch_size=1024,
        # Number of layers to keep on GPU (-1 = all layers on GPU)
        n_gpu_layers=-1,
        # Stop generation sequences
        stop=[],
        # Enable reasoning/thinking mode
        think=False,
        # Keep KV cache on GPU (True = highest speed, False = saves VRAM but slower)
        kv_on_cpu=True,
        # n_cpu_moe=10,
    ),
    system_prompt=IDE_PRIMARY_SYSTEM_PROMPT,
    parameter_optimization=RunnerParameterOptimizationConfig.model_validate(IDE_PARAMETER_OPTIMIZATION_CONFIG.model_dump()),  # type: ignore
    created_at=None,
    updated_at=None,
    gpu_config=RunnerGPUConfig.model_validate(IDE_GPU_CONFIG.model_dump()),  # type: ignore
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
        server: Optional["ServerInterface"],
        user_config: UserConfig,
        server_interface: Optional["ServerInterface"] = None,
    ):
        self.user_config = user_config
        self.logger = llmmllogger.logger.bind(component="IdeGraphBuilder")
        self.server_interface = server_interface

        # Use server_interface if provided, otherwise use server for backward compatibility
        if server_interface is not None:
            self.user_config_service = server_interface.user_config
            self.conversation_service = server_interface.conversation
            self.message_service = server_interface.message
            self.model_profile_service = server_interface.model_profile
            self.memory_service = server_interface.memory
            self.summary_service = server_interface.summary
            self.search_service = server_interface.search
            self.dynamic_tool_service = server_interface.dynamic_tool
            self.checkpoint_service = server_interface.checkpoint
        elif server is not None:
            # Backward compatibility: access services from server object
            self.user_config_service = server.user_config
            self.conversation_service = server.conversation
            self.message_service = server.message
            self.model_profile_service = server.model_profile
            self.memory_service = server.memory
            self.summary_service = server.summary
            self.search_service = server.search
            self.dynamic_tool_service = server.dynamic_tool
            self.checkpoint_service = server.checkpoint
        else:
            self.user_config_service = None  # type: ignore
            self.conversation_service = None  # type: ignore
            self.message_service = None  # type: ignore
            self.model_profile_service = None  # type: ignore
            self.memory_service = None  # type: ignore
            self.summary_service = None  # type: ignore
            self.search_service = None  # type: ignore
            self.dynamic_tool_service = None  # type: ignore
            self.checkpoint_service = None  # type: ignore

    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        client_tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
        server_tools: Optional[List[BaseTool]] = None,
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
            tool_choice: Optional tool_choice parameter for bind_tools

        Returns:
            Compiled workflow ready for execution
        """
        try:
            # Get primary profile using server interface
            primary_profile = await get_model_profile_for_task(
                cast("ServerInterface", self.server_interface),
                self.user_config.model_profiles,
                ModelProfileType.Primary,
                self.user_config.user_id,
            )
            primary_pipeline = pipeline_factory.get_pipeline(profile=primary_profile)  # type: ignore

            self.logger.debug(
                "Building workflow",
                user_id=user_id,
                model=primary_profile.model_name,
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
                profile=primary_profile,
                component_name="PrimaryCodingAgent",
            )

            workflow = StateGraph(WorkflowState)

            chat_node = AgentNode(
                agent=primary_agent,
                node_metadata=NodeMetadata(
                    node_name=AGENT_NODE_NAME,
                    node_id=uuid.uuid4().hex,
                    node_type=ModelProfileType(primary_agent.profile.type).name,
                    execution_time=datetime.now(),
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
        messages: Optional[List[MessageModel]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""
        assert messages is not None, "Messages must be provided to create initial state"
        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            MessageModel(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        # Create UserConfig using composer.models.UserConfig
        user_config = UserConfigModel(
            user_id=user_id,
            summarization=self.user_config.summarization,
            memory=self.user_config.memory,
            model_profiles=self.user_config.model_profiles,
            image_generation=self.user_config.image_generation,
            circuit_breaker=IDE_CIRCUIT_BREAKER_CONFIG,
            gpu_config=IDE_GPU_CONFIG,
            parameter_optimization=IDE_PARAMETER_OPTIMIZATION_CONFIG,
            workflow=IDE_WORKFLOW_CONFIG,
            tool=IDE_TOOL_CONFIG,
            preferences=self.user_config.preferences,
        )

        state = WorkflowState(
            messages=messages,  # type: ignore
            current_user_message=current_user_message,  # type: ignore
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],  # type: ignore
        )

        return state
