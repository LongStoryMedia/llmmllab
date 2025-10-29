"""
BaseLlamaCppPipeline as a custom BaseChatModel implementation.
Uses llama-cpp-python directly instead of LangChain's ChatLlamaCpp wrapper.
"""

import json
import multiprocessing
import os
import re
import time

from typing import Optional, List, Any, Dict, Iterator, Type, Tuple, cast

from pydantic import BaseModel
import llama_cpp
from llama_cpp import ChatCompletionResponseMessage, llama_grammar
from llama_cpp.llama_types import CreateChatCompletionResponse

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall as LangChainToolCall,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools.base import BaseTool

from models import Model, ModelProfile, OptimalParameters
from models.default_configs import DEFAULT_GPU_CONFIG
from utils.logging import llmmllogger
from runner.utils.hardware_manager import hardware_manager
from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery


class BaseLlamaCppPipeline(BaseChatModel):
    """
    Custom BaseChatModel implementation using llama-cpp-python directly.

    Features:
    - Direct Llama class instantiation from llama-cpp-python
    - Hardware optimization with GPU layers and context fallback
    - Grammar constraints support (GBNF/Pydantic)
    - Tool calling support through prompt formatting
    - Streaming and non-streaming chat completion
    """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "allow"

    model: Model
    profile: ModelProfile
    grammar: Optional[Type[BaseModel]]

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]],
        **kwargs,
    ):
        # Pass the required fields to the parent constructor for Pydantic validation
        super().__init__(model=model, profile=profile, grammar=grammar, **kwargs)  # type: ignore
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )
        self.grammar = grammar
        self._bound_tools: List[BaseTool] = kwargs.get("_bound_tools", [])
        self.hardware_manager = hardware_manager

        # Initialize intelligent OOM recovery system (can be disabled)
        self.use_intelligent_oom = (
            os.getenv("ENABLE_INTELLIGENT_OOM_RECOVERY", "false").lower() == "true"
        )
        if self.use_intelligent_oom:
            self.oom_recovery = IntelligentOOMRecovery()
            self.llama_instance = self._initialize_llama_with_intelligent_oom_recovery(
                self._get_gguf_path()
            )
        else:
            # Simple initialization - fail fast on errors
            self.oom_recovery = None
            self.llama_instance = self._initialize_llama_simple(self._get_gguf_path())

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-cpp-custom"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model.name,
            "model_path": self._get_gguf_path(),
            "n_ctx": self.profile.parameters.num_ctx or 4096,
            "temperature": self.profile.parameters.temperature or 0.7,
        }

    def bind_tools(
        self, tools: List[BaseTool], **kwargs: Any
    ) -> "BaseLlamaCppPipeline":
        """
        Bind tools to this model for tool calling support.

        For llama-cpp-python models, tools are handled through the grammar system
        and function calling via prompt formatting.

        Args:
            tools: List of tools to bind to the model
            **kwargs: Additional keyword arguments for tool binding

        Returns:
            A new instance of the model with tools bound
        """
        self._bound_tools = tools
        return self

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        return (
            self.model.details.gguf_file
            if hasattr(self.model.details, "gguf_file") and self.model.details.gguf_file
            else self.model.model
        )

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        try:
            cpu_count = multiprocessing.cpu_count()
            optimal_threads = min(max(cpu_count // 2, 2), 8)
            self._logger.debug(
                f"Using {optimal_threads} threads (CPU count: {cpu_count})"
            )
            return optimal_threads
        except Exception:
            self._logger.warning(
                "Could not determine CPU count, using default threading"
            )
            return 4

    def _clear_pipeline_and_memory(self):
        """Clear pipeline instance and GPU memory."""
        if hasattr(self, "llama_instance") and self.llama_instance:
            try:
                self.llama_instance.close()
            except Exception:
                pass
            del self.llama_instance
            self.llama_instance = None

        # Clear GPU memory
        try:
            self.hardware_manager.clear_memory(aggressive=True)
        except Exception as e:
            self._logger.warning(f"Error clearing GPU memory: {e}")

        # Give GPU a moment to clean up
        time.sleep(1)

    def _initialize_llama_with_intelligent_oom_recovery(
        self, gguf_path: str
    ) -> llama_cpp.Llama:
        """
        Initialize Llama instance using intelligent ML-based OOM recovery.

        Strategy:
        1. Use ML to predict optimal starting parameters based on model size and system resources
        2. On OOM, follow structured recovery: clear memory -> reduce batch -> move to CPU -> reduce context
        3. Learn from successful/failed attempts to improve future predictions
        """
        if llama_cpp.Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        assert self.oom_recovery

        # Get original target parameters
        target_n_ctx = self.profile.parameters.num_ctx or 32768
        target_n_batch = self.profile.parameters.batch_size or 512
        target_n_ubatch = 512

        # GPU layers - default to full offload
        requested_gpu_layers = -1
        if (
            self.profile.gpu_config is not None
            and self.profile.gpu_config.gpu_layers is not None
        ):
            requested_gpu_layers = self.profile.gpu_config.gpu_layers

        # Use ML to predict optimal starting parameters
        predicted_params = self.oom_recovery.predict_optimal_parameters_from_profile(
            model_profile=self.profile,
            model_path=gguf_path,
            hardware_manager=self.hardware_manager,
        )

        # Convert -1 (full offload) to a reasonable default for validation
        # We'll convert it back to -1 when actually initializing llama.cpp
        validated_gpu_layers = requested_gpu_layers if requested_gpu_layers >= 0 else 99

        # Keep track of original parameters for recovery calculations
        original_params = OptimalParameters(
            n_ctx=target_n_ctx,
            n_batch=target_n_batch,
            n_ubatch=target_n_ubatch,
            n_gpu_layers=validated_gpu_layers,
        )

        # Start with ML-predicted parameters
        current_params = predicted_params.model_copy()

        # Other configuration
        perplexity_enabled = bool(
            getattr(self.profile.parameters, "enable_perplexity_guard", False)
        )
        gcfg = self.profile.gpu_config or DEFAULT_GPU_CONFIG

        # OOM recovery loop with intelligent strategy
        max_attempts = 10

        for attempt in range(1, max_attempts + 1):
            try:
                self._logger.info(
                    f"🚀 Initializing {self.model.name} (attempt {attempt}): "
                    f"n_ctx={current_params.n_ctx:,}, n_batch={current_params.n_batch}, "
                    f"n_ubatch={current_params.n_ubatch}, gpu_layers={current_params.n_gpu_layers}"
                )

                start_time = time.time()

                # Convert back to -1 for full offload if originally requested
                actual_gpu_layers = (
                    requested_gpu_layers
                    if requested_gpu_layers == -1
                    else current_params.n_gpu_layers
                )

                llama_instance = llama_cpp.Llama(
                    model_path=gguf_path,
                    n_gpu_layers=actual_gpu_layers,
                    split_mode=llama_cpp.LLAMA_SPLIT_MODE_ROW,
                    tensor_split=gcfg.tensor_split,
                    vocab_only=False,
                    use_mmap=True,
                    use_mlock=False,
                    kv_overrides=None,
                    # Context Params
                    seed=self.profile.parameters.seed or llama_cpp.LLAMA_DEFAULT_SEED,
                    n_ctx=current_params.n_ctx,
                    n_batch=current_params.n_batch,
                    n_ubatch=current_params.n_ubatch,
                    n_threads=self._get_optimal_threads(),
                    temperature=self.profile.parameters.temperature or 0.7,
                    top_p=self.profile.parameters.top_p or 0.8,
                    top_k=self.profile.parameters.top_k or 20,
                    repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
                    f16_kv=True,
                    verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                    flash_attn=getattr(
                        self.profile.parameters, "flash_attention", True
                    ),
                    logits_all=perplexity_enabled,
                    logprobs=1 if perplexity_enabled else 0,
                    embedding=False,
                    chat_format=None,
                    n_threads_batch=None,
                    rope_scaling_type=None,
                    pooling_type=llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
                    rope_freq_base=0.0,
                    rope_freq_scale=0.0,
                    yarn_ext_factor=-1.0,
                    yarn_attn_factor=1.0,
                    yarn_beta_fast=32.0,
                    yarn_beta_slow=1.0,
                    yarn_orig_ctx=0,
                    offload_kqv=False,
                    op_offload=None,
                    swa_full=None,
                    no_perf=False,
                    last_n_tokens_size=64,
                    lora_base=None,
                    lora_scale=1.0,
                    lora_path=None,
                    numa=False,
                    chat_handler=None,
                    draft_model=None,
                    tokenizer=None,
                    type_k=None,
                    type_v=None,
                    smp_infill=False,
                )

                # Success! Record the configuration for ML training
                initialization_time_ms = (time.time() - start_time) * 1000

                # Get GPU memory usage if available
                gpu_memory_used_mb = 0
                try:
                    memory_stats = self.hardware_manager.update_all_memory_stats()
                    if memory_stats:
                        # Find primary GPU stats - look for GPU with id=0 first
                        primary_gpu_stats = None

                        # First, look for GPU with id=0 (cuda:0 equivalent)
                        for stats in memory_stats.values():
                            if (
                                hasattr(stats, "id")
                                and stats.id == 0
                                and hasattr(stats, "mem_used")
                                and hasattr(stats, "name")
                                and "nvidia" in stats.name.lower()
                            ):
                                primary_gpu_stats = stats
                                break

                        # If no GPU with id=0 found, use the first NVIDIA GPU
                        if primary_gpu_stats is None:
                            for stats in memory_stats.values():
                                if (
                                    hasattr(stats, "mem_used")
                                    and hasattr(stats, "name")
                                    and "nvidia" in stats.name.lower()
                                ):
                                    primary_gpu_stats = stats
                                    break

                        # Extract memory usage from primary GPU
                        if primary_gpu_stats and hasattr(primary_gpu_stats, "mem_used"):
                            gpu_memory_used_mb = primary_gpu_stats.mem_used
                except Exception:
                    pass

                # Record success for ML learning (only if OOM recovery is enabled)
                if self.oom_recovery is not None:
                    self.oom_recovery.record_success(
                        model_path=gguf_path,
                        params=current_params,
                        hardware_manager=self.hardware_manager,
                        initialization_time_ms=initialization_time_ms,
                        gpu_memory_used_mb=gpu_memory_used_mb,
                    )

                self._logger.info(
                    f"✅ Successfully initialized {self.model.name} (attempt {attempt}): "
                    f"n_ctx={current_params.n_ctx:,}, n_batch={current_params.n_batch}, "
                    f"n_ubatch={current_params.n_ubatch}, gpu_layers={current_params.n_gpu_layers} "
                    f"in {initialization_time_ms:.1f}ms"
                )

                return llama_instance

            except Exception as e:
                error_str = str(e).lower()
                is_oom = any(
                    oom_indicator in error_str
                    for oom_indicator in [
                        "out of memory",
                        "oom",
                        "cuda error",
                        "memory allocation failed",
                        "insufficient memory",
                        "cudamalloc failed",
                        "failed to create llama_context",
                        "context creation failed",
                        "failed to allocate",
                        "allocation failed",
                        "ggml_cuda_alloc_buffer",
                        "ggml_backend_alloc_ctx_tensors_from_buft",
                    ]
                )

                if not is_oom:
                    # Not an OOM error, re-raise immediately
                    self._logger.error(f"❌ Non-OOM error during initialization: {e}")
                    raise e

                # Handle OOM with intelligent recovery
                self._logger.warning(
                    f"🔥 OOM detected (attempt {attempt}): {error_str}"
                )

                # CRITICAL: Clean up any failed instance
                try:
                    if "llama_instance" in locals():
                        try:
                            llama_instance.close()
                            self._logger.debug("🧹 Closed failed llama_instance")
                        except Exception:
                            pass
                        del llama_instance
                except Exception as cleanup_e:
                    self._logger.warning(
                        f"Error during failed instance cleanup: {cleanup_e}"
                    )

                # Execute intelligent recovery strategy
                recovery_result = self.oom_recovery.execute_recovery_strategy(
                    attempt=attempt,
                    original_params=original_params,
                    current_params=current_params,
                    hardware_manager=self.hardware_manager,
                )
                new_params = recovery_result.parameters
                strategy = recovery_result.strategy_name

                # Record the failed attempt for ML training
                self.oom_recovery.record_failure(
                    attempt=attempt,
                    strategy=strategy,
                    params=current_params,
                    error_message=error_str,
                )

                # Clear memory according to strategy
                if strategy == "clear_memory" or attempt > 1:
                    self._logger.info(
                        f"🧹 Executing memory cleanup (strategy: {strategy})"
                    )
                    try:
                        self._clear_pipeline_and_memory()
                        # Give extra time for memory cleanup on higher attempts
                        time.sleep(min(attempt * 0.5, 3.0))
                    except Exception as cleanup_e:
                        self._logger.warning(
                            f"Error during memory cleanup: {cleanup_e}"
                        )

                # Check if we've reached max attempts
                if attempt >= max_attempts:
                    self._logger.error(
                        f"❌ Failed to initialize after {max_attempts} intelligent recovery attempts"
                    )

                    # Log recovery statistics for debugging
                    stats = self.oom_recovery.get_statistics()
                    self._logger.info(f"OOM Recovery Statistics: {stats}")

                    raise RuntimeError(
                        f"Failed to initialize {self.model.name} after {max_attempts} intelligent recovery attempts. "
                        f"Last strategy: {strategy}, Final params: {new_params}"
                    ) from e

                # Update parameters for next attempt
                current_params = new_params

        # Should never reach here due to the attempt >= max_attempts check above
        raise RuntimeError(f"Unexpected end of recovery loop for {self.model.name}")

    def _initialize_llama_simple(self, gguf_path: str) -> llama_cpp.Llama:
        """
        Simple Llama initialization without intelligent OOM recovery.
        Fails fast on any errors instead of attempting recovery.
        """
        if llama_cpp.Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        # Use profile parameters directly without ML optimization
        params = self.profile.parameters
        n_ctx = params.num_ctx or 8192  # Conservative default
        n_batch = params.batch_size or 64  # Conservative default
        # Use GPU configuration from profile or default to full GPU usage
        gpu_config = self.profile.gpu_config or DEFAULT_GPU_CONFIG
        n_gpu_layers = (
            gpu_config.gpu_layers if gpu_config.gpu_layers is not None else -1
        )
        gcfg = self.profile.gpu_config or DEFAULT_GPU_CONFIG

        self._logger.info(
            f"🚀 Simple initialization {self.model.name}: "
            f"n_ctx={n_ctx:,}, n_batch={n_batch}, gpu_layers={n_gpu_layers}"
        )

        try:
            # Simple, direct initialization - no retries
            llama_instance = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_ROW,
                tensor_split=gcfg.tensor_split,
                vocab_only=False,
                use_mmap=True,
                use_mlock=False,
                kv_overrides=None,
                # Context Params
                seed=self.profile.parameters.seed or llama_cpp.LLAMA_DEFAULT_SEED,
                n_threads=self._get_optimal_threads(),
                temperature=self.profile.parameters.temperature or 0.7,
                top_p=self.profile.parameters.top_p or 0.8,
                top_k=self.profile.parameters.top_k or 20,
                repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
                f16_kv=True,
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                flash_attn=getattr(self.profile.parameters, "flash_attention", True),
                embedding=False,
                chat_format=None,
                n_threads_batch=None,
                rope_scaling_type=None,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
                rope_freq_base=0.0,
                rope_freq_scale=0.0,
                yarn_ext_factor=-1.0,
                yarn_attn_factor=1.0,
                yarn_beta_fast=32.0,
                yarn_beta_slow=1.0,
                yarn_orig_ctx=0,
                offload_kqv=False,
                op_offload=None,
                swa_full=None,
                no_perf=False,
                last_n_tokens_size=64,
                lora_base=None,
                lora_scale=1.0,
                lora_path=None,
                numa=False,
                chat_handler=None,
                draft_model=None,
                tokenizer=None,
                type_k=None,
                type_v=None,
                smp_infill=False,
            )

            self._logger.info(f"✅ Simple initialization successful: {self.model.name}")
            return llama_instance

        except Exception as e:
            self._logger.error(
                f"❌ Simple initialization failed for {self.model.name}: {e}"
            )
            raise RuntimeError(
                f"Failed to initialize {self.model.name}: {e}. "
                f"Enable ENABLE_INTELLIGENT_OOM_RECOVERY=true for advanced recovery."
            ) from e

    def _format_messages_for_llama(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """Convert LangChain messages to simple dict format for llama-cpp-python."""
        llama_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                llama_messages.append(
                    {"role": "system", "content": str(message.content)}
                )
            elif isinstance(message, HumanMessage):
                llama_messages.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, AIMessage):
                llama_messages.append(
                    {"role": "assistant", "content": str(message.content)}
                )
            elif isinstance(message, ToolMessage):
                # Format tool results as user messages for now
                llama_messages.append(
                    {"role": "user", "content": f"Tool result: {message.content}"}
                )
            else:
                # Fallback: treat as user message
                llama_messages.append({"role": "user", "content": str(message.content)})

        return llama_messages

    def _calculate_usage_metadata(
        self, prompt_tokens: int, completion_tokens: int
    ) -> UsageMetadata:
        """Calculate usage metadata for the response."""
        return UsageMetadata(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _convert_tools_to_simple_format(self, tools):
        """Convert LangChain tools to simple format for llama-cpp-python."""
        if not tools:
            return None

        converted_tools = []
        for tool in tools:
            try:
                # Simple tool format - just name, description, and basic parameters
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                        },
                    }

                    # Add a simple parameters schema (filtered to exclude injected params)
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        try:
                            # Try to get schema, handling ToolRuntime CallableSchema issues
                            if hasattr(tool.args_schema, "model_json_schema"):
                                try:
                                    schema = tool.args_schema.model_json_schema()
                                except Exception as schema_error:
                                    if "CallableSchema" in str(schema_error):
                                        # ToolRuntime has CallableSchema that can't serialize
                                        # Provide minimal schema for tools with ToolRuntime
                                        schema = {
                                            "type": "object",
                                            "properties": {
                                                "query": {
                                                    "type": "string",
                                                    "description": "Search query or input text",
                                                }
                                            },
                                            "required": ["query"],
                                        }
                                    else:
                                        raise schema_error
                            else:
                                schema = {"type": "object", "properties": {}}

                            # Filter out injected LangGraph parameters and ToolRuntime
                            if "properties" in schema:
                                filtered_props = {
                                    k: v
                                    for k, v in schema["properties"].items()
                                    if k
                                    not in ["state", "tool_call_id", "tool_runtime"]
                                }

                                tool_dict["function"]["parameters"] = {
                                    "type": "object",
                                    "properties": filtered_props,
                                    "required": [
                                        req
                                        for req in schema.get("required", [])
                                        if req
                                        not in ["state", "tool_call_id", "tool_runtime"]
                                    ],
                                }
                            else:
                                tool_dict["function"]["parameters"] = {
                                    "type": "object",
                                    "properties": {},
                                }
                        except Exception as e:
                            self._logger.warning(
                                f"Could not extract schema for tool {tool.name}: {e}"
                            )
                            tool_dict["function"]["parameters"] = {
                                "type": "object",
                                "properties": {},
                            }
                    else:
                        tool_dict["function"]["parameters"] = {
                            "type": "object",
                            "properties": {},
                        }

                    converted_tools.append(tool_dict)

            except Exception as e:
                self._logger.error(f"Error converting tool: {e}")
                continue

        return converted_tools if converted_tools else None

    def _parse_tool_calls_from_content(
        self, content: str
    ) -> Tuple[str, List[LangChainToolCall]]:
        """
        Parse tool calls from LlamaCpp text output and clean content.

        Handles both XML-wrapped format:
            <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        And bare JSON format:
            {"name": "func", "parameters": {...}}

        Returns:
            Tuple of (cleaned_content, tool_calls_list)
        """

        tool_calls: List[LangChainToolCall] = []
        cleaned_content = content

        # First, try XML-wrapped format with generic tags
        tool_call_pattern = (
            r"<(?:tool|function)[-_]call>\s*(\{.*?\})\s*</(?:tool|function)[-_]call>"
        )

        matches = re.finditer(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                # Parse the JSON inside the tool_call tags
                json_str = match.group(1).strip()
                tool_data = json.loads(json_str)

                # Convert to LangChain flat format
                tool_call = LangChainToolCall(
                    id=f"call_{len(tool_calls)}",  # Generate ID
                    name=tool_data.get("name", ""),
                    args=tool_data.get("arguments", {}),
                    type="tool_call",
                )

                tool_calls.append(tool_call)

                # Remove this tool call from content
                cleaned_content = cleaned_content.replace(match.group(0), "").strip()

            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(
                    f"Failed to parse XML tool call: {e}, content: {match.group(1)}"
                )
                continue

        # Second, try XML-wrapped format with tool name as tag (e.g., <web_search>...</web_search>)
        if not tool_calls:
            tool_name_pattern = r"<([a-zA-Z_][a-zA-Z0-9_]*?)>\s*(\{.*?\})\s*</\1>"

            matches = re.finditer(tool_name_pattern, content, re.DOTALL | re.IGNORECASE)

            for match in matches:
                try:
                    tool_name_from_tag = match.group(1).strip()
                    if tool_name_from_tag.lower() not in [
                        t.name.lower() for t in self._bound_tools
                    ]:
                        self._logger.warning(
                            f"Unrecognized tool name in tag: {tool_name_from_tag}, skipping."
                        )
                        continue

                    json_str = match.group(2).strip()
                    tool_data = json.loads(json_str)

                    # Convert to LangChain flat format
                    tool_call = LangChainToolCall(
                        id=f"call_{len(tool_calls)}",  # Generate ID
                        name=tool_data.get(
                            "name", tool_name_from_tag
                        ),  # Prefer name from JSON, fallback to tag name
                        args=tool_data.get("arguments", {}),
                        type="tool_call",
                    )
                    tool_calls.append(tool_call)

                    # Remove this tool call from content
                    cleaned_content = cleaned_content.replace(
                        match.group(0), ""
                    ).strip()

                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse tool name XML format: {e}, tool_name: {tool_name_from_tag}, content: {match.group(2)}"
                    )
                    continue

        # If no XML-wrapped tool calls found, try bare JSON format
        if not tool_calls:
            # Pattern to match bare JSON tool calls like {"name": "func", "parameters": {...}}
            # Use a more flexible pattern that can handle nested JSON
            bare_json_pattern = (
                r'\{"name":\s*"([^"]+)"\s*,\s*"parameters":\s*\{.*?\}\s*\}'
            )

            matches = re.finditer(bare_json_pattern, content, re.DOTALL)

            for match in matches:
                try:
                    # Parse the full JSON tool call
                    json_str = match.group(0).strip()

                    # Handle cases where there might be extra text after the closing brace
                    # Find the balanced JSON object
                    brace_count = 0
                    end_pos = 0
                    for i, char in enumerate(json_str):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break

                    if end_pos > 0:
                        json_str = json_str[:end_pos]

                    tool_data = json.loads(json_str)

                    # Convert to LangChain flat format (parameters -> args)
                    tool_call = LangChainToolCall(
                        id=f"call_{len(tool_calls)}",  # Generate ID
                        name=tool_data.get("name", ""),
                        args=tool_data.get(
                            "parameters", {}
                        ),  # Use parameters instead of arguments
                        type="tool_call",
                    )
                    tool_calls.append(tool_call)

                    # Remove this tool call from content
                    cleaned_content = cleaned_content.replace(json_str, "").strip()

                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse bare JSON tool call: {e}, content: {match.group(0)}"
                    )
                    continue

        # Also clean up <think> tags if present
        # think_pattern = r"<think>.*?</think>"
        # cleaned_content = re.sub(
        #     think_pattern, "", cleaned_content, flags=re.DOTALL
        # ).strip()

        # CRITICAL FIX: Clean up repeated "assistant" tokens that appear after tool calls
        # This prevents the infinite loop issue where models see "assistantassistant" in conversation history
        if tool_calls and cleaned_content:
            # Remove any trailing "assistant" tokens (case insensitive)
            cleaned_content = re.sub(
                r"assistant+\s*$", "", cleaned_content, flags=re.IGNORECASE
            ).strip()
            # Also remove any repeated assistant tokens in the middle
            cleaned_content = re.sub(
                r"\s*assistant+\s*", " ", cleaned_content, flags=re.IGNORECASE
            ).strip()

        return cleaned_content, tool_calls

    def _get_res(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
    ):
        """Get response from llama-cpp-python with simplified formatting."""
        assert self.llama_instance

        # Convert tools to simple format (keeping the fix for injected params)
        converted_tools = self._convert_tools_to_simple_format(tools)

        # Simple message conversion - let llama-cpp-python handle context limits
        llama_messages = self._format_messages_for_llama(messages)

        # Basic logging without excessive detail
        self._logger.info(
            f"Chat completion: model={self.model.name}, "
            f"messages={len(llama_messages)}, "
            f"tools={len(converted_tools) if converted_tools else 0}"
        )

        # Setup grammar if needed
        response_format = None
        grammar = None
        if self.grammar:
            response_format = {
                "type": "json_object",
                "schema": self.grammar.model_json_schema(),
            }
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(self.grammar.model_json_schema())
            )

        # Simple call to llama-cpp-python - let it handle the complexity
        kwargs = {
            "messages": llama_messages,  # type: ignore
            "temperature": self.profile.parameters.temperature or 0.7,
            "top_p": self.profile.parameters.top_p or 0.95,
            "top_k": self.profile.parameters.top_k or 40,
            "stream": stream,
            "stop": self.profile.parameters.stop or stop,
            "max_tokens": self.profile.parameters.max_tokens or 4096,
            "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
        }

        if converted_tools:
            kwargs["tools"] = converted_tools  # type: ignore
            kwargs["tool_choice"] = "auto"

        if response_format:
            kwargs["response_format"] = response_format  # type: ignore

        if grammar:
            kwargs["grammar"] = grammar

        return self.llama_instance.create_chat_completion(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from messages."""
        # Combine bound tools with any tools passed in kwargs
        tools = kwargs.get("tools", [])
        if tools:
            self._bound_tools = list(set(list(self._bound_tools) + list(tools)))
        tools = self._bound_tools

        try:
            response = self._get_res(
                messages=messages,
                stop=stop,
                tools=tools,
                stream=False,
            )

            # For non-streaming, response should be a dict
            if isinstance(response, dict):
                res = cast(CreateChatCompletionResponse, response)
                message = res.get("choices", [])[0].get("message", {})
                content = message.get("content", "") or ""
                usage = res.get("usage", {})

                # Parse tool calls from content if present
                cleaned_content, tool_calls = self._parse_tool_calls_from_content(
                    content
                )

                # Create usage metadata
                usage_metadata = self._calculate_usage_metadata(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )

                # Create AI message with tool calls
                message = AIMessage(
                    content=cleaned_content,
                    tool_calls=tool_calls if tool_calls else [],
                    usage_metadata=usage_metadata,
                    response_metadata={
                        "model_name": self.model.name,
                        "finish_reason": response["choices"][0].get("finish_reason"),
                    },
                )

                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                raise ValueError("Expected dict response for non-streaming generation")

        except Exception as e:
            self._logger.error(f"Generation failed: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response chunks."""
        # Combine bound tools with any tools passed in kwargs
        tools = kwargs.get("tools", [])
        if hasattr(self, "_bound_tools") and self._bound_tools:
            tools = list(self._bound_tools) + list(tools or [])

        try:
            # Stream response using llama-cpp-python - simple dict iteration
            response_stream = self._get_res(
                messages=messages,
                stop=stop,
                tools=tools,
                stream=True,
            )

            # For streaming, response should be an iterator
            accumulated_content = ""
            for chunk in response_stream:
                # Handle chunk as a dict
                if isinstance(chunk, dict) and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "") or ""
                    finish_reason = chunk["choices"][0].get("finish_reason")

                    # Accumulate content for tool call parsing
                    if content:
                        accumulated_content += content

                    # Create usage metadata if available
                    usage_metadata = None
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        if isinstance(usage, dict):
                            usage_metadata = self._calculate_usage_metadata(
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                            )

                    # For final chunk, parse tool calls and clean content
                    if finish_reason == "stop":
                        cleaned_content, tool_calls = (
                            self._parse_tool_calls_from_content(accumulated_content)
                        )

                        # Send final chunk with tool calls if any were found
                        if tool_calls:
                            final_chunk_message = AIMessageChunk(
                                content=cleaned_content,
                                tool_calls=tool_calls,  # type: ignore
                                usage_metadata=usage_metadata,
                                response_metadata={
                                    "model_name": self.model.name,
                                    "finish_reason": finish_reason,
                                },
                                chunk_position="last",
                            )

                            final_generation_chunk = ChatGenerationChunk(
                                message=final_chunk_message
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    "", chunk=final_generation_chunk
                                )
                            yield final_generation_chunk
                            continue

                    # Create regular chunk message
                    chunk_message = AIMessageChunk(
                        content=content,
                        usage_metadata=usage_metadata,
                        response_metadata=(
                            {
                                "model_name": self.model.name,
                                "finish_reason": finish_reason,
                            }
                            if finish_reason
                            else {}
                        ),
                        chunk_position="last" if finish_reason == "stop" else None,
                    )

                    # Create and yield generation chunk
                    generation_chunk = ChatGenerationChunk(message=chunk_message)

                    if run_manager:
                        run_manager.on_llm_new_token(content, chunk=generation_chunk)

                    yield generation_chunk

        except Exception as e:
            self._logger.error(f"Streaming failed: {e}")
            raise

    def close(self):
        """Clean up resources."""
        if hasattr(self, "llama_instance") and self.llama_instance:
            try:
                self.llama_instance.close()
            except Exception:
                pass
            self.llama_instance = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


__all__ = ["BaseLlamaCppPipeline"]
