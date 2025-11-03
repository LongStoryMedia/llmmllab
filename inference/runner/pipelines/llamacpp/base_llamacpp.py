"""
BaseLlamaCppPipeline as a custom BaseChatModel implementation.
Uses llama-cpp-python directly instead of LangChain's ChatLlamaCpp wrapper.
"""

import json
import multiprocessing
import os
import re
import time

from typing import (
    Callable,
    Optional,
    List,
    Any,
    Dict,
    Iterator,
    Type,
    Tuple,
    cast,
    Sequence,
)

from langchain_core.runnables import Runnable
from pydantic import BaseModel
import llama_cpp
from llama_cpp import llama_grammar
from llama_cpp.llama_types import CreateChatCompletionResponse
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler

from langchain_core.prompt_values import PromptValue
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
from runner.pipelines.base import BasePipeline


class BaseLlamaCppPipeline(BasePipeline):
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
        """Base LlamaCpp pipeline implementation.

        Experiment 4 adds optional single-GPU isolation to rule out mixed compute capability issues.
        Enable with environment variable:
            EXPERIMENT_SINGLE_GPU=true (forces CUDA_VISIBLE_DEVICES to EXPERIMENT_SINGLE_GPU_ID or '1')
            EXPERIMENT_SINGLE_GPU_ID=1 (defaults to 1 if unset)
        """

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
        self.oom_recovery = (
            IntelligentOOMRecovery() if self.use_intelligent_oom else None
        )
        self.llama_instance = self._initialize_llama(self._get_gguf_path())

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-cpp-custom"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model.name,
            "model_path": self._get_gguf_path(),
            "n_ctx": self.profile.parameters.num_ctx or 4096,
            "temperature": self.profile.parameters.temperature or 0.7,
        }

    def _get_gguf_path(self) -> str:
        """Return resolved GGUF file path for model."""
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model

    def _get_optimal_threads(self) -> int:
        """Determine a conservative optimal thread count."""
        try:
            cpu_count = multiprocessing.cpu_count()
            return min(max(cpu_count // 2, 2), 8)
        except Exception:
            self._logger.warning(
                "Could not determine CPU count, defaulting to 4 threads"
            )
            return 4

    def _initialize_llama(
        self, gguf_path: str, handler: Optional[LlamaChatCompletionHandler] = None
    ) -> llama_cpp.Llama:
        """Initialize the llama_cpp.Llama instance with optional OOM recovery.

        Single authoritative instantiation path:
        - Always attempt profile parameters first.
        - If intelligent OOM recovery is configured, retry only on OOM-class errors.
        - Non-OOM exceptions raise immediately (no parameter downscaling).
        """
        if llama_cpp.Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        params = self.profile.parameters
        gcfg = self.profile.gpu_config or DEFAULT_GPU_CONFIG
        n_ctx_initial = params.num_ctx or 40960
        n_batch_initial = params.batch_size or 256
        n_ubatch_initial = params.batch_size or 256
        perplexity_enabled = bool(getattr(params, "enable_perplexity_guard", False))

        # Compose initial optimal parameters
        original_params = OptimalParameters(
            n_ctx=n_ctx_initial,
            n_batch=n_batch_initial,
            n_ubatch=n_ubatch_initial,
            n_gpu_layers=gcfg.gpu_layers or -1,
        )

        # Prepare attempt parameter list (profile first, then predicted if recovery available)
        attempt_params_list: List[OptimalParameters] = [original_params]
        if self.oom_recovery is not None:
            try:
                predicted = self.oom_recovery.predict_optimal_parameters_from_profile(
                    model_profile=self.profile,
                    model_path=gguf_path,
                    hardware_manager=self.hardware_manager,
                ).model_copy()
                attempt_params_list.append(predicted)
            except Exception as e:
                self._logger.warning(
                    f"OOM recovery prediction failed; continuing with original params only: {e}"
                )

        max_attempts = 1 if self.oom_recovery is None else 10
        current_params = attempt_params_list[0]
        attempt_index = 0  # track transition to predicted params after first OOM

        attempt = 1
        while attempt <= max_attempts:
            try:
                self._logger.info(
                    f"🚀 Initializing {self.model.name} (attempt {attempt}): n_ctx={current_params.n_ctx:,}, n_batch={current_params.n_batch}, n_ubatch={current_params.n_ubatch}, gpu_layers={current_params.n_gpu_layers}, stop={params.stop}"
                )
                start_time = time.time()
                llama_instance = llama_cpp.Llama(
                    model_path=gguf_path,
                    n_ctx=current_params.n_ctx,
                    n_batch=current_params.n_batch,
                    n_ubatch=current_params.n_ubatch,
                    n_gpu_layers=current_params.n_gpu_layers,
                    tensor_split=gcfg.tensor_split,
                    chat_format=self._get_chat_format(),
                    vocab_only=False,
                    use_mmap=True,
                    use_mlock=False,
                    kv_overrides=None,
                    seed=params.seed or llama_cpp.LLAMA_DEFAULT_SEED,
                    n_threads=self._get_optimal_threads(),
                    temperature=params.temperature or 0.7,
                    top_p=params.top_p or 0.95,
                    top_k=params.top_k or 40,
                    repeat_penalty=params.repeat_penalty or 1.05,
                    f16_kv=True,
                    verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                    flash_attn=getattr(params, "flash_attention", True),
                    logits_all=perplexity_enabled,
                    logprobs=1 if perplexity_enabled else 0,
                    embedding=False,
                    n_threads_batch=None,
                    rope_scaling_type=llama_cpp.llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN,
                    pooling_type=llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
                    rope_freq_base=0.0,
                    rope_freq_scale=1.0,
                    yarn_ext_factor=4.0,
                    yarn_attn_factor=1.0,
                    yarn_beta_fast=32.0,
                    yarn_beta_slow=1.0,
                    yarn_orig_ctx=262144,
                    offload_kqv=False,
                    op_offload=None,
                    swa_full=None,
                    kv_unified=None,
                    no_perf=False,
                    last_n_tokens_size=64,
                    lora_base=None,
                    lora_scale=1.0,
                    lora_path=None,
                    numa=False,
                    chat_handler=handler,  # type: ignore[arg-type]
                    draft_model=None,
                    tokenizer=None,
                    type_k=None,
                    type_v=None,
                    smp_infill=False,
                )
                init_ms = (time.time() - start_time) * 1000
                gpu_used = 0
                try:
                    stats = self.hardware_manager.update_all_memory_stats()
                    for s in stats.values():
                        if hasattr(s, "id") and s.id == 0 and hasattr(s, "mem_used"):
                            gpu_used = s.mem_used
                            break
                except Exception:
                    pass

                if self.oom_recovery is not None:
                    self.oom_recovery.record_success(
                        model_path=gguf_path,
                        params=current_params,
                        hardware_manager=self.hardware_manager,
                        initialization_time_ms=init_ms,
                        gpu_memory_used_mb=gpu_used,
                    )
                self._logger.info(
                    f"✅ Initialized {self.model.name}: n_ctx={current_params.n_ctx:,}, n_batch={current_params.n_batch}, n_ubatch={current_params.n_ubatch} in {init_ms:.1f}ms"
                )
                return llama_instance
            except Exception as e:
                err = str(e).lower()
                is_oom = any(
                    t in err
                    for t in [
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
                if not is_oom or self.oom_recovery is None:
                    # Non-OOM failure OR recovery not configured: surface immediately
                    self._logger.error(f"❌ Initialization failed: {e}")
                    raise

                # OOM recovery path
                self._logger.warning(f"🔥 OOM detected (attempt {attempt}): {e}")
                try:
                    if "llama_instance" in locals():
                        llama_instance.close()
                except Exception:
                    pass
                self.oom_recovery.record_failure(
                    attempt=attempt,
                    strategy="clear_memory",
                    params=current_params,
                    error_message=err,
                )
                recovery = self.oom_recovery.execute_recovery_strategy(
                    attempt=attempt,
                    original_params=original_params,
                    current_params=current_params,
                    hardware_manager=self.hardware_manager,
                )
                current_params = recovery.parameters
                # After first OOM move to predicted params (if available) before further downscaling
                if attempt_index == 0 and len(attempt_params_list) > 1:
                    attempt_index = 1
                    current_params = attempt_params_list[1]
                if attempt >= max_attempts:
                    stats = (
                        self.oom_recovery.get_statistics()
                        if self.oom_recovery is not None
                        else {}
                    )
                    self._logger.error(
                        f"❌ Failed to initialize {self.model.name} after {max_attempts} attempts. Stats: {stats}"
                    )
                    raise RuntimeError(
                        f"Failed to initialize {self.model.name} after {max_attempts} attempts"
                    ) from e
                attempt += 1
        # Should never reach here: attempt starts at 1 and loop raises on max_attempts
        raise RuntimeError(
            f"Initialization fell through unexpectedly for {self.model.name}"
        )

    def _format_messages_for_llama(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str | List[Dict[str, Any]]]]:
        """Convert LangChain messages to simple dict format for llama-cpp-python."""
        llama_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                llama_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                llama_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                llama_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, ToolMessage):
                # Format tool results as user messages for now
                llama_messages.append(
                    {"role": "user", "content": f"Tool result: {message.content}"}
                )
            else:
                # Fallback: treat as user message
                llama_messages.append({"role": "user", "content": message.content})

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

    def _get_chat_format(self) -> Optional[str]:
        """Hook for subclasses to specify chat_format during llama initialization."""
        return "chatml"

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

                # Handle both regular and Hermes-style arguments
                arguments = tool_data.get("arguments", {})

                # If arguments is a string (Hermes format), parse it as JSON
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        self._logger.warning(
                            f"Failed to parse arguments string as JSON: {arguments}"
                        )
                        arguments = {}

                # Convert to LangChain flat format
                tool_call = LangChainToolCall(
                    id=f"call_{len(tool_calls)}",  # Generate ID
                    name=tool_data.get("name", ""),
                    args=arguments,
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

        # If no tool calls found yet, try Hermes-style format specifically
        if not tool_calls:
            # Hermes format: {"name": "tool_name", "arguments": "{\"param\": \"value\"}"}
            # The arguments field is a JSON string, not an object
            hermes_pattern = (
                r'\{"name":\s*"([^"]+)"\s*,\s*"arguments":\s*"([^"]+)"\s*\}'
            )

            matches = re.finditer(hermes_pattern, content, re.DOTALL)

            for match in matches:
                try:
                    tool_name = match.group(1).strip()
                    arguments_str = match.group(2).strip()

                    # Parse the arguments string as JSON
                    # Need to handle escaped quotes in the arguments string
                    arguments_str = arguments_str.replace('\\"', '"').replace(
                        "\\\\", "\\"
                    )
                    arguments = json.loads(arguments_str)

                    tool_call = LangChainToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=tool_name,
                        args=arguments,
                        type="tool_call",
                    )
                    tool_calls.append(tool_call)

                    # Remove this tool call from content
                    cleaned_content = cleaned_content.replace(
                        match.group(0), ""
                    ).strip()

                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse Hermes tool call: {e}, tool_name: {tool_name if 'tool_name' in locals() else 'unknown'}, arguments: {arguments_str if 'arguments_str' in locals() else 'unknown'}"
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
        self._logger.debug(
            f"llama-cpp-python kwargs: stream={stream}, stop={self.profile.parameters.stop or stop}, tools={converted_tools is not None}, response_format={response_format is not None}, grammar={grammar is not None}"
        )
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

    def bind_tools(
        self, tools: List[BaseTool], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[
        PromptValue
        | str
        | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
        AIMessage,
    ]:
        self._bound_tools = tools
        return self

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


__all__ = ["BaseLlamaCppPipeline"]
