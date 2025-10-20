"""
BaseLlamaCppPipeline as a custom BaseChatModel implementation.
Uses llama-cpp-python directly instead of LangChain's ChatLlamaCpp wrapper.
"""

import json
import os
import multiprocessing

from typing import Optional, List, Any, Dict, Iterator, Type, cast, Tuple

from pydantic import BaseModel
import llama_cpp
from llama_cpp import llama_types
from llama_cpp import llama_grammar

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from models import Model, ModelProfile
from models.default_configs import DEFAULT_GPU_CONFIG
from utils.logging import llmmllogger
from runner.utils.hardware_manager import EnhancedHardwareManager
from .utils import calculate_optimal_gpu_layers


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
        super().__init__(model=model, profile=profile, grammar=grammar, **kwargs) # type: ignore
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )
        self.grammar = grammar
        self._bound_tools = kwargs.get("_bound_tools", [])
        self.hardware_manager = EnhancedHardwareManager()
        self.llama_instance = self._initialize_llama_with_oom_retry(
            self._get_gguf_path()
        )

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

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "BaseLlamaCppPipeline":
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
        # For large models, reuse the existing instance instead of creating a new one
        # to avoid OOM when both instances exist simultaneously
        model_size_gb = self.model.size / (1024**3) if self.model.size else 0
        
        if model_size_gb > 10:  # Large models (>10GB)
            self._logger.info(f"🔄 Reusing existing instance for large model {self.model.name} ({model_size_gb:.1f}GB)")
            # Update bound tools on the existing instance
            self._bound_tools = tools
            return self
        else:
            # For smaller models, create a new instance as before
            self._logger.debug(f"🆕 Creating new instance for small model {self.model.name} ({model_size_gb:.1f}GB)")
            new_instance = self.__class__(
                model=self.model,
                profile=self.profile,
                grammar=self.grammar,
                _bound_tools=tools,
                **kwargs,
            )
            return new_instance

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
            except:
                pass
            del self.llama_instance
            self.llama_instance = None

        # Clear GPU memory
        try:
            self.hardware_manager.clear_memory(aggressive=True)
        except Exception as e:
            self._logger.warning(f"Error clearing GPU memory: {e}")

        # Give GPU a moment to clean up
        import time

        time.sleep(1)

    def _initialize_llama_with_oom_retry(self, gguf_path: str) -> llama_cpp.Llama:
        """Initialize Llama instance - try with requested parameters, handle OOM reactively."""
        if llama_cpp.Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        # Start with requested parameters
        n_ctx = self.profile.parameters.num_ctx or 32768
        n_batch = self.profile.parameters.batch_size or 512
        n_ubatch = 512

        # GPU layers
        n_gpu_layers = -1  # Default to full offload
        if (
            self.profile.gpu_config is not None
            and self.profile.gpu_config.gpu_layers is not None
        ):
            n_gpu_layers = self.profile.gpu_config.gpu_layers

        # Perplexity / logits guard
        perplexity_enabled = bool(
            getattr(self.profile.parameters, "enable_perplexity_guard", False)
        )

        # Retry parameters for OOM handling
        oom_attempts = 0
        max_oom_attempts = 6

        gcfg = self.profile.gpu_config or DEFAULT_GPU_CONFIG

        while oom_attempts < max_oom_attempts:
            failed_instance = None
            try:
                self._logger.info(
                    f"Attempting to initialize {self.model.name} with n_ctx={n_ctx}, n_batch={n_batch}, n_ubatch={n_ubatch}, gpu_layers={n_gpu_layers}"
                )

                llama_instance = llama_cpp.Llama(
                    model_path=gguf_path,
                    n_gpu_layers=n_gpu_layers,
                    split_mode=llama_cpp.LLAMA_SPLIT_MODE_ROW,
                    tensor_split=gcfg.tensor_split,
                    vocab_only=False,
                    use_mmap=True,
                    use_mlock=False,
                    kv_overrides=None,
                    # Context Params
                    seed=self.profile.parameters.seed or llama_cpp.LLAMA_DEFAULT_SEED,
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    n_ubatch=n_ubatch,
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
                    spm_infill=False,
                )

                self._logger.info(
                    f"✅ Successfully initialized {self.model.name} with n_ctx={n_ctx:,}, n_batch={n_batch}, n_ubatch={n_ubatch}, gpu_layers={n_gpu_layers}"
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
                    ]
                )

                if is_oom:
                    oom_attempts += 1
                    self._logger.warning(
                        f"🔥 OOM detected (attempt {oom_attempts}/{max_oom_attempts}): {error_str}"
                    )

                    # CRITICAL: Immediately cleanup any failed instance that may have partial CUDA state
                    # Even if llama_cpp.Llama() constructor failed, it may have allocated GPU memory
                    try:
                        # Try to capture and close the failed instance if it exists in local scope
                        if "llama_instance" in locals():
                            try:
                                llama_instance.close()
                                self._logger.debug("🧹 Closed failed llama_instance")
                            except:
                                pass
                            del llama_instance

                        # Force aggressive cleanup
                        import gc

                        gc.collect()

                        # Nuclear memory clear to remove any GPU allocations
                        self.hardware_manager.clear_memory(
                            aggressive=True, nuclear=True
                        )
                        self._logger.info(
                            f"🧹 Cleaned up failed pipeline instance (attempt {oom_attempts})"
                        )
                    except Exception as cleanup_e:
                        self._logger.warning(
                            f"Error during failed pipeline cleanup: {cleanup_e}"
                        )

                    if oom_attempts >= max_oom_attempts:
                        self._logger.error(
                            f"❌ Failed to initialize after {max_oom_attempts} OOM recovery attempts"
                        )
                        raise RuntimeError(
                            f"Failed to initialize after {max_oom_attempts} OOM recovery attempts"
                        )

                    # Clear any remaining memory and reduce parameters
                    self._logger.info(
                        f"🧹 Final memory clear before attempt {oom_attempts + 1}"
                    )
                    self._clear_pipeline_and_memory()

                    if oom_attempts == 1:
                        # First OOM - just clear and retry (might be stale memory)
                        self._logger.info(
                            "🔄 Attempt 1: Clearing memory and retrying with same parameters"
                        )
                        continue
                    elif oom_attempts == 2:
                        # Second OOM - reduce batch sizes
                        n_batch = max(128, n_batch // 2)
                        n_ubatch = max(128, n_ubatch // 2)
                        self._logger.info(
                            f"🔄 Attempt 2: Reducing batch sizes: n_batch={n_batch}, n_ubatch={n_ubatch}"
                        )
                    elif oom_attempts == 3:
                        # Third OOM - reduce batch sizes more
                        n_batch = max(64, n_batch // 2)
                        n_ubatch = max(64, n_ubatch // 2)
                        self._logger.info(
                            f"🔄 Attempt 3: Reducing batch sizes further: n_batch={n_batch}, n_ubatch={n_ubatch}"
                        )
                    elif oom_attempts == 4:
                        # Fourth OOM - reduce context size
                        n_ctx = max(4096, n_ctx // 2)
                        self._logger.info(
                            f"🔄 Attempt 4: Reducing context size: n_ctx={n_ctx}"
                        )
                    elif oom_attempts == 5:
                        # Fifth OOM - reduce context size more
                        n_ctx = max(2048, n_ctx // 2)
                        self._logger.info(
                            f"🔄 Attempt 5: Reducing context size further: n_ctx={n_ctx}"
                        )

                else:
                    # Not an OOM error, re-raise immediately
                    self._logger.error(f"❌ Non-OOM error during initialization: {e}")
                    raise e

        raise RuntimeError(
            f"Failed to initialize {self.model.name} after all OOM recovery attempts"
        )

    def _format_messages_for_llama(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to simple dict format for llama-cpp-python."""
        llama_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                llama_messages.append({"role": "system", "content": str(message.content)})
            elif isinstance(message, HumanMessage):
                llama_messages.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, AIMessage):
                llama_messages.append({"role": "assistant", "content": str(message.content)})
            elif isinstance(message, ToolMessage):
                # Format tool results as user messages for now
                llama_messages.append({"role": "user", "content": f"Tool result: {message.content}"})
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
                        }
                    }

                    # Add a simple parameters schema (filtered to exclude injected params)
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        try:
                            if hasattr(tool.args_schema, "model_json_schema"):
                                schema = tool.args_schema.model_json_schema()
                            else:
                                schema = {"type": "object", "properties": {}}

                            # Filter out injected LangGraph parameters
                            if 'properties' in schema:
                                filtered_props = {
                                    k: v for k, v in schema['properties'].items() 
                                    if k not in ['state', 'tool_call_id']
                                }
                                
                                tool_dict["function"]["parameters"] = {
                                    "type": "object",
                                    "properties": filtered_props,
                                    "required": [
                                        req for req in schema.get('required', []) 
                                        if req not in ['state', 'tool_call_id']
                                    ]
                                }
                            else:
                                tool_dict["function"]["parameters"] = {
                                    "type": "object", 
                                    "properties": {}
                                }
                        except Exception as e:
                            self._logger.warning(f"Could not extract schema for tool {tool.name}: {e}")
                            tool_dict["function"]["parameters"] = {"type": "object", "properties": {}}
                    else:
                        tool_dict["function"]["parameters"] = {"type": "object", "properties": {}}

                    converted_tools.append(tool_dict)

            except Exception as e:
                self._logger.error(f"Error converting tool: {e}")
                continue

        return converted_tools if converted_tools else None

    def _parse_tool_calls_from_content(self, content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse tool calls from LlamaCpp text output and clean content.
        
        Returns:
            Tuple of (cleaned_content, tool_calls_list)
        """
        import re
        
        tool_calls = []
        cleaned_content = content
        
        # Pattern to match both <tool_call> and <function-call> blocks
        tool_call_pattern = r'<(?:tool_call|function-call)>\s*(\{.*?\})\s*</(?:tool_call|function-call)>'
        
        matches = re.finditer(tool_call_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                # Parse the JSON inside the tool_call tags
                json_str = match.group(1).strip()
                tool_data = json.loads(json_str)
                
                # Convert to LangChain flat format
                tool_call = {
                    "id": f"call_{len(tool_calls)}",  # Generate ID
                    "name": tool_data.get("name", ""),
                    "args": tool_data.get("arguments", {}),
                    "type": "tool_call"
                }
                tool_calls.append(tool_call)
                
                # Remove this tool call from content
                cleaned_content = cleaned_content.replace(match.group(0), "").strip()
                
            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(f"Failed to parse tool call: {e}, content: {match.group(1)}")
                continue
        
        # Also clean up <think> tags if present
        think_pattern = r'<think>.*?</think>'
        cleaned_content = re.sub(think_pattern, '', cleaned_content, flags=re.DOTALL).strip()
        
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
        return self.llama_instance.create_chat_completion(
            messages=llama_messages,
            tools=converted_tools,
            tool_choice="auto" if converted_tools else None,
            temperature=self.profile.parameters.temperature or 0.7,
            top_p=self.profile.parameters.top_p or 0.95,
            top_k=self.profile.parameters.top_k or 40,
            stream=stream,
            stop=self.profile.parameters.stop or stop or [],
            max_tokens=self.profile.parameters.max_tokens or 4096,
            repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
            response_format=response_format,
            grammar=grammar,
        )

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
        if hasattr(self, "_bound_tools") and self._bound_tools:
            tools = list(self._bound_tools) + list(tools or [])

        try:
            response = self._get_res(
                messages=messages,
                stop=stop,
                tools=tools,
                stream=False,
            )

            # For non-streaming, response should be a dict
            if isinstance(response, dict):
                content = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})

                # Parse tool calls from content if present
                cleaned_content, tool_calls = self._parse_tool_calls_from_content(content or "")

                # Create usage metadata
                usage_metadata = self._calculate_usage_metadata(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )

                # Create AI message with tool calls
                message = AIMessage(
                    content=cleaned_content,
                    tool_calls=tool_calls if tool_calls else None,
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
                        cleaned_content, tool_calls = self._parse_tool_calls_from_content(accumulated_content)
                        
                        # Send final chunk with tool calls if any were found
                        if tool_calls:
                            final_chunk_message = AIMessageChunk(
                                content=cleaned_content,
                                tool_calls=tool_calls,
                                usage_metadata=usage_metadata,
                                response_metadata={
                                    "model_name": self.model.name,
                                    "finish_reason": finish_reason,
                                },
                                chunk_position="last",
                            )
                            
                            final_generation_chunk = ChatGenerationChunk(message=final_chunk_message)
                            if run_manager:
                                run_manager.on_llm_new_token("", chunk=final_generation_chunk)
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
        if self.llama_instance:
            try:
                self.llama_instance.close()
            except Exception:
                pass
            self.llama_instance = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


__all__ = ["BaseLlamaCppPipeline"]
