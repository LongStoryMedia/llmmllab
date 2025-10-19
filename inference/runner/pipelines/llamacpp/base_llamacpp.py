"""
BaseLlamaCppPipeline as a custom BaseChatModel implementation.
Uses llama-cpp-python directly instead of LangChain's ChatLlamaCpp wrapper.
"""

import json
import os
import multiprocessing

from typing import Optional, List, Any, Dict, Iterator, Type, cast

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
from utils.logging import llmmllogger
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
        super().__init__(model=model, profile=profile, grammar=grammar, **kwargs)
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )
        self.grammar = grammar
        self._bound_tools = kwargs.get('_bound_tools', [])
        self.llama_instance = self._initialize_llama_with_fallback(
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
        # Create a new instance with the same configuration
        new_instance = self.__class__(
            model=self.model,
            profile=self.profile,
            grammar=self.grammar,
            _bound_tools=tools,
            **kwargs
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

    def _get_model_size_category(self) -> str:
        """Determine model size category from model name."""
        model_name = self.model.name.lower()

        if any(x in model_name for x in ["1b", "1.5b", "3b"]):
            return "small"
        elif any(x in model_name for x in ["7b", "8b", "13b"]):
            return "medium"
        elif any(x in model_name for x in ["20b", "30b"]):
            return "large"
        elif any(x in model_name for x in ["70b", "120b"]):
            return "xlarge"
        else:
            return "medium"

    def _build_context_candidates(self, requested_ctx: int) -> List[int]:
        """Build context size candidates for fallback."""
        candidates = [requested_ctx]

        fallbacks = [32768, 16384, 8192, 4096, 2048]
        for fallback in fallbacks:
            if fallback < requested_ctx and fallback not in candidates:
                candidates.append(fallback)

        return sorted(list(set(candidates)), reverse=True)

    def _initialize_llama_with_fallback(self, gguf_path: str) -> llama_cpp.Llama:
        """Initialize Llama instance with fallback strategies."""
        if llama_cpp.Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        # Get base parameters
        requested_ctx = self.profile.parameters.num_ctx or 4096
        requested_batch = self.profile.parameters.batch_size or 512

        # Model size and GPU layer candidates
        model_size_category = self._get_model_size_category()
        context_candidates = self._build_context_candidates(requested_ctx)
        # Perplexity / logits guard (mirrors advanced base logic simplified)
        perplexity_enabled = bool(
            getattr(self.profile.parameters, "enable_perplexity_guard", False)
        )

        # GPU layers strategy
        explicit_gpu_layers = None
        if (
            self.profile.gpu_config is not None
            and self.profile.gpu_config.gpu_layers is not None
        ):
            explicit_gpu_layers = self.profile.gpu_config.gpu_layers

        # Try different configurations
        for n_ctx in context_candidates:
            n_batch = min(requested_batch, max(256, n_ctx // 64))

            # GPU layer candidates
            if explicit_gpu_layers is not None:
                gpu_candidates = [explicit_gpu_layers]
            else:
                # Try full offload first, then calculated layers
                heuristic = calculate_optimal_gpu_layers(n_ctx, model_size_category)
                gpu_candidates = [-1, heuristic, max(1, int(heuristic * 0.8)), 16]

            for n_gpu_layers in gpu_candidates:
                try:
                    return llama_cpp.Llama(
                        model_path=gguf_path,
                        n_gpu_layers=n_gpu_layers,
                        split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
                        # main_gpu=0,
                        tensor_split=None,
                        vocab_only=False,
                        use_mmap=True,
                        use_mlock=False,
                        kv_overrides=None,
                        # Context Params
                        seed=self.profile.parameters.seed
                        or llama_cpp.LLAMA_DEFAULT_SEED,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        n_ubatch=512,
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
                        logits_all=perplexity_enabled,  # Only enable if needed for specific features
                        logprobs=1 if perplexity_enabled else 0,
                        embedding=False,  # This is for chat, not embeddings
                        chat_format=None,  # Default chat format, can be overridden
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
                        offload_kqv=False,  # Disable offload for better performance unless needed
                        op_offload=None,
                        swa_full=None,
                        # Sampling Params
                        no_perf=False,
                        last_n_tokens_size=64,
                        # LoRA Params
                        lora_base=None,
                        lora_scale=1.0,
                        lora_path=None,
                        # Backend Params
                        numa=False,
                        chat_handler=None,
                        # Speculative Decoding
                        draft_model=None,
                        # Tokenizer Override
                        tokenizer=None,
                        # KV cache quantization
                        type_k=None,
                        type_v=None,
                        # Misc
                        spm_infill=False,
                    )

                except Exception as e:
                    self._logger.warning(
                        f"Failed to initialize with ctx={n_ctx}, batch={n_batch}, "
                        f"gpu_layers={n_gpu_layers}: {e}"
                    )
                    continue

        raise RuntimeError(
            f"Failed to initialize {self.model.name} with any configuration"
        )

    def _format_messages_for_llama(
        self, messages: List[BaseMessage]
    ) -> List[llama_types.ChatCompletionRequestMessage]:
        """Convert LangChain messages to llama-cpp-python chat format."""
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

    def _convert_tools_to_openai_format(self, tools):
        """Convert LangChain tools to OpenAI function calling format for llama-cpp-python."""
        if not tools:
            return None
            
        converted_tools = []
        for tool in tools:
            try:
                # Handle different tool types
                if hasattr(tool, 'args_schema') and hasattr(tool, 'name') and hasattr(tool, 'description'):
                    # LangChain StructuredTool or similar
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                        }
                    }
                    
                    # Add parameters schema if available
                    if tool.args_schema:
                        try:
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                # Pydantic model schema
                                schema = tool.args_schema.model_json_schema()
                            elif hasattr(tool.args_schema, 'schema'):
                                # Other schema types
                                schema = tool.args_schema.schema()
                            else:
                                schema = {"type": "object", "properties": {}}
                                
                            tool_dict["function"]["parameters"] = schema
                        except Exception as e:
                            self._logger.warning(f"Could not extract schema for tool {tool.name}: {e}")
                            tool_dict["function"]["parameters"] = {"type": "object", "properties": {}}
                    else:
                        tool_dict["function"]["parameters"] = {"type": "object", "properties": {}}
                        
                    converted_tools.append(tool_dict)
                    
                elif isinstance(tool, dict):
                    # Already in the right format
                    converted_tools.append(tool)
                    
                else:
                    self._logger.warning(f"Unknown tool type: {type(tool)}, skipping")
                    
            except Exception as e:
                self._logger.error(f"Error converting tool: {e}")
                continue
                
        return converted_tools if converted_tools else None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token for most languages
        return max(1, len(text) // 4)

    def _count_message_tokens(self, messages: List[BaseMessage]) -> int:
        """Count approximate tokens in messages."""
        total_tokens = 0
        for message in messages:
            if hasattr(message, 'content') and message.content:
                total_tokens += self._estimate_tokens(str(message.content))
        return total_tokens

    def _count_tool_tokens(self, tools: Optional[List[llama_types.ChatCompletionTool]]) -> int:
        """Count approximate tokens in tool definitions."""
        if not tools:
            return 0
        
        total_tokens = 0
        for tool in tools:
            if hasattr(tool, 'function'):
                # Count function name, description, and parameters
                if hasattr(tool.function, 'name'):
                    total_tokens += self._estimate_tokens(tool.function.name)
                if hasattr(tool.function, 'description'):
                    total_tokens += self._estimate_tokens(tool.function.description)
                if hasattr(tool.function, 'parameters'):
                    # Parameters schema can be quite large
                    total_tokens += self._estimate_tokens(json.dumps(tool.function.parameters))
        
        return total_tokens

    def _trim_messages_to_context(
        self, 
        messages: List[BaseMessage], 
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        max_tokens: Optional[int] = None
    ) -> List[BaseMessage]:
        """Trim messages to fit within context window."""
        if not max_tokens:
            # Use llama instance context size if available, otherwise default
            max_tokens = getattr(self.llama_instance, 'n_ctx', lambda: 4096)()
        
        # Reserve tokens for tools, system prompt, and response
        tool_tokens = self._count_tool_tokens(tools)
        system_tokens = 0
        response_reserve = self.profile.parameters.max_tokens or 4096
        
        # Find system message tokens
        if messages and hasattr(messages[0], 'content'):
            if any(isinstance(msg, SystemMessage) for msg in messages[:1]):
                system_tokens = self._count_message_tokens(messages[:1])
        
        # Available tokens for conversation history
        available_tokens = max_tokens - tool_tokens - system_tokens - response_reserve - 500  # Safety buffer
        
        if available_tokens <= 0:
            self._logger.warning(
                f"Context window too small: max={max_tokens}, tools={tool_tokens}, "
                f"system={system_tokens}, response={response_reserve}"
            )
            # Keep only system message if any
            return [msg for msg in messages if isinstance(msg, SystemMessage)][:1]
        
        # Count tokens from the end (most recent messages)
        trimmed_messages = []
        current_tokens = 0
        
        # Always keep system message first
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Add system messages
        trimmed_messages.extend(system_messages)
        
        # Add other messages from most recent, checking token limits
        for message in reversed(other_messages):
            message_tokens = self._count_message_tokens([message])
            if current_tokens + message_tokens <= available_tokens:
                trimmed_messages.insert(len(system_messages), message)  # Insert after system messages
                current_tokens += message_tokens
            else:
                self._logger.info(f"Trimming message due to context limit: {message_tokens} tokens")
                break
        
        if len(trimmed_messages) < len(messages):
            self._logger.info(
                f"Trimmed {len(messages) - len(trimmed_messages)} messages to fit context window"
            )
        
        return trimmed_messages

    def _get_res(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        stream: bool = False,
    ) -> (
        llama_types.CreateChatCompletionResponse
        | Iterator[llama_types.CreateChatCompletionStreamResponse]
    ):
        """Get response from llama-cpp-python, either streaming or non-streaming."""
        assert self.llama_instance

        # Convert LangChain tools to OpenAI format for llama-cpp-python
        converted_tools = self._convert_tools_to_openai_format(tools)
        
        # Trim messages to fit context window
        trimmed_messages = self._trim_messages_to_context(messages, converted_tools)

        # Format messages for llama-cpp-python
        llama_messages = self._format_messages_for_llama(trimmed_messages)

        # Log token usage
        message_tokens = self._count_message_tokens(trimmed_messages)
        tool_tokens = self._count_tool_tokens(converted_tools)
        total_estimated = message_tokens + tool_tokens
        context_limit = getattr(self.llama_instance, 'n_ctx', lambda: 4096)()
        
        self._logger.info(
            f"Token usage: messages={message_tokens}, tools={tool_tokens}, "
            f"total_estimated={total_estimated}, context_limit={context_limit}"
        )
        
        if total_estimated > context_limit * 0.9:  # Warn at 90% capacity
            self._logger.warning(
                f"Approaching context limit: {total_estimated}/{context_limit} tokens"
            )

        response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = (
            None
        )
        grammar: Optional[llama_grammar.LlamaGrammar] = None
        if self.grammar:
            response_format = {
                "type": "json_object",
                "schema": self.grammar.model_json_schema(),
            }
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(self.grammar.model_json_schema())
            )

        return self.llama_instance.create_chat_completion(
            messages=llama_messages,
            functions=converted_tools,  # Use converted tools
            function_call="auto" if converted_tools else None,
            tools=converted_tools,  # Use converted tools
            tool_choice="auto" if converted_tools else None,
            temperature=self.profile.parameters.temperature or 0.7,
            top_p=self.profile.parameters.top_p or 0.95,
            top_k=self.profile.parameters.top_k or 40,
            min_p=self.profile.parameters.min_p or 0.05,
            typical_p=1.0,
            stream=stream,
            stop=self.profile.parameters.stop or stop or [],
            seed=self.profile.parameters.seed or llama_cpp.LLAMA_DEFAULT_SEED,
            response_format=response_format,
            max_tokens=self.profile.parameters.max_tokens or 4096,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
            # tfs_z=1.0,  # Commented out - not supported in all llama-cpp-python versions
            mirostat_mode=0,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
            model=self.model.name,
            logits_processor=None,
            grammar=grammar,
            logit_bias=None,
            logprobs=None,
            top_logprobs=None,
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
        if hasattr(self, '_bound_tools') and self._bound_tools:
            tools = list(self._bound_tools) + list(tools or [])

        try:
            res = self._get_res(
                messages=messages,
                stop=stop,
                tools=tools,
                stream=False,
            )
            response = cast(llama_types.CreateChatCompletionResponse, res)

            # Extract content and usage
            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})

            # Create usage metadata
            usage_metadata = self._calculate_usage_metadata(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )

            # Create AI message
            message = AIMessage(
                content=content,
                usage_metadata=usage_metadata,
                response_metadata={
                    "model_name": self.model.name,
                    "finish_reason": response["choices"][0].get("finish_reason"),
                },
            )

            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

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
        if hasattr(self, '_bound_tools') and self._bound_tools:
            tools = list(self._bound_tools) + list(tools or [])

        try:
            # Stream response using llama-cpp-python
            res = self._get_res(
                messages=messages,
                stop=stop,
                tools=tools,
                stream=True,
            )
            stream = cast(Iterator[llama_types.CreateChatCompletionStreamResponse], res)

            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "") or ""
                finish_reason = chunk["choices"][0].get("finish_reason")

                # Create usage metadata if available
                usage_metadata = None
                if "usage" in chunk:
                    usage = chunk["usage"]
                    usage_metadata = self._calculate_usage_metadata(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                    )

                # Create chunk message
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
