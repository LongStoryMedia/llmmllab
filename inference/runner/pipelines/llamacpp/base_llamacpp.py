"""
BaseLlamaCppPipeline as a custom BaseChatModel implementation.
Uses llama-cpp-python directly instead of LangChain's ChatLlamaCpp wrapper.
"""

import os
import logging
import multiprocessing

from typing import Optional, List, Any, Dict, Iterator

from pydantic import BaseModel
import llama_cpp

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
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
from utils.grammar_generator import (
    get_grammar_for_model,
)
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

    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model
        self.profile_config = profile
        self.llama_instance: Optional[llama_cpp.Llama] = None  # Optional[LlamaType]
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )
        self._grammar_model_class: Optional[BaseModel] = None  # Optional[type]

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-cpp-custom"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_config.name,
            "model_path": self._get_gguf_path(),
            "n_ctx": self.profile_config.parameters.num_ctx or 4096,
            "temperature": self.profile_config.parameters.temperature or 0.7,
        }

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        return (
            self.model_config.details.gguf_file
            if hasattr(self.model_config.details, "gguf_file")
            and self.model_config.details.gguf_file
            else self.model_config.model
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
        model_name = self.model_config.name.lower()

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

    def _process_grammar_input(self, grammar: type[BaseModel]) -> Optional[str]:
        """Process grammar input and return GBNF string if possible."""
        try:
            self._grammar_model_class = grammar
            return get_grammar_for_model(grammar)
        except Exception as e:
            self._logger.warning(f"Grammar processing failed: {e}")

        return None

    def _initialize_llama_with_fallback(
        self,
        gguf_path: str,
    ) -> llama_cpp.Llama:
        """Initialize Llama instance with fallback strategies."""
        if llama_cpp.Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        # Get base parameters
        requested_ctx = self.profile_config.parameters.num_ctx or 4096
        requested_batch = self.profile_config.parameters.batch_size or 512

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
            self.profile_config.gpu_config is not None
            and self.profile_config.gpu_config.gpu_layers is not None
        ):
            explicit_gpu_layers = self.profile_config.gpu_config.gpu_layers

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
                    llama_instance = llama_cpp.Llama(
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
                        seed=self.profile_config.parameters.seed
                        or llama_cpp.LLAMA_DEFAULT_SEED,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        n_ubatch=512,
                        n_threads=self._get_optimal_threads(),
                        temperature=self.profile_config.parameters.temperature or 0.7,
                        top_p=self.profile_config.parameters.top_p or 0.8,
                        top_k=self.profile_config.parameters.top_k or 20,
                        repeat_penalty=self.profile_config.parameters.repeat_penalty
                        or 1.05,
                        f16_kv=True,
                        verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                        flash_attn=getattr(
                            self.profile_config.parameters, "flash_attention", True
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

                    self._logger.info(
                        f"Initialized Llama model: ctx={n_ctx}, batch={n_batch}, gpu_layers={n_gpu_layers}"
                    )

                    self.llama_instance = llama_instance

                    return llama_instance

                except Exception as e:
                    self._logger.warning(
                        f"Failed to initialize with ctx={n_ctx}, batch={n_batch}, "
                        f"gpu_layers={n_gpu_layers}: {e}"
                    )
                    continue

        raise RuntimeError(
            f"Failed to initialize {self.model_config.name} with any configuration"
        )

    def _format_messages_for_llama(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from messages."""
        tools = kwargs.get("tools")
        grammar: type[BaseModel] = kwargs.get("grammar")
        llama = self.llama_instance or self._initialize_llama_with_fallback(
            self._get_gguf_path()
        )
        grammar_string = self._process_grammar_input(grammar)

        # Format messages for llama-cpp-python
        llama_messages = self._format_messages_for_llama(messages)

        try:
            response = llama.create_chat_completion(
                messages=llama_messages,
                functions=tools,
                function_call="auto",
                tools=tools,
                tool_choice="auto",
                temperature=self.profile_config.parameters.temperature or 0.7,
                top_p=self.profile_config.parameters.top_p or 0.95,
                top_k=self.profile_config.parameters.top_k or 40,
                min_p=self.profile_config.parameters.min_p or 0.05,
                typical_p=self.profile_config.parameters.typical_p or 1.0,
                stream=False,
                stop=self.profile_config.parameters.stop or stop or [],
                seed=self.profile_config.parameters.seed
                or llama_cpp.LLAMA_DEFAULT_SEED,
                response_format=(
                    {"type": "json", "schema": self._grammar_model_class.model_dump()}
                    if self._grammar_model_class
                    else None
                ),
                max_tokens=self.profile_config.parameters.max_tokens or 4096,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repeat_penalty=self.profile_config.parameters.repeat_penalty or 1.05,
                tfs_z=1.0,
                mirostat_mode=0,
                mirostat_tau=5.0,
                mirostat_eta=0.1,
                model=self.model_config.name,
                logits_processor=None,
                grammar=grammar_string,
                logit_bias=None,
                logprobs=None,
                top_logprobs=None,
            )

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
                    "model_name": self.model_config.name,
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
        tools = kwargs.get("tools")
        grammar = kwargs.get("grammar")
        llama = self.llama_instance or self._initialize_llama_with_fallback(
            self._get_gguf_path()
        )
        grammar_string = self._process_grammar_input(grammar)

        # Format messages for llama-cpp-python
        llama_messages = self._format_messages_for_llama(messages)
        try:
            # Stream response using llama-cpp-python
            stream = llama.create_chat_completion(
                messages=llama_messages,
                functions=tools,
                function_call="auto",
                tools=tools,
                tool_choice="auto",
                temperature=self.profile_config.parameters.temperature or 0.7,
                top_p=self.profile_config.parameters.top_p or 0.95,
                top_k=self.profile_config.parameters.top_k or 40,
                min_p=self.profile_config.parameters.min_p or 0.05,
                typical_p=self.profile_config.parameters.typical_p or 1.0,
                stream=True,
                stop=self.profile_config.parameters.stop or stop or [],
                seed=self.profile_config.parameters.seed
                or llama_cpp.LLAMA_DEFAULT_SEED,
                response_format=(
                    {"type": "json", "schema": self._grammar_model_class.model_dump()}
                    if self._grammar_model_class
                    else None
                ),
                max_tokens=self.profile_config.parameters.max_tokens or 4096,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repeat_penalty=self.profile_config.parameters.repeat_penalty or 1.05,
                tfs_z=1.0,
                mirostat_mode=0,
                mirostat_tau=5.0,
                mirostat_eta=0.1,
                model=self.model_config.name,
                logits_processor=None,
                grammar=grammar_string,
                logit_bias=None,
                logprobs=None,
                top_logprobs=None,
            )

            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
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
                            "model_name": self.model_config.name,
                            "finish_reason": finish_reason,
                        }
                        if finish_reason
                        else {}
                    ),
                )

                # Create and yield generation chunk
                generation_chunk = ChatGenerationChunk(message=chunk_message)

                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=generation_chunk)

                yield generation_chunk

        except Exception as e:
            self._logger.error(f"Streaming failed: {e}")
            raise

    # Support for tool binding (optional)
    def bind_tools(self, tools: List[BaseTool], **kwargs) -> "BaseLlamaCppPipeline":
        """Bind tools to this model instance."""
        # For now, return a new instance that remembers the tools
        # Tools will be used in prompt formatting
        new_instance = self.__class__(
            model=self.model_config, profile=self.profile_config, **kwargs
        )
        new_instance.llama_instance = self.llama_instance
        new_instance._bound_tools = (  # pylint: disable=protected-access,attribute-defined-outside-init
            tools
        )
        return new_instance

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


BaseLlamaCppPipeline

__all__ = ["BaseLlamaCppPipeline"]
