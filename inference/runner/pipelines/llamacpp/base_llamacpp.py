"""
BaseLlamaCppPipeline as a custom BaseChatModel implementation.
Uses llama-cpp-python directly instead of LangChain's ChatLlamaCpp wrapper.
"""

import os
import logging
import multiprocessing
from typing import Optional, List, Any, Dict, Iterator, AsyncIterator, Union

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
from pydantic import Field

from models import Model, ModelProfile
from runner.pipelines.base import GrammarInput
from .utils import calculate_optimal_gpu_layers

# Import llama-cpp-python directly
try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None
    LlamaGrammar = None


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

    model_config: Model = Field(description="Model configuration")
    profile_config: ModelProfile = Field(description="Model profile configuration")
    
    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model
        self.profile_config = profile
        self.llama_instance = None  # Optional[LlamaType]
        self._logger = logging.getLogger(self.__class__.__name__)
        self._grammar_model_class = None  # Optional[type]

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
            if hasattr(self.model_config.details, "gguf_file") and self.model_config.details.gguf_file
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

    def _process_grammar_input(self, grammar: Optional[GrammarInput]) -> Optional[str]:
        """Process grammar input and return GBNF string if possible."""
        if grammar is None:
            return None
        
        try:
            from utils.grammar_generator import (
                get_grammar_for_model,
                load_grammar_from_file,
            )
            from pydantic import BaseModel
            from pathlib import Path

            if isinstance(grammar, str):
                return grammar
            if isinstance(grammar, Path):
                return load_grammar_from_file(grammar)
            if isinstance(grammar, type) and issubclass(grammar, BaseModel):
                self._grammar_model_class = grammar
                return get_grammar_for_model(grammar)
        except Exception as e:
            self._logger.warning(f"Grammar processing failed: {e}")
        
        return None

    def _initialize_llama_with_fallback(
        self,
        gguf_path: str,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
    ) -> Any:
        """Initialize Llama instance with fallback strategies."""
        if Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        # Get base parameters
        requested_ctx = self.profile_config.parameters.num_ctx or 4096
        requested_batch = self.profile_config.parameters.batch_size or 512
        
        # Model size and GPU layer candidates
        model_size_category = self._get_model_size_category()
        context_candidates = self._build_context_candidates(requested_ctx)
        
        # GPU layers strategy
        explicit_gpu_layers = None
        if (
            self.profile_config.gpu_config is not None
            and self.profile_config.gpu_config.gpu_layers is not None
        ):
            explicit_gpu_layers = self.profile_config.gpu_config.gpu_layers

        # Grammar preparation
        grammar_string = self._process_grammar_input(grammar)

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
                    # Initialize Llama directly
                    llama_params = {
                        "model_path": gguf_path,
                        "n_gpu_layers": n_gpu_layers,
                        "n_ctx": n_ctx,
                        "n_batch": n_batch,
                        "n_threads": self._get_optimal_threads(),
                        "seed": self.profile_config.parameters.seed or -1,
                        "temperature": self.profile_config.parameters.temperature or 0.7,
                        "top_p": self.profile_config.parameters.top_p or 0.8,
                        "top_k": self.profile_config.parameters.top_k or 20,
                        "repeat_penalty": self.profile_config.parameters.repeat_penalty or 1.05,
                        "use_mmap": True,
                        "use_mlock": False,
                        "f16_kv": True,
                        "verbose": os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                        "flash_attn": getattr(self.profile_config.parameters, "flash_attention", True),
                        "logits_all": False,  # Only enable if needed for specific features
                        "embedding": False,   # This is for chat, not embeddings
                        "chat_format": "chatml",  # Default chat format, can be overridden
                    }

                    llama_instance = Llama(**llama_params)
                    
                    self._logger.info(
                        f"Initialized Llama model: ctx={n_ctx}, batch={n_batch}, "
                        f"gpu_layers={n_gpu_layers}, threads={llama_params['n_threads']}"
                    )
                    
                    return llama_instance

                except Exception as e:
                    self._logger.warning(
                        f"Failed to initialize with ctx={n_ctx}, batch={n_batch}, "
                        f"gpu_layers={n_gpu_layers}: {e}"
                    )
                    continue

        raise RuntimeError(f"Failed to initialize {self.model_config.name} with any configuration")

    def _get_llama_instance(
        self, 
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None
    ) -> Any:
        """Get or create Llama instance."""
        if self.llama_instance is None:
            gguf_path = self._get_gguf_path()
            self.llama_instance = self._initialize_llama_with_fallback(gguf_path, tools, grammar)
        return self.llama_instance

    def _format_messages_for_llama(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
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
                llama_messages.append({
                    "role": "user", 
                    "content": f"Tool result: {message.content}"
                })
            else:
                # Fallback: treat as user message
                llama_messages.append({"role": "user", "content": str(message.content)})
        
        return llama_messages

    def _calculate_usage_metadata(
        self, 
        prompt_tokens: int, 
        completion_tokens: int
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
        llama = self._get_llama_instance(
            tools=kwargs.get("tools"),
            grammar=kwargs.get("grammar")
        )
        
        # Format messages for llama-cpp-python
        llama_messages = self._format_messages_for_llama(messages)
        
        # Prepare generation parameters
        generation_params = {
            "messages": llama_messages,
            "max_tokens": self.profile_config.parameters.max_tokens or 4096,
            "temperature": self.profile_config.parameters.temperature or 0.7,
            "top_p": self.profile_config.parameters.top_p or 0.8,
            "top_k": self.profile_config.parameters.top_k or 20,
            "repeat_penalty": self.profile_config.parameters.repeat_penalty or 1.05,
            "stop": stop or self.profile_config.parameters.stop or [],
            "stream": False,
        }

        try:
            # Generate response using llama-cpp-python
            response = llama.create_chat_completion(**generation_params)
            
            # Extract content and usage
            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            
            # Create usage metadata
            usage_metadata = self._calculate_usage_metadata(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0)
            )
            
            # Create AI message
            message = AIMessage(
                content=content,
                usage_metadata=usage_metadata,
                response_metadata={
                    "model_name": self.model_config.name,
                    "finish_reason": response["choices"][0].get("finish_reason"),
                }
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
        llama = self._get_llama_instance(
            tools=kwargs.get("tools"),
            grammar=kwargs.get("grammar")
        )
        
        # Format messages for llama-cpp-python
        llama_messages = self._format_messages_for_llama(messages)
        
        # Prepare streaming parameters
        generation_params = {
            "messages": llama_messages,
            "max_tokens": self.profile_config.parameters.max_tokens or 4096,
            "temperature": self.profile_config.parameters.temperature or 0.7,
            "top_p": self.profile_config.parameters.top_p or 0.8,
            "top_k": self.profile_config.parameters.top_k or 20,
            "repeat_penalty": self.profile_config.parameters.repeat_penalty or 1.05,
            "stop": stop or self.profile_config.parameters.stop or [],
            "stream": True,
        }

        try:
            # Stream response using llama-cpp-python
            stream = llama.create_chat_completion(**generation_params)
            
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
                        completion_tokens=usage.get("completion_tokens", 0)
                    )
                
                # Create chunk message
                chunk_message = AIMessageChunk(
                    content=content,
                    usage_metadata=usage_metadata,
                    response_metadata={
                        "model_name": self.model_config.name,
                        "finish_reason": finish_reason,
                    } if finish_reason else {}
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
            model=self.model_config,
            profile=self.profile_config,
            **kwargs
        )
        new_instance.llama_instance = self.llama_instance
        new_instance._bound_tools = tools
        return new_instance

    def close(self):
        """Clean up resources."""
        if self.llama_instance:
            try:
                self.llama_instance.close()
            except:
                pass
            self.llama_instance = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Backwards compatibility alias
SimpleLlamaCppPipeline = BaseLlamaCppPipeline

__all__ = ["BaseLlamaCppPipeline", "SimpleLlamaCppPipeline"]