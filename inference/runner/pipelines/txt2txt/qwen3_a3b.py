"""
Optimized pipeline for Qwen3 A3B MoE model with performance enhancements.
"""

import os
import datetime
import logging
from typing import AsyncIterator, List, Optional, cast
import torch
from transformers import AutoTokenizer
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_community.tools import BaseTool
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.schema import StandardStreamEvent
from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from ..base_dual_pipeline import TextPipeline
from ..helpers import get_role, to_lc_message, extract_message_text
from ..streaming import StreamingCallbackHandler, EventStreamProcessor


logger = logging.getLogger(__name__)


class QwenGGUFPipe(TextPipeline):
    """
    Optimized pipeline for Qwen GGUF models with enhanced performance and reliability.

    Key optimizations:
    - Improved context size management
    - Better thread allocation
    - Enhanced error handling and recovery
    - Memory-efficient processing
    - Optimized stop token management
    """

    llm: ChatLlamaCpp
    tokenizer: AutoTokenizer
    _performance_metrics: dict
    _context_manager: Optional[object] = None

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize an optimized QwenGGUFPipe instance."""
        super().__init__(model, profile)

        # Initialize performance tracking
        self._performance_metrics = {
            "total_tokens": 0,
            "total_requests": 0,
            "average_response_time": 0.0,
            "error_count": 0,
        }

        # Validate model requirements
        if not (model.details and model.model and model.details.parent_model):
            raise ValueError(
                "Model definition for QwenGGUFPipe must include details for 'gguf_file' and 'parent_model'."
            )

        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initializing optimized QwenGGUFPipe for model: {model.id}")

        # Get and validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # Initialize components with optimizations
        self._initialize_tokenizer()
        self._initialize_llm(gguf_path)

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path with fallback logic."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Enhanced GGUF file validation."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf_path}"
            )

        # Test file readability
        try:
            with open(gguf_path, "rb") as f:
                f.read(8)  # Read first 8 bytes
        except Exception as e:
            raise IOError(f"Cannot read GGUF file {gguf_path}: {e}")

        self._logger.info(
            f"Using GGUF file: {gguf_path} (size: {file_size/1_000_000:.2f} MB)"
        )

    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer with error handling."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.details.parent_model,
                trust_remote_code=True,
                use_fast=True,  # Use fast tokenizer when available
            )
            self._logger.info("Fast tokenizer loaded successfully")
        except Exception as e:
            self._logger.warning(f"Fast tokenizer failed, falling back to slow: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model.details.parent_model,
                    trust_remote_code=True,
                    use_fast=False,
                )
                self._logger.info("Slow tokenizer loaded successfully")
            except Exception as e2:
                raise RuntimeError(f"Failed to load any tokenizer: {e2}")

    def _initialize_llm(self, gguf_path: str) -> None:
        """Initialize LLM with optimized settings."""
        context_size = self._get_optimal_context_size()
        optimal_threads = self._get_optimal_threads()
        stop_tokens = self._get_optimized_stop_tokens()

        try:
            self.llm = ChatLlamaCpp(
                model_path=gguf_path,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_batch=512,  # Optimized batch size for balance of speed/memory
                n_ctx=self.profile.parameters.num_ctx or context_size,
                f16_kv=True,  # Use FP16 for KV cache
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler(), StreamingCallbackHandler()]
                ),
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_parts=-1,
                seed=self.profile.parameters.seed or -1,
                logits_all=False,
                vocab_only=False,
                use_mlock=False,
                n_threads=optimal_threads,
                suffix="",
                logprobs=0,
                # Optimized generation parameters
                temperature=self.profile.parameters.temperature or 0.7,
                max_tokens=self.profile.parameters.num_predict or 4096,
                top_p=self.profile.parameters.top_p or 0.8,
                top_k=self.profile.parameters.top_k or 20,
                repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
                streaming=True,
                stop=stop_tokens,
            )

            self._logger.info(
                f"Qwen GGUF model loaded: context={context_size}, threads={optimal_threads}"
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _get_optimal_context_size(self) -> int:
        """Calculate optimal context size based on model and system capabilities."""
        # Model size to context mapping for Qwen models
        context_map = {
            "0.5b": 32768,
            "1.5b": 32768,
            "3b": 32768,
            "7b": 32768,
            "14b": 65536,
            "30b": 131072,  # Larger models can handle more context
            "72b": 131072,
        }

        # Try to determine model size from name
        model_name = self.model.name.lower()
        context_size = self.profile.parameters.num_ctx

        if not context_size:
            for size, ctx in context_map.items():
                if size in model_name:
                    context_size = ctx
                    break
            else:
                context_size = 32768  # Safe default

        # Cap based on available VRAM (rough estimate)
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 8:  # Low VRAM
                    context_size = min(context_size, 16384)
                elif vram_gb < 16:  # Medium VRAM
                    context_size = min(context_size, 32768)
        except Exception:
            pass  # Ignore VRAM check errors

        self._logger.info(f"Using context size: {context_size}")
        return context_size

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()

            # Adaptive threading based on CPU count and model size
            if "30b" in self.model.name.lower() or "72b" in self.model.name.lower():
                optimal_threads = min(
                    max(cpu_count // 2, 4), 12
                )  # More threads for large models
            else:
                optimal_threads = min(
                    max(cpu_count // 2, 2), 8
                )  # Fewer threads for small models

            self._logger.debug(
                f"Using {optimal_threads} threads (CPU count: {cpu_count})"
            )
            return optimal_threads
        except Exception:
            return 4  # Safe default

    def _get_optimized_stop_tokens(self) -> List[str]:
        """Get optimized stop tokens for Qwen models."""
        base_stops = [
            "<|im_end|>",  # Primary EOS for Qwen
            "<|endoftext|>",  # Fallback EOS
            "</tool_call>",  # Tool call boundaries
            "</tool_response>",
            "</think>",  # Thinking boundaries
        ]

        # Role-specific stops to prevent bleeding
        role_stops = [
            "<|im_start|>user",
            "<|im_start|>assistant",
            "<|im_start|>system",
            "Human:",  # Additional role prevents
            "Assistant:",
        ]

        # Combine with profile stops
        profile_stops = self.profile.parameters.stop or []
        all_stops = base_stops + role_stops + profile_stops

        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_stops))

    async def run(
        self,
        messages: List[Message],
        prompt: Optional[ChatPromptTemplate] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> AsyncIterator[ChatResponse]:
        """
        Enhanced processing with performance monitoring and error recovery.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)
        self._performance_metrics["total_requests"] += 1

        # Input validation
        if not messages:
            yield self._create_error_response("No messages provided")
            return

        if not prompt:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an honest AI assistant."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

        try:
            # Create agent with optimized settings
            agent = create_openai_tools_agent(
                llm=self.llm, tools=tools or [], prompt=prompt
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools or [],
                verbose=True,
                max_iterations=3,  # Balanced limit
                max_execution_time=120,  # 2 minute timeout
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                # callbacks=[StreamingCallbackHandler()],
                early_stopping_method="generate",  # Better resource management
            )

            # Prepare inputs
            chat_history = [to_lc_message(msg) for msg in messages[:-1]]
            processor = EventStreamProcessor(thinking_phase=True)

            # Execute with comprehensive monitoring
            token_count = 0
            chunk_count = 0

            async for event in agent_executor.astream_events(
                {
                    "input": extract_message_text(messages[-1]),
                    "chat_history": chat_history,
                },
                version="v2",
                include_types=["chat_model", "tool", "llm", "agent"],
            ):
                try:
                    for chunk in processor.stream_event(
                        cast(StandardStreamEvent, event)
                    ):
                        chunk_count += 1

                        # Count tokens for monitoring
                        if chunk.message and chunk.message.content:
                            for content in chunk.message.content:
                                if content.text:
                                    token_count += len(content.text.split())

                        yield chunk

                except Exception as chunk_error:
                    self._logger.warning(
                        f"Error processing chunk {chunk_count}: {chunk_error}"
                    )
                    # Continue processing other chunks
                    continue

        except Exception as e:
            self._logger.error(f"Error in agent streaming: {e}", exc_info=True)
            self._performance_metrics["error_count"] += 1
            yield self._create_error_response(f"Processing error: {str(e)}")

        finally:
            # Update performance metrics
            duration = (
                datetime.datetime.now(datetime.timezone.utc) - start_time
            ).total_seconds()
            self._performance_metrics["total_tokens"] += token_count

            # Update rolling average response time
            current_avg = self._performance_metrics["average_response_time"]
            request_count = self._performance_metrics["total_requests"]
            self._performance_metrics["average_response_time"] = (
                current_avg * (request_count - 1) + duration
            ) / request_count

            self._logger.info(
                f"Request completed: {duration:.2f}s, tokens: {token_count}, "
                f"avg_time: {self._performance_metrics['average_response_time']:.2f}s"
            )

    def _create_error_response(self, error_message: str) -> ChatResponse:
        """Create standardized error response."""
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"I apologize, but I encountered an error: {error_message}",
                    )
                ],
            ),
            model=self.model.model,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            finish_reason="error",
        )

    async def health_check(self) -> bool:
        """Enhanced health check with comprehensive testing."""
        try:
            test_message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text="Hello, how are you?"
                    )
                ],
                id=None,
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )

            test_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. Respond briefly."),
                    ("human", "{input}"),
                ]
            )

            # Test basic functionality
            response_count = 0
            start_time = datetime.datetime.now(datetime.timezone.utc)

            async for response in self.run([test_message], test_prompt, []):
                response_count += 1
                if response_count > 2:  # Just need a few chunks
                    break

            # Check response time
            duration = (
                datetime.datetime.now(datetime.timezone.utc) - start_time
            ).total_seconds()
            if duration > 30:  # Too slow
                self._logger.warning(f"Health check slow: {duration:.2f}s")
                return False

            return response_count > 0

        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        return self._performance_metrics.copy()

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self._performance_metrics = {
            "total_tokens": 0,
            "total_requests": 0,
            "average_response_time": 0.0,
            "error_count": 0,
        }

    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        return {
            "model_name": self.model.name,
            "model_id": self.model.id,
            "model_path": self.model.details.gguf_file if self.model.details else None,
            "context_size": getattr(self.llm, "n_ctx", None),
            "temperature": getattr(self.llm, "temperature", None),
            "max_tokens": getattr(self.llm, "max_tokens", None),
            "gpu_layers": getattr(self.llm, "n_gpu_layers", None),
            "performance_metrics": self.get_performance_metrics(),
        }

    def __del__(self) -> None:
        """Enhanced cleanup with performance logging."""
        try:
            model_name = (
                getattr(self.model, "name", "unknown")
                if hasattr(self, "model")
                else "unknown"
            )

            # Log final performance metrics
            if hasattr(self, "_performance_metrics"):
                metrics = self._performance_metrics
                self._logger.info(
                    f"QwenGGUFPipe {model_name} final metrics: "
                    f"requests={metrics['total_requests']}, "
                    f"tokens={metrics['total_tokens']}, "
                    f"avg_time={metrics['average_response_time']:.2f}s, "
                    f"errors={metrics['error_count']}"
                )

            # Clean up resources
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            if hasattr(self, "llm"):
                del self.llm

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error cleaning up QwenGGUFPipe: {e}")
