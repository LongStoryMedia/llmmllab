"""
Optimized LangGraph-based implementation for Qwen3 A3B MoE models.
Clean implementation with only essential methods for public API.
"""

import os
import datetime
import logging
import asyncio
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Type,
    cast,
    TypeVar,
    Union,
)
import hashlib
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
import torch

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
    LangGraphState,
)
from utils.message import to_lc_message
from utils.langgraph import (
    build_langgraph_state,
    coerce_to_langchain_message_dict,
    build_lc_messages,
)
from ..base import BasePipelineCore
from .context_manager import ContextManager

T = TypeVar("T", bound=Union[str, ChatResponse])

logger = logging.getLogger(__name__)


class QwenLangGraphPipe(BasePipelineCore[T]):
    """LangGraph pipeline for Qwen models with clean, optimized implementation."""

    # Only str or ChatResponse are supported return types for this pipeline
    allowed_return_types = (str, ChatResponse)

    def __init__(
        self, model: Model, profile: ModelProfile, return_type: type = Type[T]
    ):
        # Allow either str or ChatResponse, validated via BasePipelineCore
        # Note: Qwen can function as chat (ChatResponse) or text (str)
        super().__init__(model, profile, expected_return_type=return_type)
        self._return_type = return_type
        self._logger = logging.getLogger(__name__)

        if not (model.details and model.model and model.details.parent_model):
            raise ValueError(
                "Model definition requires 'gguf_file' and 'parent_model' details."
            )

        # Validate GGUF file at initialization
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # LLM and graph initialized lazily in process_messages
        self.llm = None
        self.memory = MemorySaver()
        self.graph_cache = {}
        # Serialize llama.cpp inference to avoid CUDA mem-pool races
        self._llm_lock = asyncio.Lock()
        # Track if model is in corrupted state after timeout
        self._model_corrupted = False
        # Context manager for handling context window limits
        # Initialize context manager with larger context for high-VRAM setups
        context_tokens = 32768 if "30b" in self.model.name.lower() else 16384
        self.context_manager = ContextManager(max_context_tokens=context_tokens)

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> T:
        """Process messages and return appropriate response type based on generic parameter."""

        # Check if model is corrupted and needs reset
        if self._model_corrupted:
            self._logger.warning("Model in corrupted state, attempting reset...")
            try:
                await self._reset_model()
            except Exception as e:
                self._logger.error(f"Model reset failed: {e}")
                # Return error response if reset fails
                error_msg = "Model is in unstable state and cannot be reset. Please restart the service."
                if self._return_type == str:
                    return cast(T, error_msg)
                else:
                    return cast(
                        T,
                        ChatResponse(
                            done=True,
                            message=Message(
                                role=MessageRole.ASSISTANT,
                                content=[
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=error_msg,
                                    )
                                ],
                            ),
                            created_at=datetime.datetime.now(datetime.timezone.utc),
                            finish_reason="error",
                        ),
                    )

        # Initialize LLM with tools if not already done
        if self.llm is None:
            gguf_path = self._get_gguf_path()
            self._initialize_llm(gguf_path, tools)

        try:
            # Convert to LangChain messages
            lc_messages = [to_lc_message(msg) for msg in messages]

            # Pre-emptively check and truncate messages if needed
            try:
                truncated_messages, estimated_tokens = (
                    self.context_manager.truncate_messages(lc_messages)
                )
                self._logger.info(
                    f"Using {estimated_tokens} estimated tokens for context"
                )
                lc_messages = truncated_messages
            except Exception as e:
                self._logger.warning(
                    f"Context truncation failed: {e}, proceeding with original messages"
                )

            # Create and run the graph
            graph = self.create_graph(tools)

            # Prepare initial state via builder to avoid touching generated models
            initial_state = build_langgraph_state(
                lc_messages,
                user_input="",
                current_iteration=0,
                max_iterations=10,
                error_count=0,
            )

            # Build runnable config with a stable thread id for MemorySaver
            thread_source = ""
            try:
                last = messages[-1] if messages else None
                conv_id = getattr(last, "conversation_id", None)
                thread_source = (
                    f"{conv_id}-{len(messages)}"
                    if conv_id is not None
                    else str(uuid.uuid4())
                )
            except Exception:
                thread_source = str(uuid.uuid4())

            thread_id = hashlib.md5(thread_source.encode()).hexdigest()[:16]
            config = RunnableConfig(configurable={"thread_id": f"qwen-{thread_id}"})

            # Run the graph
            result = await graph.ainvoke(initial_state, config=config)

            # Extract the final message
            if result["messages"]:
                final_message = result["messages"][-1]
                response_text = ""

                if isinstance(final_message, AIMessage):
                    if final_message.content:
                        if isinstance(final_message.content, str):
                            response_text = final_message.content
                        elif isinstance(final_message.content, list):
                            for content in final_message.content:
                                if isinstance(content, str):
                                    response_text += content + " "
                                elif isinstance(content, dict) and "text" in content:
                                    response_text += content["text"] + " "

                response_text = response_text.strip()

                # Return appropriate type based on generic parameter
                if self._return_type == str:
                    # Enforce: must be plain text for str mode
                    self.validate_return_value(response_text)
                    return cast(T, response_text)
                else:
                    # Enforce: must be ChatResponse for ChatResponse mode
                    chat = ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=response_text,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="stop",
                    )
                    self.validate_return_value(chat)
                    return cast(T, chat)

            # If no response, return error
            error_msg = "No response generated"
            if self._return_type == str:
                return cast(T, error_msg)
            else:
                return cast(
                    T,
                    ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=error_msg,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="error",
                    ),
                )

        except Exception as e:
            self._logger.error(f"Error in process_messages: {e}")
            error_msg = f"Error processing request: {str(e)}"

            if self._return_type == str:
                return cast(T, error_msg)
            else:
                return cast(
                    T,
                    ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=error_msg,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="error",
                    ),
                )

    async def prompt(self, text: str | List[str]) -> T:
        """Process a single message and return appropriate response type."""
        if isinstance(text, list):
            text = " ".join(text)

        # Create a simple user message
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=text,
                )
            ],
        )

        return await self.process_messages([message])

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState]:
        """Create LangGraph with simplified caching."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Ensure LLM initialized here as well
        if self.llm is None:
            gguf_path = self._get_gguf_path()
            self._initialize_llm(gguf_path, tools)
        else:
            # If LLM exists and tools provided, ensure they're bound
            try:
                if tools:
                    binder = getattr(self.llm, "bind_tools", None)
                    if callable(binder):
                        self.llm = binder(tools)
            except Exception:
                pass

        # Build graph
        workflow = StateGraph(LangGraphState)
        workflow.add_node("agent", self._agent_node)

        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges(
                "agent", tools_condition, {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        workflow.add_edge(START, "agent")

        compiled_graph = workflow.compile(checkpointer=self.memory)
        self.graph_cache[tool_signature] = compiled_graph
        return compiled_graph

    async def _agent_node(
        self, state: LangGraphState, config: RunnableConfig | None = None
    ) -> Dict[str, Any]:
        """Simplified agent node with essential error handling."""
        # Check limits
        if state.current_iteration >= state.max_iterations:
            return {
                "messages": [
                    coerce_to_langchain_message_dict(
                        AIMessage(content="I've reached the iteration limit.")
                    )
                ],
                "current_iteration": state.current_iteration + 1,
            }

        if state.error_count >= 3:
            return {
                "messages": [
                    coerce_to_langchain_message_dict(
                        AIMessage(content="I'm experiencing technical difficulties.")
                    )
                ],
                "error_count": state.error_count + 1,
            }

        try:
            # Convert schema/dict messages back to LangChain BaseMessage objects
            messages = build_lc_messages(state.messages)
            assert self.llm is not None, "LLM not initialized"

            # Forward runnable config so astream_events can capture callbacks
            # NOTE: llama.cpp contexts are not thread-safe; guard with a lock
            # CRITICAL: Add timeout to prevent infinite loops in LLM reasoning
            async with self._llm_lock:
                try:
                    # Extended timeout for complex reasoning - deep research may take longer
                    # Check if this is a deep research query that needs more time
                    is_deep_research = any(
                        keyword in str(messages).lower()
                        for keyword in [
                            "research",
                            "analyze",
                            "comprehensive",
                            "detailed",
                            "thorough",
                        ]
                    )
                    timeout_seconds = 120.0 if is_deep_research else 60.0

                    # Create the coroutine for LLM invocation
                    coroutine = (
                        self.llm.ainvoke(messages, config=config)  # type: ignore
                        if config is not None
                        else self.llm.ainvoke(messages)  # type: ignore
                    )

                    response = await asyncio.wait_for(
                        coroutine,
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    self._logger.error(
                        f"LLM invocation timed out after {timeout_seconds} seconds - likely infinite loop"
                    )
                    # Mark model as corrupted so it gets reset on next request
                    self._model_corrupted = True
                    raise RuntimeError(
                        "LLM got stuck in reasoning loop - timeout triggered"
                    )

            return {
                "messages": [coerce_to_langchain_message_dict(response)],
                "current_iteration": state.current_iteration + 1,
            }
        except Exception as e:
            error_msg = str(e)
            self._logger.error(f"Agent node error: {error_msg}")

            # Handle specific error types with better recovery
            if "exceed context window" in error_msg or "tokens" in error_msg:
                # Try to recover by truncating context more aggressively
                try:
                    messages = build_lc_messages(state.messages)
                    recovered_messages = self.context_manager.handle_context_overflow(
                        messages, error_msg
                    )

                    # Try again with truncated context (with timeout protection)
                    async with self._llm_lock:
                        try:
                            response = await asyncio.wait_for(
                                (
                                    self.llm.ainvoke(recovered_messages, config=config)  # type: ignore
                                    if config is not None
                                    else self.llm.ainvoke(recovered_messages)
                                ),  # type: ignore
                                timeout=60.0,  # Extended timeout for recovery
                            )
                        except asyncio.TimeoutError:
                            self._logger.error(
                                "Context recovery also timed out - model unstable"
                            )
                            # Mark model as corrupted
                            self._model_corrupted = True
                            raise RuntimeError(
                                "Context recovery timed out - model stuck in loops"
                            )

                    self._logger.info("Successfully recovered from context overflow")
                    return {
                        "messages": [coerce_to_langchain_message_dict(response)],
                        "current_iteration": state.current_iteration + 1,
                    }
                except Exception as recovery_error:
                    self._logger.error(
                        f"Context recovery also failed: {recovery_error}"
                    )

                # If recovery fails, provide helpful message
                context_error = (
                    "The conversation has become too long for the current context window. "
                    "Please start a new conversation or summarize the key points you'd like to continue with."
                )
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=context_error)
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }
            elif "CUDA" in error_msg or "memory" in error_msg.lower():
                # Memory/CUDA errors
                memory_error = (
                    "I'm experiencing memory constraints. Please try a shorter prompt "
                    "or restart the conversation."
                )
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=memory_error)
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }
            elif (
                "timeout" in error_msg.lower() or "stuck in reasoning loop" in error_msg
            ):
                # Timeout/infinite loop errors
                timeout_error = (
                    "The model got stuck in a reasoning loop and was stopped for safety. "
                    "Please try rephrasing your request or starting a new conversation."
                )
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=timeout_error)
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }
            else:
                # Generic error
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=f"Error: {error_msg[:200]}...")
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }

    def _get_gguf_path(self) -> str:
        """Get GGUF file path."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate GGUF file exists and is readable."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(f"GGUF file too small ({file_size} bytes): {gguf_path}")

        try:
            with open(gguf_path, "rb") as f:
                f.read(8)  # Test readability
        except Exception as e:
            raise IOError(f"Cannot read GGUF file {gguf_path}: {e}") from e

    def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Initialize LLM with optimized settings and optional tools."""
        # Aggressive CUDA graphs disabling: set ALL possible environment variables
        # Some builds read different variants, so set them all defensively
        os.environ["LLAMA_CUDA_USE_GRAPHS"] = "0"
        os.environ["GGML_CUDA_USE_GRAPHS"] = "0"
        os.environ["CUDA_USE_GRAPHS"] = "0"
        os.environ["LLAMA_GRAPH"] = "0"
        os.environ["GGML_GRAPH"] = "0"

        # Also set compile-time graph disabling flags
        os.environ["LLAMA_CUBLAS"] = (
            "1"  # Force basic CUBLAS instead of graph optimization
        )
        os.environ["GGML_CUDA_FORCE_CUBLAS"] = "1"

        # Calculate context size based on model and practical limits
        model_name = self.model.name.lower()

        # Base context sizes - use model's full training capacity when possible
        if "30b" in model_name:
            # 30B models support 40K context, use substantial portion with high VRAM
            base_context = 32768  # Significantly increased for 30B models
        elif "72b" in model_name:
            base_context = 8192  # Large models need conservative settings due to memory
        elif "7b" in model_name or "14b" in model_name:
            base_context = 16384
        else:
            base_context = 16384  # Default to higher context

        # Check available VRAM and adjust accordingly
        try:
            if torch.cuda.is_available():
                # Get total VRAM across all devices
                total_vram_gb = 0
                for i in range(torch.cuda.device_count()):
                    total_vram_gb += torch.cuda.get_device_properties(
                        i
                    ).total_memory / (1024**3)

                # Adjust context based on total VRAM
                if total_vram_gb >= 40:  # High VRAM setup - use model's full capacity
                    if "30b" in model_name:
                        base_context = 40960  # Use model's full 40K training context
                    # Keep larger contexts for other models too
                elif total_vram_gb >= 20:  # Medium VRAM setup
                    base_context = min(base_context, 16384)
                else:  # Low VRAM setup
                    base_context = min(base_context, 8192)

                self._logger.info(
                    f"Detected {total_vram_gb:.1f}GB total VRAM, using context size: {base_context}"
                )
        except Exception as e:
            self._logger.warning(f"Could not detect VRAM: {e}, using default context")

        # Allow profile override but prioritize our dynamic calculation for 30B models
        # Only use profile num_ctx if it's larger than our calculated base_context
        profile_ctx = self.profile.parameters.num_ctx or 0
        final_ctx = max(
            base_context, profile_ctx, 8192
        )  # Use the largest reasonable value

        self._logger.info(f"Initializing Qwen LLM with context size: {final_ctx}")

        # Initialize ChatLlamaCpp with optimized parameters
        self.llm = ChatLlamaCpp(
            model_path=gguf_path,
            n_gpu_layers=-1,  # Offload all layers to GPU
            # Very conservative batch size to minimize memory spikes
            n_batch=64,  # Reduced from 128 for extra stability
            f16_kv=True,
            verbose=os.environ.get("LOG_LEVEL", "warning") == "debug",
            n_parts=-1,
            streaming=True,
            n_ctx=final_ctx,
            # Add memory management parameters
            use_mmap=True,  # Enable memory mapping for efficiency
            use_mlock=False,  # Disable mlock to avoid memory pressure
            # Model parameters
            seed=self.profile.parameters.seed or -1,
            temperature=self.profile.parameters.temperature or 0.7,
            max_tokens=self.profile.parameters.max_tokens or 4096,
            top_p=self.profile.parameters.top_p or 0.8,
            top_k=self.profile.parameters.top_k or 20,
            repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
            # Enhanced stop sequences to prevent infinite reasoning loops
            stop=self.profile.parameters.stop
            or [
                "<|im_end|>",
                "<|endoftext|>",
                "<|end|>",
                "\n\n\n",  # Stop on excessive newlines (repetition indicator)
            ],
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        # Update context manager with actual context size
        self.context_manager.max_context_tokens = final_ctx

        if tools:
            try:
                self.llm = self.llm.bind_tools(tools)
            except Exception:
                pass

    async def _reset_model(self) -> None:
        """Reset the model after timeout/corruption to restore functionality."""
        self._logger.info("Resetting corrupted model state...")

        # Clear LLM instance to force reinitialization
        old_llm = self.llm
        self.llm = None

        # Clear graph cache to force rebuild
        self.graph_cache.clear()

        # Try to clean up old LLM resources
        if old_llm is not None:
            try:
                # Try to access any cleanup methods that might exist
                if hasattr(old_llm, "client") and hasattr(old_llm.client, "close"):
                    old_llm.client.close()
                elif hasattr(old_llm, "_client") and hasattr(old_llm._client, "close"):
                    old_llm._client.close()
                # Allow some time for cleanup
                await asyncio.sleep(0.5)
            except Exception as e:
                self._logger.warning(f"Error during LLM cleanup: {e}")

        # Reset corruption flag
        self._model_corrupted = False
        self._logger.info("Model reset completed")

    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self.llm is not None:
            try:
                # Try to clean up LLM resources
                if hasattr(self.llm, "client") and hasattr(self.llm.client, "close"):
                    self.llm.client.close()
                elif hasattr(self.llm, "_client") and hasattr(
                    self.llm._client, "close"
                ):
                    self.llm._client.close()
            except Exception as e:
                self._logger.warning(f"Error during cleanup: {e}")
            finally:
                self.llm = None

        # Clear caches
        self.graph_cache.clear()
        self._model_corrupted = False
