"""
Base pipeline class with shared timeout protection, circuit breaker, and context management.
"""

import asyncio
import hashlib
import uuid
import logging
import os
from typing import Any, Dict, List, Optional, TypeVar, cast
from abc import ABC, abstractmethod
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    ChatResponse,
    ModelProfile,
    Model,
)
from utils.langgraph import LangGraphState, build_langgraph_state, build_lc_messages
from utils.message import to_lc_message
from utils.langgraph import coerce_to_langchain_message_dict
from runner.pipelines.base import BasePipelineCore, PipeType

T = TypeVar("T")


class CircuitBreakerConfig:
    """Configuration for circuit breaker and timeout protection."""

    def __init__(
        self,
        base_timeout: float = 60.0,
        deep_research_timeout: float = 120.0,
        max_retries: int = 2,
        cooldown_period: float = 30.0,
        perplexity_threshold: float = 15.0,
    ):
        self.base_timeout = base_timeout
        self.deep_research_timeout = deep_research_timeout
        self.max_retries = max_retries
        self.cooldown_period = cooldown_period
        self.perplexity_threshold = perplexity_threshold


class ModelState:
    """Tracks the state of a model pipeline."""

    def __init__(self):
        self.is_corrupted = False
        self.corruption_time: Optional[datetime] = None
        self.retry_count = 0
        self.last_error: Optional[str] = None


class BaseLangGraphPipeline(BasePipelineCore[PipeType], ABC):
    """
    Base class for LangGraph-based pipelines with built-in timeout protection,
    circuit breaker functionality, and proper resource management.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        super().__init__(model, profile, expected_return_type)
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.model_state = ModelState()
        self.memory = MemorySaver()
        self.graph_cache: Dict[int, CompiledStateGraph] = {}
        self._llm_lock = asyncio.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

        # Abstract attributes that subclasses must implement
        self.llm: Optional[BaseChatModel] = None

    @property
    @abstractmethod
    def total_model_layers(self) -> int:
        """Total number of layers in the model. Must be implemented by subclasses."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_size_category(self) -> str:
        """Return model size category ('small', 'medium', 'large', 'xlarge')."""
        raise NotImplementedError

    @abstractmethod
    def _create_llm_instance(
        self,
        gguf_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        tools: Optional[List[BaseTool]] = None,
    ) -> Any:
        """Create an instance of the LLM. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _get_gguf_path(self) -> str:
        """Get the GGUF file path for the model. Must be implemented by subclasses."""
        raise NotImplementedError

    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """
        Initialize the LLM with a sophisticated OOM fallback strategy.
        1. Try optimal settings.
        2. If OOM, gradually reduce GPU layers to a minimum threshold.
        3. If still OOM, reduce n_ctx and repeat GPU layer reduction.
        """
        self._logger.info(f"Initializing LLM from {gguf_path}")

        if os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in (
            "1",
            "true",
            "yes",
        ):

            class _DummyLLM(BaseChatModel):
                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    # This is a dummy implementation for sync generation
                    return AIMessage(content="Dummy response for dev/test.")

                async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                    # This is a dummy implementation for async generation
                    return AIMessage(content="Dummy response for dev/test.")

                def _llm_type(self) -> str:
                    return "dummy-llm"

                def bind_tools(self, tools):
                    return self

                async def astream(self, messages, **kwargs):
                    yield AIMessageChunk(content="Dummy response for dev/test.")

                async def ainvoke(self, _messages, **_):
                    return AIMessage(content="Dummy response for dev/test.")

            self.llm = _DummyLLM()
            if tools:
                self.llm = cast(BaseChatModel, self.llm.bind_tools(tools))
            return

        from runner.pipeline_factory import pipeline_factory
        from utils.hardware_manager import hardware_manager

        min_gpu_layers = (self.total_model_layers * 2) // 3

        requested_ctx = (
            self.profile.parameters.num_ctx or 131072
        )  # Default to a reasonable context
        max_ctx = self.profile.parameters.num_ctx or 1048576

        context_candidates = sorted(
            list(
                set(
                    [
                        min(requested_ctx, max_ctx),
                        524288,
                        262144,
                        131072,
                        65536,
                        32768,
                        16384,
                    ]
                )
            ),
            reverse=True,
        )
        unique_context_candidates = [
            x for x in context_candidates if x <= requested_ctx
        ]

        for n_ctx in unique_context_candidates:
            optimal_gpu_layers = self._calculate_optimal_gpu_layers(
                n_ctx, self.model_size_category
            )

            gpu_layer_steps = sorted(
                list(
                    set(
                        [
                            optimal_gpu_layers,
                            (optimal_gpu_layers + min_gpu_layers) // 2,
                            min_gpu_layers + (optimal_gpu_layers - min_gpu_layers) // 4,
                            min_gpu_layers,
                        ]
                    )
                ),
                reverse=True,
            )

            for n_gpu_layers in gpu_layer_steps:
                if n_gpu_layers < min_gpu_layers:
                    continue

                try:
                    self._logger.info(
                        f"Attempting to load model with n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}"
                    )

                    self.llm = self._create_llm_instance(
                        gguf_path=gguf_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        tools=tools,
                    )

                    self._logger.info(
                        f"Successfully loaded model with n_ctx={n_ctx} and n_gpu_layers={n_gpu_layers}"
                    )
                    return

                except Exception as e:
                    self._logger.warning(
                        f"Failed to load model with n_ctx={n_ctx} and n_gpu_layers={n_gpu_layers}: {e}. "
                        "Freeing memory and retrying."
                    )
                    self.llm = None
                    pipeline_factory.clear_unused_pipelines()
                    hardware_manager.clear_memory(aggressive=True)
                    await asyncio.sleep(1)

        error_msg = (
            "Failed to load model even with minimal settings after all fallbacks."
        )
        self._logger.error(error_msg)
        self._mark_model_corrupted(error_msg)
        raise RuntimeError(error_msg)

    @abstractmethod
    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt with anti-loop instructions. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _create_system_prompt")

    def _calculate_optimal_gpu_layers(
        self, n_ctx: int, model_size_category: str = "medium"
    ) -> int:
        """
        Dynamically calculate optimal GPU layers based on context size and model category.

        Args:
            n_ctx: Context size in tokens
            model_size_category: Model size category ("small", "medium", "large", "xlarge")
                - "small": ~3B-7B models (e.g., 3B embeddings)
                - "medium": ~13B-20B models (e.g., OpenAI GPT OSS 20B)
                - "large": ~30B-40B models (e.g., Qwen3-Coder 30B)
                - "xlarge": ~70B+ models

        Strategy:
        - Higher context = fewer GPU layers (more KV cache in GPU VRAM)
        - Lower context = more GPU layers (model weights in GPU VRAM)
        - Larger models need more conservative GPU allocation
        """
        # Define base allocations per model size (AGGRESSIVE for research/testing with 48GB VRAM)
        base_allocations = {
            "small": {"max": 99, "high": 95, "medium": 90, "low": 85, "min": 80},
            "medium": {"max": 95, "high": 90, "medium": 85, "low": 80, "min": 75},
            "large": {
                "max": 90,
                "high": 85,
                "medium": 80,
                "low": 75,
                "min": 70,
            },  # Much more aggressive for 30B
            "xlarge": {"max": 85, "high": 80, "medium": 75, "low": 70, "min": 65},
        }

        # Get allocation thresholds for the model category
        alloc = base_allocations.get(model_size_category, base_allocations["medium"])

        # Determine GPU layers based on context size (AGGRESSIVE allocation)
        if n_ctx <= 4096:  # 4K context or less
            return alloc["max"]  # Maximum layers on GPU
        elif n_ctx <= 8192:  # 8K context
            return alloc["high"]  # High allocation
        elif n_ctx <= 16384:  # 16K context
            return alloc["medium"]  # Still aggressive
        elif n_ctx <= 32768:  # 32K context
            return alloc["low"]  # Reasonable allocation
        elif n_ctx <= 65536:  # 64K context
            return alloc["min"]  # Still good allocation
        elif n_ctx <= 131072:  # 128K context (model's trained context)
            return max(
                20, alloc["min"] - 5
            )  # Slightly more conservative but still reasonable
        elif n_ctx <= 262144:  # 256K context
            return max(18, alloc["min"] - 7)  # More conservative for larger context
        elif n_ctx <= 524288:  # 512K context
            return max(15, alloc["min"] - 10)  # Very conservative for large context
        else:  # > 512K context (1M tokens)
            return max(12, alloc["min"] - 13)  # Minimal layers for extreme context

    def _should_use_extended_timeout(self, messages: List[Message]) -> bool:
        """
        Determine if this request should use extended timeout.
        Override in subclasses for custom logic.
        """
        # Check if any message mentions research, web search, or complex analysis
        extended_keywords = [
            "research",
            "web search",
            "analyze",
            "investigate",
            "detailed analysis",
            "comprehensive",
            "deep dive",
            "arduino",
            "code",
            "BOM",
            "bill of materials",
        ]

        for message in messages:
            if message.content:
                for content in message.content:
                    if content.type == MessageContentType.TEXT and content.text:
                        text_lower = content.text.lower()
                        if any(keyword in text_lower for keyword in extended_keywords):
                            return True
        return False

    def _get_timeout_for_request(self, messages: List[Message]) -> float:
        """Get appropriate timeout based on request complexity."""
        if self._should_use_extended_timeout(messages):
            return self.circuit_config.deep_research_timeout
        return self.circuit_config.base_timeout

    async def _check_model_health(self) -> bool:
        """Check if model is in a healthy state for processing."""
        if not self.model_state.is_corrupted:
            return True

        # Check if enough time has passed for cooldown
        if self.model_state.corruption_time:
            time_since_corruption = (
                datetime.now() - self.model_state.corruption_time
            ).total_seconds()
            if time_since_corruption >= self.circuit_config.cooldown_period:
                self._logger.info("Cooldown period elapsed, attempting model reset")
                return await self._reset_model()

        return False

    async def _reset_model(self) -> bool:
        """Reset the model after corruption to restore functionality."""
        try:
            self._logger.info("Resetting corrupted model state...")

            # Clear LLM instance to force reinitialization
            old_llm = self.llm
            self.llm = None

            # Clear graph cache to force rebuild
            self.graph_cache.clear()

            # Try to clean up old LLM resources
            if old_llm is not None:
                try:
                    # Allow some time for cleanup
                    await asyncio.sleep(0.5)
                except Exception as e:
                    self._logger.warning(f"Error during LLM cleanup: {e}")

            # Reset model state
            self.model_state.is_corrupted = False
            self.model_state.corruption_time = None
            self.model_state.retry_count = 0
            self.model_state.last_error = None

            self._logger.info("Model reset completed successfully")
            return True

        except Exception as e:
            self._logger.error(f"Model reset failed: {e}")
            return False

    def _mark_model_corrupted(self, error: str) -> None:
        """Mark the model as corrupted due to timeout or error."""
        self.model_state.is_corrupted = True
        self.model_state.corruption_time = datetime.now()
        self.model_state.retry_count += 1
        self.model_state.last_error = error
        self._logger.error(f"Model marked as corrupted: {error}")

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> Any:
        """Process messages with circuit breaker protection."""

        # Check model health first
        if not await self._check_model_health():
            error_msg = (
                f"Model is in corrupted state (retry {self.model_state.retry_count}). "
                f"Last error: {self.model_state.last_error or 'Unknown'}. "
                "Please wait for cooldown period or restart the service."
            )
            return self._create_error_response(error_msg)

        # Initialize LLM if needed
        if self.llm is None:
            try:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path, tools)
            except Exception as e:
                error_msg = f"Failed to initialize LLM: {str(e)}"
                self._mark_model_corrupted(error_msg)
                return self._create_error_response(error_msg)

        # Get appropriate timeout for this request
        timeout = self._get_timeout_for_request(messages)
        self._logger.info(f"Using timeout of {timeout}s for this request")

        try:
            # Convert to LangChain messages
            lc_messages = [to_lc_message(msg) for msg in messages]

            # Create and run the graph with timeout protection
            graph = await self._create_graph_with_timeout(tools, timeout)

            # Prepare initial state
            initial_state = build_langgraph_state(
                lc_messages,
                user_input="",
                current_iteration=0,
                max_iterations=10,
                error_count=0,
            )

            # Build runnable config
            thread_source = self._generate_thread_id(messages)
            config = RunnableConfig(
                configurable={"thread_id": f"pipeline-{thread_source}"}
            )

            # Run the graph with timeout protection
            try:
                async with self._llm_lock:
                    result = await asyncio.wait_for(
                        graph.ainvoke(initial_state, config=config), timeout=timeout
                    )
            except asyncio.TimeoutError:
                error_msg = f"Pipeline timed out after {timeout}s - likely infinite loop or complex processing"
                self._logger.error(error_msg)
                self._mark_model_corrupted(error_msg)
                return self._create_error_response(
                    f"Request timed out after {timeout}s. This may be due to complex processing or model issues. "
                    "Please try a simpler request or wait for the system to recover."
                )

            # Extract and return response
            return self._extract_response_from_result(result)

        except Exception as e:
            error_msg = f"Error in process_messages: {str(e)}"
            self._logger.error(error_msg, exc_info=True)

            # Mark as corrupted if it seems like a model issue
            if any(
                keyword in str(e).lower()
                for keyword in ["timeout", "stuck", "loop", "memory", "cuda"]
            ):
                self._mark_model_corrupted(error_msg)

            return self._create_error_response(error_msg)

    async def _create_graph_with_timeout(
        self, tools: Optional[List[BaseTool]], timeout: float
    ) -> CompiledStateGraph:
        """Create LangGraph with timeout-aware agent node."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Ensure LLM initialized
        if self.llm is None:
            gguf_path = self._get_gguf_path()
            await self._initialize_llm(gguf_path, tools)
        else:
            # Bind tools if provided
            try:
                if tools and self.llm:
                    binder = getattr(self.llm, "bind_tools", None)
                    if callable(binder):
                        self.llm = cast(BaseChatModel, binder(tools))
            except Exception:
                pass

        # Build graph with timeout-aware agent node
        workflow = StateGraph(LangGraphState)

        # Create agent node with timeout protection
        async def timeout_protected_agent_node(
            state: LangGraphState, config: RunnableConfig | None = None
        ):
            return await self._agent_node_with_timeout(state, config, timeout)

        workflow.add_node("agent", timeout_protected_agent_node)

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

    async def _agent_node_with_timeout(
        self, state: LangGraphState, config: RunnableConfig | None, timeout: float
    ) -> Dict[str, Any]:
        """Agent node with built-in timeout protection and repetition detection."""

        # Check iteration limits
        if state.current_iteration >= state.max_iterations:
            return {
                "messages": [
                    coerce_to_langchain_message_dict(
                        AIMessage(
                            content="I've reached the iteration limit for safety."
                        )
                    )
                ],
                "current_iteration": state.current_iteration + 1,
            }

        if state.error_count >= 3:
            return {
                "messages": [
                    coerce_to_langchain_message_dict(
                        AIMessage(
                            content="I'm experiencing technical difficulties and need to stop."
                        )
                    )
                ],
                "error_count": state.error_count + 1,
            }

        try:
            # Convert messages
            messages = build_lc_messages(state.messages)
            assert self.llm is not None, "LLM not initialized"

            # Use a portion of the total timeout for individual LLM calls
            # Allow pipeline-specific timeouts while preventing infinite loops
            llm_timeout = min(
                timeout * 0.8, max(90.0, timeout)
            )  # Allow up to 80% of pipeline timeout, min 90s

            try:
                llm = cast(Any, self.llm)
                response = await asyncio.wait_for(
                    (
                        llm.ainvoke(messages, config=config)
                        if config is not None
                        else llm.ainvoke(messages)
                    ),
                    timeout=llm_timeout,
                )

                # Check for signs of repetitive content in response
                if self._is_response_repetitive(response):
                    self._logger.warning("Detected repetitive response, truncating")
                    if hasattr(response, "content") and isinstance(
                        response.content, str
                    ):
                        # Truncate repetitive response
                        response.content = self._truncate_repetitive_content(
                            response.content
                        )

                return {
                    "messages": [coerce_to_langchain_message_dict(response)],
                    "current_iteration": state.current_iteration + 1,
                }

            except asyncio.TimeoutError as exc:
                self._logger.error(f"LLM invocation timed out after {llm_timeout}s")
                raise RuntimeError(
                    "LLM got stuck in reasoning loop - timeout triggered"
                ) from exc

        except Exception as e:
            error_msg = str(e)
            self._logger.error(f"Agent node error: {error_msg}")

            # Return appropriate error response
            if "timeout" in error_msg.lower() or "stuck in reasoning loop" in error_msg:
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
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=f"I encountered an error: {error_msg}")
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }

    def _is_response_repetitive(self, response: Any) -> bool:
        """Check if response contains repetitive patterns."""
        if not hasattr(response, "content"):
            return False

        content = response.content
        if not isinstance(content, str) or len(content) < 200:
            return False

        # Check for repeated phrases (simple heuristic)
        words = content.split()
        if len(words) < 50:
            return False

        # Look for repeated sequences of 10+ words
        for i in range(len(words) - 20):
            sequence = " ".join(words[i : i + 10])
            remaining_text = " ".join(words[i + 10 :])
            if sequence in remaining_text:
                self._logger.warning(f"Detected repeated sequence: {sequence[:100]}...")
                return True

        return False

    def _truncate_repetitive_content(self, content: str) -> str:
        """Truncate repetitive content at the first major repetition."""
        # Simple truncation strategy with multiple guards:
        # 1) Limit to first 4 sentences
        # 2) Limit to first 12 lines
        # 3) Remove exact duplicate lines
        text = content
        # Limit lines
        lines = [ln.strip() for ln in text.splitlines()]
        seen = set()
        deduped = []
        for ln in lines:
            if ln and ln not in seen:
                deduped.append(ln)
                seen.add(ln)
        deduped = deduped[:12]
        text = "\n".join(deduped)
        # Limit sentences
        sentences = [
            s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()
        ]
        sentences = sentences[:4]
        truncated = ". ".join(sentences)
        if truncated and not truncated.endswith("."):
            truncated += "."
        return truncated + "\n\n[Response truncated due to repetitive content]"

    def _generate_thread_id(self, messages: List[Message]) -> str:
        """Generate a consistent thread ID for the conversation."""
        try:
            latest_message = messages[-1] if messages else None
            conv_id = getattr(latest_message, "conversation_id", None)
            thread_source = (
                f"{conv_id}-{len(messages)}"
                if conv_id is not None
                else str(uuid.uuid4())
            )
        except Exception:
            thread_source = str(uuid.uuid4())

        return hashlib.md5(thread_source.encode()).hexdigest()[:16]

    def _extract_response_from_result(self, result: Dict[str, Any]) -> Any:
        """Extract response from graph result."""
        if result.get("messages"):
            final_message = result["messages"][-1]
            response_text = ""

            # Support both LangChain AIMessage and dict-shaped messages (from coerce_to_langchain_message_dict)
            if isinstance(final_message, AIMessage):
                content = getattr(final_message, "content", None)
                if content:
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, str):
                                response_text += c + " "
                            elif isinstance(c, dict) and "text" in c:
                                response_text += c["text"] + " "
            elif isinstance(final_message, dict):
                content = final_message.get("content")
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, str):
                            response_text += c + " "
                        elif isinstance(c, dict) and "text" in c:
                            response_text += c["text"] + " "

            response_text = response_text.strip()

            # Strip hidden chain-of-thought / <think> blocks if model profile disables thinking
            try:
                if not self.profile.parameters.think:
                    # Remove <think>...</think> sections and similar markers
                    import re

                    response_text = re.sub(
                        r"<think>[\s\S]*?</think>",
                        "",
                        response_text,
                        flags=re.IGNORECASE,
                    )
                    # Also drop leading bracketed meta lines
                    lines = [
                        ln
                        for ln in response_text.splitlines()
                        if not ln.lower().startswith("[think")
                    ]
                    response_text = "\n".join(lines).strip()
            except Exception:
                pass

            # Return appropriate type based on generic parameter
            if self.expected_return_type == str:
                # Guard against pathological repetition in plain text mode
                if len(response_text) > 4000:
                    response_text = self._truncate_repetitive_content(response_text)
                self.validate_return_value(response_text)
                return response_text
            else:
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
                    created_at=datetime.now(),
                    finish_reason="stop",
                )
                self.validate_return_value(chat)
                return chat

        # If no response, return error
        return self._create_error_response("No response generated")

    def _create_error_response(self, error_msg: str) -> Any:
        """Create an error response of the appropriate type."""
        if self.expected_return_type == str:
            return error_msg
        else:
            return ChatResponse(
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
                created_at=datetime.now(),
                finish_reason="error",
            )

    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self.llm is not None:
            try:
                # Basic cleanup
                pass
            except Exception as e:
                self._logger.warning(f"Error during cleanup: {e}")
            finally:
                self.llm = None

        # Clear caches
        self.graph_cache.clear()
        self.model_state = ModelState()

    async def _generate_with_monitoring(self, messages: List[Dict[str, Any]]):
        """
        Generate response with real-time monitoring for perplexity and repetition,
        using the modern astream_events API.
        """
        from collections import deque
        import re
        import math

        # Configuration for monitoring
        repetition_window_size = 30
        repetition_threshold = 4
        perplexity_threshold = self.circuit_config.perplexity_threshold

        recent_phrases = deque(maxlen=repetition_window_size)
        full_response = ""

        if not self.llm:
            raise RuntimeError("LLM not initialized. Cannot generate response.")

        try:
            # Convert dict messages to LangChain BaseMessage objects
            lc_messages = build_lc_messages(messages)

            async for event in self.llm.astream_events(lc_messages, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if not isinstance(chunk, AIMessageChunk):
                        continue

                    # Aggregate content
                    content_piece = chunk.content
                    if isinstance(content_piece, str):
                        full_response += content_piece
                    elif isinstance(content_piece, list):
                        for part in content_piece:
                            if isinstance(part, str):
                                full_response += part
                            elif isinstance(part, dict) and part.get("type") == "text":
                                full_response += part.get("text", "")

                    # --- Perplexity Check ---
                    if (
                        perplexity_threshold > 0
                        and chunk.response_metadata
                        and "logprobs" in chunk.response_metadata
                    ):
                        logprobs_data = chunk.response_metadata["logprobs"]
                        if isinstance(logprobs_data, list) and logprobs_data:
                            logprob_item = logprobs_data[0]
                            if isinstance(logprob_item, dict) and "logprob" in logprob_item:
                                logprob = logprob_item["logprob"]
                                if logprob != 0.0:
                                    perplexity = math.exp(-logprob)
                                    if perplexity > perplexity_threshold:
                                        self._logger.warning(
                                            f"High perplexity detected: {perplexity:.2f} > {perplexity_threshold}. "
                                            "Model may be confused. Stopping generation."
                                        )
                                        full_response += "\n[Generation stopped due to high perplexity]"
                                        return AIMessage(content=full_response)

                    # --- Repetition Check ---
                    words = re.findall(r"\b\w+\b", full_response.lower())
                    if len(words) > 5:
                        last_phrase = " ".join(words[-5:])
                        if last_phrase in recent_phrases:
                            count = recent_phrases.count(last_phrase)
                            if count >= repetition_threshold:
                                self._logger.warning(
                                    f"Detected repetitive sequence: '{last_phrase}'. Stopping generation."
                                )
                                full_response += "\n[Response truncated due to repetition]"
                                return AIMessage(content=full_response)
                        recent_phrases.append(last_phrase)

            return AIMessage(content=full_response)

        except Exception as e:
            self._logger.error(f"Error during monitored generation: {e}", exc_info=True)
            # Fallback to a simple invoke on streaming error
            if self.llm:
                return await self.llm.ainvoke(messages)
            return AIMessage(content=f"[Error in generation: {e}]")