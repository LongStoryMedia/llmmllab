"""
Optimized LangGraph-based implementation for Qwen3 A3B MoE models.
Refactored to use the base LangGraph pipeline with improved timeout protection.
"""

import os
import logging
import asyncio
from typing import List, Optional, TypeVar, Union, Dict, Any, cast

# Avoid importing torch at module import time (can hang on GPU init in some envs)
torch = None  # type: ignore

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

# Avoid importing ChatLlamaCpp at module import time to prevent heavy GPU lib load in dev/test
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from utils.langgraph import (
    LangGraphState,
    build_lc_messages,
    coerce_to_langchain_message_dict,
)
from ..base_langgraph import BaseLangGraphPipeline, CircuitBreakerConfig
from .context_manager import ContextManager

T = TypeVar("T", bound=Union[str, ChatResponse])


class QwenLangGraphPipe(BaseLangGraphPipeline[T]):
    """
    Qwen3 A3B MoE pipeline with enhanced timeout protection and circuit breaker functionality.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
    ):
        # Configure circuit breaker with appropriate timeouts for Qwen
        circuit_config = CircuitBreakerConfig(
            base_timeout=60.0,  # Increased from 30s
            deep_research_timeout=180.0,  # 3 minutes for complex research
            max_retries=2,
            cooldown_period=45.0,
        )

        super().__init__(model, profile, expected_return_type, circuit_config)
        self.model = model
        self.profile = profile
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize context manager
        context_tokens = 32768 if "30b" in self.model.name.lower() else 16384
        self.context_manager = ContextManager(max_context_tokens=context_tokens)

        # Validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path for the model."""
        # Use the same logic as v1 - rely on model details or model path
        return (
            self.model.details.gguf_file
            if self.model.details and self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate that the GGUF file exists and is accessible."""
        # Allow bypassing validation in dev/test environments
        if os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in (
            "1",
            "true",
            "yes",
        ):  # pragma: no cover
            self._logger.warning(
                f"Skipping GGUF validation for dev/test (ALLOW_MISSING_GGUF set). Expected at: {gguf_path}"
            )
            return

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        if not os.access(gguf_path, os.R_OK):
            raise PermissionError(f"Cannot read GGUF file: {gguf_path}")

    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Initialize the ChatLlamaCpp LLM with optimized parameters."""
        self._logger.info(f"Initializing Qwen LLM from {gguf_path}")

        # Dev/test fallback or missing deps: use a simple dummy LLM
        allow_dummy = os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        if allow_dummy:

            class _DummyLLM:
                async def ainvoke(self, _messages, **_):  # noqa: ANN001
                    return AIMessage(content="[dev] test response")

            self._logger.warning(
                "Using dummy LLM (dev/test mode or missing ChatLlamaCpp)"
            )
            self.llm = _DummyLLM()
            # Bind tools if provided (no-op for dummy)
            if tools and hasattr(self.llm, "bind_tools"):
                try:  # noqa: SIM105
                    self.llm = cast(Any, self.llm).bind_tools(tools)  # type: ignore[attr-defined]
                except Exception as e:  # noqa: BLE001
                    self._logger.warning(f"Failed to bind tools: {e}")
            return

        else:
            # Get context size with reasonable limits to prevent memory issues
            requested_ctx = self.profile.parameters.num_ctx or 32768
            # Cap context size for 30B models to prevent memory exhaustion
            max_ctx = 65536 if "30b" in self.model.name.lower() else 32768
            n_ctx = min(requested_ctx, max_ctx)
            
            if requested_ctx > max_ctx:
                self._logger.warning(
                    f"Requested context {requested_ctx} exceeds safe limit {max_ctx} for this model, "
                    f"using {n_ctx} instead"
                )
            
            optimal_gpu_layers = self._calculate_optimal_gpu_layers(n_ctx, "large")  # 30B is large size
            
            self._logger.info(
                f"Dynamic GPU allocation: n_ctx={n_ctx}, using {optimal_gpu_layers} GPU layers"
            )
            
            # Initialize ChatLlamaCpp with optimized parameters
            # System prompt is incorporated upstream; creating LLM instance
            try:
                self.llm = ChatLlamaCpp(
                    model_path=gguf_path,
                    n_gpu_layers=optimal_gpu_layers,  # Dynamic GPU allocation based on context
                    n_batch=512,  # Restore original batch size
                    f16_kv=True,  # Enable f16 for GPU efficiency
                    verbose=os.environ.get("LOG_LEVEL", "warning") == "debug",
                    n_parts=-1,
                    streaming=True,
                    n_ctx=n_ctx,  # Use the determined context size
                    use_mmap=True,  # Enable memory mapping for efficiency
                    use_mlock=False,  # Disable mlock to avoid memory pressure
                    # Model parameters
                    seed=self.profile.parameters.seed or -1,
                    temperature=self.profile.parameters.temperature or 0.7,
                    max_tokens=self.profile.parameters.max_tokens or 4096,
                    top_p=self.profile.parameters.top_p or 0.8,
                    top_k=self.profile.parameters.top_k or 20,
                    repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
                    stop=self.profile.parameters.stop
                    or [
                        "<|im_end|>",
                        "<|endoftext|>",
                        "<|end|>",
                    ],
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                )
            except Exception as primary_error:
                # Try with very conservative settings for 30B model
                self._logger.warning(
                    f"Primary model load failed: {primary_error}. "
                    f"Trying with ultra-conservative settings for 30B model..."
                )
                try:
                    conservative_ctx = min(n_ctx, 16384)  # Much smaller context
                    conservative_gpu_layers = max(0, optimal_gpu_layers - 15)  # Fewer GPU layers
                    
                    self.llm = ChatLlamaCpp(
                        model_path=gguf_path,
                        n_gpu_layers=conservative_gpu_layers,
                        n_batch=256,  # Smaller batch
                        f16_kv=False,  # Disable f16_kv for compatibility
                        verbose=True,  # Enable verbose for debugging
                        n_parts=-1,
                        streaming=True,
                        n_ctx=conservative_ctx,
                        use_mmap=False,  # Disable mmap
                        use_mlock=False,
                        seed=-1,
                        temperature=0.7,
                        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                    )
                    self._logger.info(
                        f"30B model loaded with conservative settings: "
                        f"ctx={conservative_ctx}, gpu_layers={conservative_gpu_layers}"
                    )
                except Exception as secondary_error:
                    self._logger.error(
                        f"Failed to load 30B model with both standard and conservative settings. "
                        f"Primary: {primary_error}. Secondary: {secondary_error}"
                    )
                    raise secondary_error

            # Prepend system prompt via bind if supported

        # Bind tools if provided
        if tools and hasattr(self.llm, "bind_tools"):
            self.llm = self.llm.bind_tools(tools)

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create optimized system prompt with anti-loop instructions."""
        base_prompt = (
            self.profile.system_prompt
            or """You are a helpful AI assistant. When thinking through problems:

CRITICAL THINKING GUIDELINES:
- Keep your reasoning concise and focused (max 2-3 short paragraphs)
- Avoid repeating the same logic or analysis multiple times
- If you find yourself restating similar points, STOP and provide your answer
- Do not elaborate on the same concept repeatedly
- Make your thinking efficient and direct

RESPONSE STRUCTURE:
1. Brief analysis (if needed)
2. Direct, clear answer
3. Move on immediately

Avoid circular reasoning, excessive elaboration, or repetitive explanations. Be decisive and concise."""
        )

        # Add tool information if available
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            tool_info = "\n".join(tool_descriptions)
            base_prompt += f"\n\nAvailable tools:\n{tool_info}\n\nUse tools when appropriate, but keep explanations brief."

        return base_prompt

    def _should_use_extended_timeout(self, messages: List[Message]) -> bool:
        """
        Determine if this request should use extended timeout.
        Enhanced detection for Qwen-specific patterns.
        """
        # Keywords that indicate complex processing
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
            "surprise party",
            "scrolling newsfeed",
            "programming",
            "step by step",
            "explain",
            "tutorial",
            "guide",
        ]

        for message in messages:
            if message.content:
                for content in message.content:
                    if content.type == MessageContentType.TEXT and content.text:
                        text_lower = content.text.lower()
                        # Check for multiple keywords or long text
                        keyword_count = sum(
                            1 for keyword in extended_keywords if keyword in text_lower
                        )
                        if keyword_count >= 2 or len(content.text) > 200:
                            return True
        return False

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
    ) -> CompiledStateGraph:
        """Create LangGraph with optimized caching and timeout protection."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Initialize LLM if not done yet (synchronously)
        if self.llm is None:
            # This will be handled during first agent node execution
            pass

        # Build graph with our custom agent node
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

    async def _agent_node(self, state: LangGraphState, config=None) -> Dict[str, Any]:
        """Agent node with enhanced timeout protection and circuit breaker."""
        _ = config  # Acknowledge unused parameter

        # Check iteration limits
        if state.current_iteration >= state.max_iterations:
            timeout_error = f"Maximum iterations ({state.max_iterations}) reached. Stopping to prevent infinite loops."
            return {
                "messages": [
                    coerce_to_langchain_message_dict(AIMessage(content=timeout_error))
                ],
                "current_iteration": state.current_iteration + 1,
            }

        try:
            # Initialize LLM if not done yet
            if self.llm is None:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path)

            # Build messages for LLM
            messages = build_lc_messages(state.messages)

            # Determine timeout based on query complexity using original messages from state
            # Convert LangChain messages back to our Message format for analysis
            original_messages = []
            for lc_msg in state.messages:
                if hasattr(lc_msg, "content"):
                    msg = Message(
                        role=(
                            MessageRole.USER
                            if lc_msg.type == "human"
                            else MessageRole.ASSISTANT
                        ),
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=str(lc_msg.content)
                            )
                        ],
                    )
                    original_messages.append(msg)

            is_deep_research = self._should_use_extended_timeout(original_messages)
            timeout_seconds = (
                min(self.circuit_config.deep_research_timeout, 120.0)
                if is_deep_research
                else min(self.circuit_config.base_timeout, 60.0)
            )

            # Execute with timeout protection - use ainvoke directly
            # Execute with timeout protection - use ainvoke directly
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=timeout_seconds,
            )

            return {
                "messages": [coerce_to_langchain_message_dict(response)],
                "current_iteration": state.current_iteration + 1,
            }

        except asyncio.TimeoutError:
            timeout_error = f"LLM request timed out after {timeout_seconds}s. This may indicate the model got stuck in reasoning loops."
            self._logger.warning(timeout_error)
            return {
                "messages": [
                    coerce_to_langchain_message_dict(AIMessage(content=timeout_error))
                ],
                "current_iteration": state.current_iteration + 1,
            }
        except Exception as e:
            error_msg = f"Error in agent node: {str(e)}"
            self._logger.error(error_msg)
            return {
                "messages": [
                    coerce_to_langchain_message_dict(AIMessage(content=error_msg))
                ],
                "current_iteration": state.current_iteration + 1,
            }

    def cleanup(self) -> None:
        """Enhanced cleanup for Qwen-specific resources."""
        super().cleanup()

        # Additional Qwen-specific cleanup if needed
        try:
            if hasattr(self, "context_manager"):
                # Reset context manager state
                pass
        except Exception as e:
            self._logger.warning(f"Error during Qwen-specific cleanup: {e}")
