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
        # Configure circuit breaker with MUCH longer timeouts for research/testing
        circuit_config = CircuitBreakerConfig(
            base_timeout=300.0,  # 5 minutes base timeout
            deep_research_timeout=900.0,  # 15 minutes for complex research
            max_retries=3,
            cooldown_period=60.0,
        )

        super().__init__(model, profile, expected_return_type, circuit_config)
        self.model = model
        self.profile = profile
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize context manager with max possible context
        context_tokens = 1048576 if "30b" in self.model.name.lower() else 131072
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

    def total_model_layers(self) -> int:
        return 48

    @property
    def model_size_category(self) -> str:
        return "large"

    def _create_llm_instance(
        self,
        gguf_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        tools: Optional[List[BaseTool]] = None,
    ) -> ChatLlamaCpp:
        """Create an instance of the ChatLlamaCpp LLM."""
        batch_size = min(2048, max(512, n_ctx // 64))

        llm = ChatLlamaCpp(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=batch_size,
            temperature=self.profile.parameters.temperature or 0.7,
            max_tokens=self.profile.parameters.max_tokens or -1,
            top_p=self.profile.parameters.top_p or 0.95,
            top_k=self.profile.parameters.top_k or 40,
            repeat_penalty=getattr(self.profile.parameters, "repetition_penalty", 1.1)
            or 1.1,
            streaming=True,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            logprobs=1,  # Enable logprobs for perplexity calculation
        )

        if tools:
            llm = cast(ChatLlamaCpp, llm.bind_tools(tools))

        return llm



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

            # Execute with timeout protection and monitoring
            response = await asyncio.wait_for(
                self._generate_with_monitoring(messages),
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
