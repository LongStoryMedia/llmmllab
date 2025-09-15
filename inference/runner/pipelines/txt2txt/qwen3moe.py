"""Qwen3 A3B MoE pipeline leveraging shared adaptive streaming (perplexity & repetition guards)."""

import os
import logging
from typing import List, Optional, Union, Dict, Any, Type

# Avoid importing torch at module import time (can hang on GPU init in some envs)
torch = None  # type: ignore

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

# Avoid importing ChatLlamaCpp at module import time to prevent heavy GPU lib load in dev/test
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
from ..base_langgraph import CircuitBreakerConfig
from ..llamacpp.base_llamacpp import BaseLlamaCppPipeline
from .context_manager import ContextManager

ReturnType = Union[str, ChatResponse]


class QwenLangGraphPipe(BaseLlamaCppPipeline):
    """
    Qwen3 A3B MoE pipeline with enhanced timeout protection and circuit breaker functionality.
    """

    # Override allowed return types to include Type for compatibility with typing system
    allowed_return_types: tuple[type, ...] = (str, ChatResponse, list, Type)

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        # Create logger early so we can use it
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Log the received circuit config for debugging
        if circuit_config is not None:
            self._logger.info(
                f"QwenLangGraphPipe: Received circuit_config with perplexity_guard={circuit_config.enable_perplexity_guard}"
            )
        else:
            self._logger.info(
                "QwenLangGraphPipe: No circuit_config provided, will use defaults from BaseLangGraphPipeline"
            )

        # Let the parent class handle circuit breaker configuration and defaults
        super().__init__(model, profile, expected_return_type, circuit_config)
        self.model = model
        self.profile = profile

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

    # llama.cpp initialization inherited from BaseLlamaCppPipeline

    # Perplexity / repetition guard logic now inherited from BaseLangGraphPipeline

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

    async def prompt(self, text: str | List[str]) -> ReturnType:
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

            # Stream with shared adaptive controls
            response = await self._stream_with_adaptive_controls(messages)

            return {
                "messages": [coerce_to_langchain_message_dict(response)],
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
