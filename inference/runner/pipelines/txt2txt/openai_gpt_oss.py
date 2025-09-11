"""
OpenAI GPT OSS 20B Abliterated Uncensored pipeline implementation.
Based on the QwenLangGraphPipe with optimizations for the OpenAI GPT OSS 20B model.
"""

import os
import logging
import asyncio
from typing import List, Optional, Dict, Any

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


class OpenAiGptOssPipe(BaseLlamaCppPipeline):
    """
    OpenAI GPT OSS 20B pipeline with enhanced timeout protection and circuit breaker functionality.
    Optimized for the DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored model.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
    ):
        # Configure circuit breaker with appropriate timeouts for 20B model
        circuit_config = CircuitBreakerConfig(
            base_timeout=90.0,  # Increased for 20B model
            deep_research_timeout=240.0,  # 4 minutes for complex research
            max_retries=2,
            cooldown_period=60.0,
        )

        super().__init__(model, profile, expected_return_type, circuit_config, "medium")
        self.model = model
        self.profile = profile
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize context manager with dynamic context for 20B model
        context_tokens = (
            self.profile.parameters.num_ctx or 4096
        )  # Match dynamic context
        self.context_manager = ContextManager(max_context_tokens=context_tokens)

        # Validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path for the model."""
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

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create optimized system prompt for OpenAI GPT OSS 20B."""
        base_prompt = (
            self.profile.system_prompt
            or """You are a helpful, honest, and capable AI assistant. You provide direct, informative responses without unnecessary restrictions. When thinking through problems:

RESPONSE GUIDELINES:
- Be direct and honest in your responses
- Provide comprehensive information when requested
- Think through problems step by step when helpful
- Acknowledge uncertainty when you don't know something
- Be helpful while maintaining ethical considerations

TECHNICAL CAPABILITIES:
- Code analysis and generation
- Research and information synthesis
- Creative writing and ideation
- Problem-solving and reasoning
- Educational explanations

Answer questions thoroughly and helpfully."""
        )

        # Add tool information if available
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            tool_info = "\n".join(tool_descriptions)
            base_prompt += f"\n\nAvailable tools:\n{tool_info}\n\nUse tools when appropriate to provide accurate, up-to-date information."

        return base_prompt

    def _should_use_extended_timeout(self, messages: List[Message]) -> bool:
        """
        Determine if this request should use extended timeout.
        Enhanced detection for complex processing patterns.
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
            "code review",
            "programming",
            "step by step",
            "explain",
            "tutorial",
            "guide",
            "compare",
            "evaluate",
            "write",
            "create",
            "design",
            "algorithm",
            "architecture",
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
                        if keyword_count >= 2 or len(content.text) > 300:
                            return True
        return False

    async def prompt(self, text: str | List[str]) -> str | ChatResponse:
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

            # Ensure LLM is properly initialized after the initialization call
            if self.llm is None:
                raise RuntimeError(
                    "Failed to initialize LLM - llm is still None after initialization"
                )

            # Build messages for LLM
            messages = build_lc_messages(state.messages)

            # Determine timeout based on query complexity
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
                min(self.circuit_config.deep_research_timeout, 180.0)
                if is_deep_research
                else min(self.circuit_config.base_timeout, 90.0)
            )

            # Execute with timeout protection
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=timeout_seconds,
            )

            return {
                "messages": [coerce_to_langchain_message_dict(response)],
                "current_iteration": state.current_iteration + 1,
            }

        except asyncio.TimeoutError:
            timeout_error = f"Request timed out after {timeout_seconds:.1f}s. The model may be processing a complex query."
            self._logger.warning(f"OpenAI GPT OSS 20B timeout: {timeout_error}")
            return {
                "messages": [
                    coerce_to_langchain_message_dict(AIMessage(content=timeout_error))
                ],
                "current_iteration": state.current_iteration + 1,
            }
        except Exception as e:
            error_msg = f"Error in OpenAI GPT OSS 20B agent node: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            return {
                "messages": [
                    coerce_to_langchain_message_dict(AIMessage(content=error_msg))
                ],
                "current_iteration": state.current_iteration + 1,
            }
