"""
Qwen3 A3B MoE pipeline - Simplified for deterministic <think> tag processing.

This pipeline leverages the Qwen3 model's native tokenizer support for <think>...</think> tags
(tokens 151667 and 151668) to provide clean separation between reasoning and response content.
"""

import os
import re
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
from ..context_manager import ContextManager

ReturnType = Union[str, ChatResponse]


class QwenLangGraphPipe(BaseLlamaCppPipeline):
    """
    Qwen3 A3B MoE pipeline with deterministic <think> tag processing.

    This pipeline is optimized for the Qwen3 model's native think tokens:
    - Token 151667: <think>
    - Token 151668: </think>

    Features:
    - Real-time streaming with think content separation
    - Thinking field population during streaming
    - Clean response content without think tags
    - Circuit breaker protection for safety
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

        # Initialize think tag state
        self._reset_think_state()

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

    def _reset_think_state(self) -> None:
        """Reset think tag tracking state."""
        self.think_content: str = ""
        self.in_think_tag: bool = False
        self.buffer: str = ""

    def reset_streaming_state(self) -> None:
        """Reset streaming state (overrides base method)."""
        super().reset_streaming_state()
        self._reset_think_state()

    def process_streaming_token(self, content: str) -> Optional[ChatResponse]:
        """Process streaming token with <think> tag detection (overrides base method)."""
        self.buffer += content

        # Check for opening think tag
        if "<think>" in content and not self.in_think_tag:
            self.in_think_tag = True
            return None

        # Check for closing think tag
        elif "</think>" in content and self.in_think_tag:
            self.in_think_tag = False
            return None

        # If we're inside think tags, accumulate in think_content but don't return
        if self.in_think_tag:
            return self._create_thinking_response(content)

        # Normal content outside think tags
        return self._create_streaming_response(content)

    def finalize_streaming(self) -> Optional[ChatResponse]:
        """Finalize streaming and return any remaining content (overrides base method)."""
        # Create final message with thinking content and any remaining buffer
        thinking = self.think_content if self.think_content else None
        content = []

        if self.buffer and not self.in_think_tag:
            content = [
                MessageContent(type=MessageContentType.TEXT, text=self.buffer.strip())
            ]

        # Reset state
        self._reset_think_state()

        # Return message with thinking field if we have thinking content
        if thinking or content:
            message = Message(
                role=MessageRole.ASSISTANT, thinking=thinking, content=content
            )
            return ChatResponse(message=message, done=True)

        return None

    # llama.cpp initialization inherited from BaseLlamaCppPipeline

    # Perplexity / repetition guard logic now inherited from BaseLangGraphPipeline

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt for Qwen3 thinking model."""
        base_prompt = (
            self.profile.system_prompt
            or """You are a helpful AI assistant. Think through problems step by step using <think>...</think> tags.

Your thinking process will be captured separately from your response. Use the thinking space to:
- Analyze the problem
- Consider different approaches  
- Work through the logic

Then provide your clear, direct answer outside the thinking tags."""
        )

        # Add tool information if available
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            tool_info = "\n".join(tool_descriptions)
            base_prompt += (
                f"\n\nAvailable tools:\n{tool_info}\n\nUse tools when appropriate."
            )

        return base_prompt

    def extract_channels(self, response_text: str) -> Dict[str, Any]:
        """
        Qwen3-specific channel extraction for deterministic <think> tags only.
        """
        result = {
            "thinking": None,
            "tool_calls": None,
            "status": None,
            "cleaned_response": response_text,
            "channels": {},
        }

        cleaned_text = response_text

        # Extract thinking content from <think>...</think> tags
        import re

        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, cleaned_text, re.DOTALL)

        if think_matches:
            # Combine all thinking content
            result["thinking"] = "\n\n".join(match.strip() for match in think_matches)
            # Remove think tags from cleaned response
            cleaned_text = re.sub(think_pattern, "", cleaned_text, flags=re.DOTALL)

        # Extract tool calls from <tool_call>...</tool_call> tags (if present)
        tool_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        tool_matches = re.findall(tool_pattern, cleaned_text, re.DOTALL)

        if tool_matches:
            tool_calls = []
            for match in tool_matches:
                try:
                    import json

                    tool_call = json.loads(match.strip())
                    tool_calls.append(tool_call)
                except:
                    # Store as raw text if JSON parsing fails
                    tool_calls.append({"raw": match.strip()})

            if tool_calls:
                result["tool_calls"] = tool_calls
                # Remove tool call tags from cleaned response
                cleaned_text = re.sub(tool_pattern, "", cleaned_text, flags=re.DOTALL)

        # Clean up extra whitespace
        result["cleaned_response"] = re.sub(
            r"\n\s*\n\s*\n", "\n\n", cleaned_text
        ).strip()

        return result

    def _should_use_extended_timeout(self, messages: List[Message]) -> bool:
        """
        Determine if this request should use extended timeout.
        Simple heuristic based on message length and complexity indicators.
        """
        # Check for long messages or complex keywords
        for message in messages:
            if message.content:
                for content in message.content:
                    if content.type == MessageContentType.TEXT and content.text:
                        text = content.text.lower()
                        # Long messages likely need more time
                        if len(content.text) > 500:
                            return True
                        # Complex task indicators
                        complex_keywords = [
                            "analyze",
                            "research",
                            "detailed",
                            "comprehensive",
                            "step by step",
                        ]
                        if any(keyword in text for keyword in complex_keywords):
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
