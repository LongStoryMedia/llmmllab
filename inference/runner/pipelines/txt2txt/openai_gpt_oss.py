"""
OpenAI GPT OSS 20B Abliterated Uncensored pipeline implementation.
Based on the QwenLangGraphPipe with optimizations for the OpenAI GPT OSS 20B model.
"""

import os
import logging
from typing import List, Optional, Dict, Any

# Avoid importing torch at module import time (can hang on GPU init in some envs)
torch = None  # type: ignore

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

# LangGraph components (mirroring qwen3moe pattern)
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Import harmony formatting utilities
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role as HarmonyRole,
    Message as HarmonyMessage,
    Conversation as HarmonyConversation,
    SystemContent,
)

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
    coerce_to_langchain_message_dict,
)
from utils.message import from_lc_message
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
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        # Circuit breaker config fallback is handled by BaseLangGraphPipeline
        super().__init__(model, profile, expected_return_type, circuit_config, "medium")
        self.model = model
        self.profile = profile
        self._logger.setLevel(logging.DEBUG)

        # Initialize context manager with dynamic context for 20B model
        # OpenAI GPT OSS 20B supports large context windows, default to 131072 if not explicitly set
        context_tokens = (
            self.profile.parameters.num_ctx or 131072
        )  # Increased from 4096 to full 131072 for better context support

        # Ensure the profile's num_ctx parameter reflects the increased context
        if not self.profile.parameters.num_ctx:
            self.profile.parameters.num_ctx = 131072
            self._logger.info(f"Set OpenAI GPT OSS context window to {131072} tokens")

        # Initialize context manager
        self.context_manager = ContextManager(max_context_tokens=context_tokens)

        # Default reasoning effort (may be overridden later)
        self._reasoning_effort = getattr(
            self.profile.parameters, "reasoning_effort", "medium"
        )

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

    def _get_optimal_gpt_oss_parameters(self) -> Dict[str, Any]:
        """Get optimal parameters for GPT OSS models based on research and documentation.

        Based on documentation from llama.cpp discussions and OpenAI harmony format,
        these parameters provide the best balance of creativity and coherence for GPT OSS models.

        Returns:
            Dict of optimal parameter overrides for GPT OSS models
        """
        # Optimal sampling parameters for GPT OSS models
        optimal_params = {
            "temperature": 1.0,  # Full range for creative responses
            "top_p": 1.0,  # No nucleus sampling restriction
            "top_k": 0,  # Disable top-k filtering to allow full vocabulary
            "min_p": 0.001,  # Small min_p for quality while preserving creativity
            "repeat_penalty": 1.0,  # Disable repeat penalty (can interfere with reasoning)
        }

        # Override with profile parameters if explicitly set (respect user preferences)
        if self.profile.parameters.temperature is not None:
            optimal_params["temperature"] = self.profile.parameters.temperature
        if self.profile.parameters.top_p is not None:
            optimal_params["top_p"] = self.profile.parameters.top_p
        if self.profile.parameters.top_k is not None:
            optimal_params["top_k"] = self.profile.parameters.top_k
        if self.profile.parameters.min_p is not None:
            optimal_params["min_p"] = self.profile.parameters.min_p
        if self.profile.parameters.repeat_penalty is not None:
            optimal_params["repeat_penalty"] = self.profile.parameters.repeat_penalty

        # Add reasoning effort configuration if available
        reasoning_effort = getattr(
            self.profile.parameters, "reasoning_effort", "medium"
        )
        if reasoning_effort:
            # Reasoning effort affects the model's chain-of-thought depth
            # This is used in system prompt formatting rather than llama.cpp parameters
            self._reasoning_effort = reasoning_effort
            self._logger.info(f"GPT OSS reasoning effort level: {reasoning_effort}")

        self._logger.info(f"Applied GPT OSS optimal parameters: {optimal_params}")
        return optimal_params

    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Initialize llama.cpp model with GPT OSS-optimized parameters.

        Overrides the base class to apply optimal sampling parameters for GPT OSS models
        before calling the parent initialization with heuristic auto-backoff.
        """
        # Get optimal parameters for GPT OSS models
        optimal_params = self._get_optimal_gpt_oss_parameters()

        # Apply optimal parameters to the profile before initialization
        # This ensures the base class uses these values during initialization
        if optimal_params.get("temperature") is not None:
            self.profile.parameters.temperature = optimal_params["temperature"]
        if optimal_params.get("top_p") is not None:
            self.profile.parameters.top_p = optimal_params["top_p"]
        if optimal_params.get("top_k") is not None:
            self.profile.parameters.top_k = optimal_params["top_k"]
        if optimal_params.get("min_p") is not None:
            self.profile.parameters.min_p = optimal_params["min_p"]
        if optimal_params.get("repeat_penalty") is not None:
            self.profile.parameters.repeat_penalty = optimal_params["repeat_penalty"]

        self._logger.info(
            "GPT OSS pipeline applying optimal parameters for initialization"
        )

        # Call parent initialization with optimized parameters
        await super()._initialize_llm(gguf_path, tools)

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create harmony-compatible system prompt for OpenAI GPT OSS 20B."""

        # Base system prompt with harmony format requirements
        base_prompt = (
            self.profile.system_prompt
            or """You are a large language model with non-restrictive policies.
Knowledge cutoff: 2024-06
Current date: 2025-09-16

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Use 'analysis' channel for chain-of-thought reasoning
# Use 'commentary' channel for tool calls and function descriptions  
# Use 'final' channel for your response to the user

You are a helpful, honest, and capable AI assistant. You provide direct, informative responses while showing your reasoning process.

RESPONSE GUIDELINES:
- Always use the appropriate channel for your content
- Show your reasoning in the analysis channel when helpful
- Be direct and honest in your final responses
- Provide comprehensive information when requested
- Acknowledge uncertainty when you don't know something

TECHNICAL CAPABILITIES:
- Code analysis and generation
- Research and information synthesis
- Creative writing and ideation
- Problem-solving and reasoning
- Educational explanations"""
        )

        # Add tool information if available (goes to commentary channel)
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            tool_info = "\n".join(tool_descriptions)
            base_prompt += f"""

Calls to these tools must go to the commentary channel: 'functions'.

# Tools

namespace functions {{

{tool_info}

}} // namespace functions

Use tools when appropriate to provide accurate, up-to-date information."""

        return base_prompt

    def _convert_to_harmony_messages(self, messages: List[Message]) -> List[Any]:
        """Convert internal Message objects to harmony-compatible format."""
        harmony_messages = []

        for message in messages:
            # Extract text content from message
            text_content = ""
            if message.content:
                for content in message.content:
                    if content.type == MessageContentType.TEXT and content.text:
                        text_content += content.text

            # Convert role to harmony format
            if message.role == MessageRole.SYSTEM:
                harmony_role = HarmonyRole.SYSTEM
                harmony_content = SystemContent.new()
            elif message.role == MessageRole.USER:
                harmony_role = HarmonyRole.USER
                harmony_content = text_content
            elif message.role == MessageRole.ASSISTANT:
                harmony_role = HarmonyRole.ASSISTANT
                harmony_content = text_content
            else:
                # Default to user for unknown roles
                harmony_role = HarmonyRole.USER
                harmony_content = text_content

            harmony_message = HarmonyMessage.from_role_and_content(
                harmony_role, harmony_content
            )
            harmony_messages.append(harmony_message)

        return harmony_messages

    def _message_to_lc_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message to LangChain-compatible dict format."""
        text_content = ""
        if message.content:
            for content in message.content:
                if content.type == MessageContentType.TEXT and content.text:
                    text_content += content.text

        if message.role == MessageRole.SYSTEM:
            role_type = "system"
        elif message.role == MessageRole.USER:
            role_type = "human"
        elif message.role == MessageRole.ASSISTANT:
            role_type = "ai"
        else:
            role_type = "human"  # Default fallback

        return {
            "content": text_content,
            "type": role_type,
            "additional_kwargs": {},
            "response_metadata": {},
        }

    def _validate_harmony_format(self, text_content: str) -> Dict[str, Any]:
        """Validate harmony format requirements for GPT OSS models.

        Checks for required channels (analysis, commentary, final) and proper token usage.

        Args:
            text_content: The text content to validate

        Returns:
            Dict with validation results and recommendations
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "channels_found": [],
            "missing_channels": [],
            "recommendations": [],
        }

        # Required channels for proper harmony format
        required_channels = ["analysis", "commentary", "final"]

        # Check for channel usage
        for channel in required_channels:
            if (
                f"#{channel}" in text_content.lower()
                or f"channel: {channel}" in text_content.lower()
            ):
                validation_result["channels_found"].append(channel)
            else:
                validation_result["missing_channels"].append(channel)

        # Validate if essential channels are present
        if "final" not in validation_result["channels_found"]:
            validation_result["is_valid"] = False
            validation_result["warnings"].append(
                "Missing required 'final' channel for user response"
            )
            validation_result["recommendations"].append(
                "Include 'final' channel for direct user responses"
            )

        # Check for reasoning indicators
        reasoning_indicators = ["thinking", "reasoning", "analysis", "chain-of-thought"]
        has_reasoning = any(
            indicator in text_content.lower() for indicator in reasoning_indicators
        )

        if not has_reasoning and "analysis" not in validation_result["channels_found"]:
            validation_result["warnings"].append(
                "No reasoning or analysis channel detected"
            )
            validation_result["recommendations"].append(
                "Consider using 'analysis' channel for chain-of-thought reasoning"
            )

        # Log validation results
        if not validation_result["is_valid"]:
            self._logger.warning(
                f"Harmony format validation failed: {validation_result['warnings']}"
            )
        elif validation_result["warnings"]:
            self._logger.info(
                f"Harmony format warnings: {validation_result['warnings']}"
            )

        return validation_result

    def _format_with_harmony(self, messages: List[Message]) -> str:
        """Format messages using harmony encoding for proper GPT OSS format with enhanced validation."""
        try:
            # Load the harmony encoding
            enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

            # Convert messages to harmony format
            harmony_messages = self._convert_to_harmony_messages(messages)

            # Validate harmony format requirements before processing
            last_message_content = ""
            if messages and messages[-1].content:
                for content in messages[-1].content:
                    if content.type == MessageContentType.TEXT and content.text:
                        last_message_content += content.text

            if last_message_content:
                validation_result = self._validate_harmony_format(last_message_content)
                if validation_result["recommendations"]:
                    self._logger.info(
                        f"Harmony format recommendations: {validation_result['recommendations']}"
                    )

            # Create conversation
            convo = HarmonyConversation.from_messages(harmony_messages)

            # Render conversation for completion
            tokens = enc.render_conversation_for_completion(
                convo, HarmonyRole.ASSISTANT
            )

            # Convert tokens to string (decode)
            formatted_result = enc.decode(tokens)

            self._logger.debug("Successfully formatted messages with harmony encoding")
            return formatted_result

        except Exception as e:
            self._logger.error(f"Failed to format with harmony: {e}")
            # Fallback to standard formatting with warning
            self._logger.warning(
                "Falling back to standard formatting (may not work correctly with GPT OSS)"
            )
            return self._format_standard_messages(messages)

    def _format_standard_messages(self, messages: List[Message]) -> str:
        """Fallback standard message formatting (not recommended for GPT OSS)."""
        formatted_parts = []

        for message in messages:
            text_content = ""
            if message.content:
                for content in message.content:
                    if content.type == MessageContentType.TEXT and content.text:
                        text_content += content.text

            if message.role == MessageRole.SYSTEM:
                formatted_parts.append(
                    f"<|start|>system<|message|>{text_content}<|end|>"
                )
            elif message.role == MessageRole.USER:
                formatted_parts.append(f"<|start|>user<|message|>{text_content}<|end|>")
            elif message.role == MessageRole.ASSISTANT:
                formatted_parts.append(
                    f"<|start|>assistant<|message|>{text_content}<|end|>"
                )

        # Add assistant start for completion
        formatted_parts.append("<|start|>assistant")

        return "\n".join(formatted_parts)

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

        # LLM will be initialized lazily in agent node
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
        """Harmony-format agent node with timeout + iteration safeguards."""
        _ = config

        # Iteration guard
        if state.current_iteration >= state.max_iterations:
            msg = f"Maximum iterations ({state.max_iterations}) reached. Stopping to prevent infinite loops."
            return {
                "messages": [coerce_to_langchain_message_dict(AIMessage(content=msg))],
                "current_iteration": state.current_iteration + 1,
            }

        try:
            # Lazy initialize LLM
            if self.llm is None:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path)
                if self.llm is None:  # safety
                    raise RuntimeError("LLM failed to initialize")

            # Convert LangGraph state messages -> internal messages (for harmony)
            internal_messages: List[Message] = []
            for lc_msg in state.messages:
                internal_messages.append(from_lc_message(lc_msg))  # type: ignore

            # Determine dynamic timeout (reuse earlier heuristic)
            is_deep = self._should_use_extended_timeout(internal_messages)
            timeout_seconds = (
                min(self.circuit_config.deep_research_timeout or 180.0, 180.0)
                if is_deep
                else min(self.circuit_config.base_timeout or 90.0, 90.0)
            )

            # Harmony formatting
            formatted_prompt = self._format_with_harmony(internal_messages)
            self._logger.debug(
                f"Harmony formatted prompt preview: {formatted_prompt[:200]}..."
            )

            # Invoke model with timeout
            import asyncio

            response = await asyncio.wait_for(
                self._invoke_with_harmony_format(formatted_prompt),
                timeout=timeout_seconds,
            )

            preview = (
                str(response.content)[:80]
                if hasattr(response, "content")
                else str(response)[:80]
            )
            self._logger.info(
                f"GPT OSS harmony response len={len(str(response.content)) if hasattr(response,'content') else 'n/a'} preview={preview}"
            )

            return {
                "messages": [coerce_to_langchain_message_dict(response)],
                "current_iteration": state.current_iteration + 1,
            }

        except asyncio.TimeoutError:
            warn = f"Request timed out after {timeout_seconds:.1f}s (harmony processing exceeded limit)."
            self._logger.warning(warn)
            return {
                "messages": [coerce_to_langchain_message_dict(AIMessage(content=warn))],
                "current_iteration": state.current_iteration + 1,
            }
        except Exception as e:
            err = f"Error in GPT OSS harmony agent node: {e}"
            self._logger.error(err, exc_info=True)
            return {
                "messages": [coerce_to_langchain_message_dict(AIMessage(content=err))],
                "current_iteration": state.current_iteration + 1,
            }

    async def _invoke_with_harmony_format(self, formatted_prompt: str) -> AIMessage:
        """Invoke the model with harmony-formatted prompt."""
        from langchain_core.messages import HumanMessage

        if self.llm is None:
            self._logger.error("LLM is not initialized")
            return AIMessage(content="Error: LLM is not initialized")

        try:
            # Use standard invoke method with harmony-formatted prompt
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])

            # Ensure we return an AIMessage
            if isinstance(response, AIMessage):
                return response
            else:
                # Convert BaseMessage to AIMessage if needed
                return AIMessage(
                    content=(
                        str(response.content)
                        if hasattr(response, "content")
                        else str(response)
                    )
                )

        except Exception as e:
            self._logger.error(f"Failed to invoke model with harmony format: {e}")
            # Return error message as AIMessage
            return AIMessage(content=f"Error: {str(e)}")
