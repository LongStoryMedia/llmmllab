"""
OpenAI GPT OSS 20B Abliterated Uncensored pipeline implementation.
Based on the QwenLangGraphPipe with optimizations for the OpenAI GPT OSS 20B model.
"""

import os
import logging
import re
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
    build_lc_messages,
)
from utils.message import from_lc_message
from models.lang_chain_message import LangChainMessage
from ..base_langgraph import CircuitBreakerConfig
from ..llamacpp.base_llamacpp import BaseLlamaCppPipeline
from ..context_manager import ContextManager


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

        # Initialize harmony channel state
        self._reset_harmony_state()

        # Store current tools for agent node access
        self._current_tools: Optional[List[BaseTool]] = None

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

    def _reset_harmony_state(self) -> None:
        """Reset harmony channel tracking state."""
        self.harmony_buffer: str = ""
        self.current_channel: str = "final"  # Default to final channel
        self.in_analysis_channel: bool = False
        self.analysis_complete: bool = False
        self.detected_channels: set = set()

    def reset_streaming_state(self) -> None:
        """Reset streaming state (overrides base method)."""
        super().reset_streaming_state()
        self._reset_harmony_state()

    def process_streaming_token(self, content: str) -> Optional[ChatResponse]:
        """Process streaming token with harmony channel detection (overrides base method)."""
        self.harmony_buffer += content
        analysis_marker = "<|channel|>analysis<|message|>"
        end_marker = "<|end|>"

        # Detect harmony channel markers
        if (
            analysis_marker in self.harmony_buffer
            and not self.analysis_complete
            and not self.in_analysis_channel
        ):
            self.in_analysis_channel = True
            self.current_channel = "analysis"
            self.detected_channels.add("analysis")
            # Remove the analysis marker from what we return
            return None

        if end_marker in content and self.in_analysis_channel:
            self.in_analysis_channel = False
            self.analysis_complete = True
            self.current_channel = "final"
            # Remove the closing analysis marker
            return None

        if self.in_analysis_channel:
            return self._create_thinking_response(content)
        if self.current_channel == "final":
            return self._create_streaming_response(content)

        return None

    def finalize_streaming(self) -> Optional[ChatResponse]:
        """Finalize streaming and return any remaining content (overrides base method)."""
        if self.harmony_buffer:
            if self.current_channel == "analysis" or self.in_analysis_channel:
                # Return thinking content
                message = Message(
                    role=MessageRole.ASSISTANT,
                    thinking=self.harmony_buffer,
                    content=[],
                )
                self._reset_harmony_state()
                return ChatResponse(message=message, done=True)
            else:
                # Return regular content
                message = Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=self.harmony_buffer,
                        )
                    ],
                )
                self._reset_harmony_state()
                return ChatResponse(message=message, done=True)

        self._reset_harmony_state()
        return None

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
        
        # CRITICAL: Fix stop sequences for harmony format
        # The harmony format requires <|end|> markers to transition between channels
        # Default stop sequences include <|end|> which prevents proper harmony format generation
        optimal_params["stop"] = ["<|im_end|>", "<|endoftext|>"]  # Removed <|end|>
        self._logger.info("GPT OSS: Configured stop sequences for harmony format compatibility")

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
        if optimal_params.get("stop") is not None:
            self.profile.parameters.stop = optimal_params["stop"]

        self._logger.info(
            "GPT OSS pipeline applying optimal parameters for harmony format compatibility"
        )

        # Call parent initialization with optimized parameters
        await super()._initialize_llm(gguf_path, tools)

    def _create_system_prompt(self, tools: Optional[List[BaseTool]] = None) -> str:
        """Create harmony-compatible system prompt for OpenAI GPT OSS 20B."""

        # Base system prompt with harmony format requirements
        base_prompt = (
            self.profile.system_prompt
            or """You are a large language model with non-restrictive policies.
Knowledge cutoff: 2024-06
Current date: 2025-09-17

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

        # Add tool information following OpenAI Harmony documentation
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            tool_info = "\n".join(tool_descriptions)
            base_prompt += f"""

Available tools:
{tool_info}

TOOL CALLING INSTRUCTIONS (CRITICAL - OpenAI Harmony Format):

MANDATORY RULE: When user requests web search, current information, or research, you MUST use the web_search tool.

EXACT FORMAT REQUIRED:
<|channel|>commentary to=functions <|constrain|>json
<|message|>{{"name": "EXACT_TOOL_NAME", "arguments": {{"param": "value"}}}}

CRITICAL EXAMPLES:

User says "search the web" or "find information" → USE web_search:
<|channel|>commentary to=functions <|constrain|>json
<|message|>{{"name": "web_search", "arguments": {{"query": "your search terms here"}}}}

EXACT TOOL NAMES (DO NOT MODIFY):
- web_search (for any web/internet searches - NOT "search" or "websearch")

MANDATORY RULES:
1. ALWAYS use web_search when user asks to search web, find information, or get current data
2. Use EXACT tool names - "web_search" NOT "search", "websearch", or variations
3. Complete the entire JSON structure - do not truncate
4. After tool execution, provide final response with results

FAILURE TO USE TOOLS WHEN REQUESTED IS NOT ACCEPTABLE."""

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

    def _format_with_harmony(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Format messages using harmony encoding for proper GPT OSS format with custom system prompt."""
        try:
            # Load the harmony encoding
            enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

            # Create custom system prompt with harmony format and tool information
            system_prompt = self._create_system_prompt(tools)

            # Convert messages to harmony format
            harmony_messages = self._convert_to_harmony_messages(messages)

            # Replace or insert system message with our custom prompt
            # Use our carefully crafted system prompt instead of empty content
            custom_system_message = HarmonyMessage.from_role_and_content(
                HarmonyRole.SYSTEM, system_prompt
            )

            # Find and replace existing system message, or insert at beginning
            found_system = False
            for i, msg in enumerate(harmony_messages):
                if hasattr(msg, "role") and msg.role == HarmonyRole.SYSTEM:
                    harmony_messages[i] = custom_system_message
                    found_system = True
                    break

            if not found_system:
                harmony_messages.insert(0, custom_system_message)

            # Validate harmony format requirements
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

            # Convert tokens to string (decode) and inject our system prompt
            formatted_result = enc.decode(tokens)

            # Manual injection of system prompt at the beginning of the formatted result
            # This ensures our system prompt is used instead of the default
            if formatted_result.startswith("<|start|>system<|message|>"):
                # Replace everything between system tags with our prompt
                import re

                pattern = r"(<\|start\|>system<\|message\|>).*?(<\|start\|>)"
                replacement = f"\\1{system_prompt}\\2"
                formatted_result = re.sub(
                    pattern, replacement, formatted_result, flags=re.DOTALL
                )

            self._logger.debug(
                "Successfully formatted messages with harmony encoding and custom system prompt"
            )
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



    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create LangGraph with optimized caching and timeout protection."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Store tools for later access in agent node
        self._current_tools = tools

        # LLM will be initialized lazily in agent node
        workflow = StateGraph(LangGraphState)
        workflow.add_node("agent", self._agent_node)

        if tools:
            # Create a wrapper around ToolNode to handle our LangChainMessage format
            tool_node = ToolNode(tools)
            
            async def tool_node_wrapper(state: LangGraphState, config=None):
                """Wrapper around ToolNode to handle LangChainMessage conversion."""
                if not state.messages:
                    return {"messages": state.messages}

                last_message = state.messages[-1]
                
                # Convert LangChainMessage to AIMessage for ToolNode
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    # Create an AIMessage from our LangChainMessage
                    ai_message = AIMessage(
                        content=last_message.content,
                        tool_calls=last_message.tool_calls
                    )
                    
                    # Create a temporary state with AIMessage for ToolNode
                    temp_state = {
                        "messages": state.messages[:-1] + [ai_message]
                    }
                    
                    # Execute tools using standard ToolNode
                    result = await tool_node.ainvoke(temp_state, config)
                    
                    # Convert ToolMessage results back to LangChainMessage format
                    converted_messages = []
                    for msg in result["messages"]:
                        if hasattr(msg, '__class__') and msg.__class__.__name__ == "ToolMessage":
                            # Convert ToolMessage to LangChainMessage
                            tool_msg = LangChainMessage(
                                content=msg.content,
                                type="tool",
                                name=getattr(msg, 'name', None),
                                id=getattr(msg, 'tool_call_id', None),
                                tool_calls=None
                            )
                            converted_messages.append(tool_msg)
                        else:
                            # Keep other message types as-is or convert as needed
                            converted_messages.append(msg)
                    
                    return {"messages": state.messages + converted_messages}
                
                return {"messages": state.messages}

            workflow.add_node("tools", tool_node_wrapper)
            
            # Use standard tools_condition but check our LangChainMessage format
            def custom_tools_condition(state: LangGraphState):
                """Check for tool calls in our LangChainMessage format."""
                if not state.messages:
                    return END
                
                last_message = state.messages[-1]
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    self._logger.debug(f"tools_condition: Found {len(last_message.tool_calls)} tool calls")
                    return "tools"
                
                self._logger.debug("tools_condition: No tool calls found, routing to END")
                return END
            
            workflow.add_conditional_edges(
                "agent", custom_tools_condition, {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        workflow.add_edge(START, "agent")
        compiled_graph = workflow.compile(checkpointer=self.memory)
        self.graph_cache[tool_signature] = compiled_graph
        return compiled_graph

    def _parse_harmony_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse harmony format tool calls and convert to LangChain format following OpenAI Harmony docs."""
        import json
        import re

        tool_calls = []
        seen_json_strings = set()  # Prevent duplicates

        # Look for commentary channel with tool calls - multiple patterns for robustness:
        patterns = [
            # Standard format: <|channel|>commentary to=functions <|constrain|>json<|message|>
            r"<\|channel\|>commentary\s+to=functions\s+<\|constrain\|>json<\|message\|>(.+?)(?=<\|end\|>|<\|channel\|>|$)",
            # Format with newline: <|channel|>commentary to=functions <|constrain|>json\n<|message|>
            r"<\|channel\|>commentary\s+to=functions\s+<\|constrain\|>json\s*\n\s*<\|message\|>(.+?)(?=<\|end\|>|<\|channel\|>|$)",
            # Flexible format: handle optional <|message|> tag
            r"<\|channel\|>commentary\s+to=functions\s+<\|constrain\|>json\s*(?:<\|message\|>)?\s*(\{[^}]*\})"
        ]
        
        matches = []
        for pattern in patterns:
            pattern_matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            matches.extend(pattern_matches)
            if pattern_matches:
                self._logger.debug(f"Pattern matched {len(pattern_matches)} tool calls: {pattern}")
        
        # Also try a simple JSON extraction as fallback
        if not matches:
            # Look for any JSON objects that might be tool calls
            json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            if json_matches:
                self._logger.debug(f"Fallback JSON pattern found {len(json_matches)} potential tool calls")
                matches.extend(json_matches)

        # Debug: Log what we're trying to parse
        self._logger.debug(f"Looking for harmony tool calls in content (first 500 chars): {content[:500]}")
        self._logger.debug(f"Found {len(matches)} potential tool call matches")

        for i, match in enumerate(matches):
            try:
                # Clean up the JSON string - remove any trailing text after the JSON
                json_str = match.strip()
                self._logger.debug(f"Processing match {i}: {json_str[:200]}...")

                # Skip if we've already processed this JSON string
                if json_str in seen_json_strings:
                    self._logger.debug(f"Skipping duplicate JSON string")
                    continue
                seen_json_strings.add(json_str)

                # Try to find JSON object boundaries
                if "{" in json_str:
                    start_idx = json_str.find("{")
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for j, char in enumerate(json_str[start_idx:], start_idx):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = j + 1
                                break
                    json_str = json_str[start_idx:end_idx]
                    self._logger.debug(f"Extracted JSON: {json_str}")

                # Parse the JSON
                tool_call_data = json.loads(json_str)
                self._logger.debug(f"Parsed tool call data: {tool_call_data}")

                # Convert to exact LangGraph/LangChain format
                if "name" in tool_call_data:
                    args = tool_call_data.get("arguments") or tool_call_data.get(
                        "args", {}
                    )
                    # Use exact format that LangGraph ToolNode expects
                    tool_call = {
                        "name": tool_call_data["name"],
                        "args": args,
                        "id": f"call_{len(tool_calls)}_{tool_call_data['name']}",
                        "type": "tool_call"  # Required by LangGraph ToolNode
                    }
                    tool_calls.append(tool_call)
                    self._logger.debug(f"Successfully parsed harmony tool call: {tool_call}")
                else:
                    self._logger.warning(f"Tool call data missing 'name' field: {tool_call_data}")

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self._logger.warning(
                    f"Failed to parse tool call from: {match[:100]}... Error: {e}"
                )
                continue

        return tool_calls

    def _extract_final_content(self, content: str) -> str:
        """Extract content from final channel or return the content as-is."""
        # Look for final channel content
        final_pattern = (
            r"<\|channel\|>final<\|message\|>(.+?)(?=<\|end\|>|<\|channel\|>|$)"
        )
        final_match = re.search(final_pattern, content, re.DOTALL | re.IGNORECASE)

        if final_match:
            return final_match.group(1).strip()

        # If no final channel, extract meaningful content before tool calls
        # Look for content before the first commentary channel
        commentary_pattern = r"^(.*?)(?=<\|channel\|>commentary\s+to=functions|$)"
        before_commentary = re.search(
            commentary_pattern, content, re.DOTALL | re.IGNORECASE
        )

        if before_commentary:
            preliminary_content = before_commentary.group(1).strip()

            # Clean up harmony tags from preliminary content
            cleaned = re.sub(r"<\|[^>]+\|>", "", preliminary_content)

            # If we have substantial content, return it
            if len(cleaned.strip()) > 10:  # Minimum meaningful content
                return cleaned.strip()

        # Fallback: return a placeholder for tool calls
        return "I need to use tools to help with your request."

    async def _agent_node(self, state: LangGraphState, config=None) -> Dict[str, Any]:
        """Agent node following LangGraph standard patterns with harmony formatting."""
        _ = config

        # Debug: Log the state we receive
        self._logger.debug(
            f"Agent node received state with {len(state.messages)} messages, iteration {state.current_iteration}"
        )
        if state.messages:
            last_msg = state.messages[-1]
            self._logger.debug(
                f"Last message: type={getattr(last_msg, 'type', 'unknown')}, content_preview={str(getattr(last_msg, 'content', ''))[:100]}..."
            )

        # Iteration guard
        if state.current_iteration >= state.max_iterations:
            msg = f"Maximum iterations ({state.max_iterations}) reached. Stopping to prevent infinite loops."
            return {
                "messages": [LangChainMessage(
                    content=msg,
                    type="ai",
                    tool_calls=None,
                    additional_kwargs={},
                    response_metadata={}
                )],
                "current_iteration": state.current_iteration + 1,
            }

        try:
            # Initialize LLM if not done yet
            if self.llm is None:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path)

            # Convert to internal messages for harmony formatting
            internal_messages: List[Message] = []
            for lc_msg in state.messages:
                internal_messages.append(from_lc_message(lc_msg))

            # Get current tools and format prompt with harmony
            current_tools = self._get_current_tools()
            formatted_prompt = self._format_with_harmony(
                internal_messages, current_tools
            )

            self._logger.debug(
                f"Harmony formatted prompt preview: {formatted_prompt[:200]}..."
            )

            # Use the base class streaming method with harmony-formatted prompt
            # This leverages the adaptive controls and timeout handling
            from langchain_core.messages import HumanMessage

            # Create a single message with the harmony-formatted prompt
            harmony_messages = [HumanMessage(content=formatted_prompt)]

            # Use base class adaptive streaming with proper timeout
            response = await self._stream_with_adaptive_controls(
                harmony_messages, is_tool_generation=False
            )

            # The response from _stream_with_adaptive_controls is already an AIMessage
            # Check if it contains tool calls in harmony format and convert them
            response_content = (
                str(response.content) if hasattr(response, "content") else str(response)
            )

            # Parse any harmony tool calls and convert to LangChain format
            tool_calls = self._parse_harmony_tool_calls(response_content)
            final_content = self._extract_final_content(response_content)

            # Create properly formatted response as LangChainMessage
            if tool_calls:
                # When we have tool calls, the content might be minimal since the model
                # is expecting tool results. Use the full response content or a placeholder
                content_for_tool_call = (
                    final_content
                    if final_content.strip()
                    else "I need to search for this information."
                )
                formatted_response = LangChainMessage(
                    content=content_for_tool_call, 
                    tool_calls=tool_calls,
                    type="ai",
                    additional_kwargs={},
                    response_metadata={}
                )
                self._logger.info(
                    f"Converted {len(tool_calls)} harmony tool calls to LangChain format"
                )
                self._logger.debug(
                    f"Tool call content: {content_for_tool_call[:100]}..."
                )

                # Debug: Log the formatted response structure for tools_condition
                self._logger.debug(
                    f"LangChainMessage for tools_condition: content='{formatted_response.content[:50]}...' tool_calls={len(formatted_response.tool_calls) if formatted_response.tool_calls else 0}"
                )
                self._logger.debug(
                    f"Tool calls structure: {formatted_response.tool_calls}"
                )
            else:
                formatted_response = LangChainMessage(
                    content=final_content,
                    type="ai",
                    tool_calls=None,
                    additional_kwargs={},
                    response_metadata={}
                )

            preview = final_content[:80] if final_content else "No content"
            self._logger.info(
                f"GPT OSS response len={len(final_content)} preview={preview}"
            )

            # Log the actual tool calls for debugging
            if tool_calls:
                for i, tool_call in enumerate(tool_calls):
                    self._logger.debug(
                        f"Tool call {i}: {tool_call['name']} with args {tool_call['args']}"
                    )

            # Return the LangChainMessage directly
            return {
                "messages": [formatted_response],
                "current_iteration": state.current_iteration + 1,
            }

        except Exception as e:
            err = f"Error in GPT OSS agent node: {e}"
            self._logger.error(err, exc_info=True)
            # Return LangChainMessage for consistency
            return {
                "messages": [LangChainMessage(
                    content=err,
                    type="ai", 
                    tool_calls=None,
                    additional_kwargs={},
                    response_metadata={}
                )],
                "current_iteration": state.current_iteration + 1,
            }

    def _get_current_tools(self) -> Optional[List[BaseTool]]:
        """Get the current tools stored during graph creation."""
        return self._current_tools

    async def _invoke_with_harmony_format(self, formatted_prompt: str) -> AIMessage:
        """Invoke the model with harmony-formatted prompt."""
        from langchain_core.messages import HumanMessage

        if self.llm is None:
            self._logger.error("LLM is not initialized")
            return AIMessage(content="Error: LLM is not initialized")

        try:
            # Use standard invoke method with harmony-formatted prompt
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])

            # Return response directly - real-time harmony filtering handles cleanup
            if isinstance(response, AIMessage):
                return response
            else:
                # Convert BaseMessage to AIMessage if needed
                content = (
                    str(response.content)
                    if hasattr(response, "content")
                    else str(response)
                )
                return AIMessage(content=content)

        except Exception as e:
            self._logger.error(f"Failed to invoke model with harmony format: {e}")
            # Return error message as AIMessage
            return AIMessage(content=f"Error: {str(e)}")
