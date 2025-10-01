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
from langchain.agents import ToolNode, tools_condition

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
    coerce_to_lc_message,
)
from models.lang_chain_message import LangChainMessage
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
        if content is None:
            return None

        # Ensure buffer is initialized
        if not hasattr(self, "buffer") or self.buffer is None:
            self.buffer = ""

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

    def _create_thinking_response(self, content: str) -> Optional[ChatResponse]:
        """Create a response for thinking content."""
        try:
            # Accumulate thinking content
            if content is not None:
                if not hasattr(self, "think_content") or self.think_content is None:
                    self.think_content = ""
                self.think_content += content
            # Don't return anything for thinking - it will be added to the final message
            return None
        except Exception as e:
            self._logger.error(f"Error creating thinking response: {e}")
            return None

    def _create_streaming_response(self, content: str) -> Optional[ChatResponse]:
        """Create a streaming response for regular content."""
        try:
            if not content:
                return None

            message_content = [
                MessageContent(type=MessageContentType.TEXT, text=content)
            ]
            message = Message(
                role=MessageRole.ASSISTANT,
                content=message_content,
                thinking=None,  # Thinking will be added in finalize_streaming
            )
            return ChatResponse(message=message, done=False)
        except Exception as e:
            self._logger.error(f"Error creating streaming response: {e}")
            return None

    def finalize_streaming(self) -> Optional[ChatResponse]:
        """Finalize streaming and return any remaining content (overrides base method)."""
        try:
            # Create final message with thinking content and any remaining buffer
            thinking = self.think_content if self.think_content else None
            content = []

            if self.buffer and not self.in_think_tag:
                content = [
                    MessageContent(type=MessageContentType.TEXT, text=self.buffer)
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
        except Exception as e:
            self._logger.error(f"Error in finalize_streaming: {e}", exc_info=True)
            return None

    # llama.cpp initialization inherited from BaseLlamaCppPipeline

    # Perplexity / repetition guard logic now inherited from BaseLangGraphPipeline

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt for Qwen3 thinking model with tool calling support."""
        base_prompt = (
            self.profile.system_prompt
            or """You are a helpful AI assistant. Think through problems step by step using <think>...</think> tags.

Your thinking process will be captured separately from your response. Use the thinking space to:
- Analyze the problem
- Consider different approaches  
- Work through the logic

Then provide your clear, direct answer outside the thinking tags."""
        )

        # Add tool information if available with Qwen3 native format
        if tools:
            tool_descriptions = []
            for tool in tools:
                # Create tool signature in JSON format for the prompt
                tool_signature = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }

                # Add specific parameter definitions for known tools
                if tool.name == "web_search":
                    tool_signature["parameters"]["properties"] = {
                        "query": {"type": "string", "description": "The search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    }
                    tool_signature["parameters"]["required"] = ["query"]
                elif tool.name == "memory_retrieval":
                    tool_signature["parameters"]["properties"] = {
                        "tool_input": {
                            "type": "array",
                            "description": "List of embeddings for retrieval",
                        }
                    }
                    tool_signature["parameters"]["required"] = ["tool_input"]
                elif tool.name == "summarization":
                    tool_signature["parameters"]["properties"] = {
                        "tool_input": {
                            "type": "array",
                            "description": "List of messages to summarize",
                        }
                    }
                    tool_signature["parameters"]["required"] = ["tool_input"]
                else:
                    # Generic parameter for unknown tools
                    tool_signature["parameters"]["properties"] = {
                        "query": {"type": "string", "description": "Input for the tool"}
                    }
                    tool_signature["parameters"]["required"] = ["query"]

                import json

                tool_descriptions.append(json.dumps(tool_signature, indent=2))

            # Format tools for Qwen native function calling (Hermes-style)
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }

                # Add specific parameter definitions for known tools
                if tool.name == "web_search":
                    formatted_tool["function"]["parameters"]["properties"] = {
                        "query": {"type": "string", "description": "The search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    }
                    formatted_tool["function"]["parameters"]["required"] = ["query"]
                elif tool.name == "memory_retrieval":
                    formatted_tool["function"]["parameters"]["properties"] = {
                        "tool_input": {
                            "type": "array",
                            "description": "List of embeddings for retrieval",
                        }
                    }
                    formatted_tool["function"]["parameters"]["required"] = [
                        "tool_input"
                    ]
                elif tool.name == "summarization":
                    formatted_tool["function"]["parameters"]["properties"] = {
                        "tool_input": {
                            "type": "array",
                            "description": "List of messages to summarize",
                        }
                    }
                    formatted_tool["function"]["parameters"]["required"] = [
                        "tool_input"
                    ]
                else:
                    # Generic parameter for unknown tools including dynamic ones
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        # Use the tool's actual schema if available
                        schema = tool.args_schema
                        if hasattr(schema, "model_json_schema"):
                            schema_dict = schema.model_json_schema()  # type: ignore
                            formatted_tool["function"]["parameters"] = schema_dict
                        else:
                            # Fallback to generic query parameter
                            formatted_tool["function"]["parameters"]["properties"] = {
                                "query": {
                                    "type": "string",
                                    "description": "Input for the tool",
                                }
                            }
                            formatted_tool["function"]["parameters"]["required"] = [
                                "query"
                            ]
                    else:
                        # Fallback to generic query parameter
                        formatted_tool["function"]["parameters"]["properties"] = {
                            "query": {
                                "type": "string",
                                "description": "Input for the tool",
                            }
                        }
                        formatted_tool["function"]["parameters"]["required"] = ["query"]

                formatted_tools.append(formatted_tool)

            # Store tools for template processing
            self._available_tools = formatted_tools

        return base_prompt

    def _parse_qwen_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse Qwen function calls from generated content - supports multiple formats."""
        import json
        import re

        tool_calls = []

        # Pattern 1a: Look for proper Qwen function call format (arguments as string)
        function_call_pattern_str = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*"([^"]+)"\s*\}'
        function_matches_str = re.findall(function_call_pattern_str, content, re.DOTALL)

        for i, (name, args_str) in enumerate(function_matches_str):
            try:
                # Parse the arguments JSON string
                args = json.loads(args_str)
                formatted_call = {
                    "name": name,
                    "args": args,
                    "id": f"call_{i}_{name}",
                    "type": "tool_call",
                }
                tool_calls.append(formatted_call)
                self._logger.debug(
                    f"Parsed Qwen function_call (string args): {formatted_call}"
                )
            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(
                    f"Failed to parse function_call arguments '{args_str}': {e}"
                )
                continue

        # Pattern 1b: Look for proper Qwen function call format (arguments as object)
        if not tool_calls:
            function_call_pattern_obj = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\s*\}'
            function_matches_obj = re.findall(
                function_call_pattern_obj, content, re.DOTALL
            )

            for i, (name, args_str) in enumerate(function_matches_obj):
                try:
                    args = json.loads(args_str)
                    formatted_call = {
                        "name": name,
                        "args": args,
                        "id": f"call_{i}_{name}",
                        "type": "tool_call",
                    }
                    tool_calls.append(formatted_call)
                    self._logger.debug(
                        f"Parsed Qwen function_call (object args): {formatted_call}"
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse function_call arguments '{args_str}': {e}"
                    )
                    continue

        # Pattern 2: Look for <tool_call> XML tags (custom format - for backwards compatibility)
        if not tool_calls:
            tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
            matches = re.findall(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)

            for i, match in enumerate(matches):
                try:
                    # Parse the JSON content - don't strip here to preserve formatting
                    tool_data = json.loads(match)

                    if "name" in tool_data:
                        formatted_call = {
                            "name": tool_data["name"],
                            "args": tool_data.get("arguments", {}),
                            "id": f"call_{i}_{tool_data['name']}",
                            "type": "tool_call",
                        }
                        tool_calls.append(formatted_call)
                        self._logger.debug(f"Parsed XML tool call: {formatted_call}")
                    else:
                        self._logger.warning(
                            f"Tool call missing 'name' field: {match[:100]}..."
                        )

                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse XML tool call from: {match[:100]}... Error: {e}"
                    )
                    continue

        # Pattern 3: Look for mixed function_call tags (what we see in the logs)
        if not tool_calls:
            mixed_pattern = (
                r"<function_call>\s*(\{.*?\})\s*</(?:function_call|FunctionCall)>"
            )
            mixed_matches = re.findall(
                mixed_pattern, content, re.DOTALL | re.IGNORECASE
            )

            for i, match in enumerate(mixed_matches):
                try:
                    tool_data = json.loads(match)

                    if "name" in tool_data:
                        formatted_call = {
                            "name": tool_data["name"],
                            "args": tool_data.get("arguments", {}),
                            "id": f"call_{i}_{tool_data['name']}",
                            "type": "tool_call",
                        }
                        tool_calls.append(formatted_call)
                        self._logger.debug(
                            f"Parsed mixed function_call: {formatted_call}"
                        )
                    else:
                        self._logger.warning(
                            f"Mixed function call missing 'name' field: {match[:100]}..."
                        )

                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse mixed function call from: {match[:100]}... Error: {e}"
                    )
                    continue

        # Pattern 4: Legacy JSON blocks (final fallback)
        if not tool_calls:
            self._logger.debug(
                "No function calls found, checking for legacy JSON blocks..."
            )

            json_pattern = r"```json\s*(\{.*?\})\s*```"
            json_matches = re.findall(json_pattern, content, re.DOTALL | re.IGNORECASE)

            for match in json_matches:
                try:
                    data = json.loads(match)
                    if "tool_calls" in data:
                        for i, tool_call in enumerate(data["tool_calls"]):
                            if "name" in tool_call:
                                formatted_call = {
                                    "name": tool_call["name"],
                                    "args": tool_call.get("arguments", {}),
                                    "id": f"call_{i}_{tool_call['name']}",
                                    "type": "tool_call",
                                }
                                tool_calls.append(formatted_call)
                                self._logger.debug(
                                    f"Parsed legacy tool call: {formatted_call}"
                                )
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(f"Failed to parse legacy tool call: {e}")
                    continue

        if tool_calls:
            self._logger.info(
                f"Successfully parsed {len(tool_calls)} tool calls from content"
            )
        else:
            self._logger.warning("No tool calls found in content")

        return tool_calls

    def _clean_tool_calls_from_content(self, content: str) -> str:
        """Remove tool call patterns from content to get clean user-facing text."""
        import re

        # Remove function_call JSON patterns (proper Qwen format)
        func_call_pattern = (
            r'"function_call":\s*\{\s*"name":\s*"[^"]+",\s*"arguments":\s*"[^"]+"\s*\}'
        )
        content = re.sub(func_call_pattern, "", content, flags=re.DOTALL)

        # Remove <tool_call> XML tags (custom format)
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        content = re.sub(
            tool_call_pattern, "", content, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove mixed function_call tags (what we see in logs)
        mixed_pattern = r"<function_call>\s*\{.*?\}\s*</(?:function_call|FunctionCall)>"
        content = re.sub(mixed_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove legacy JSON code blocks that contain tool_calls
        json_pattern = r'```json\s*\{.*?"tool_calls".*?\}\s*```'
        content = re.sub(json_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove standalone function call patterns
        func_pattern = (
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        )
        content = re.sub(func_pattern, "", content)

        # Clean up extra whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        return content

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
            # Create ToolNode with enhanced debugging
            tool_node = ToolNode(tools)

            # Wrap ToolNode to add debugging and handle LangChainMessage conversion
            def debug_tool_node(state: LangGraphState):
                """Debug wrapper for ToolNode execution."""
                try:
                    self._logger.error(
                        f"QwenMoE debug_tool_node: ENTRY - state type: {type(state)}"
                    )
                    self._logger.error(
                        f"QwenMoE debug_tool_node: ENTRY - state.messages length: {len(state.messages) if hasattr(state, 'messages') and state.messages else 'NO MESSAGES'}"
                    )

                    if hasattr(state, "messages") and state.messages:
                        last_message = state.messages[-1]
                        self._logger.error(
                            f"QwenMoE debug_tool_node: ENTRY - last_message type: {type(last_message)}"
                        )
                        if isinstance(last_message, dict):
                            self._logger.error(
                                f"QwenMoE debug_tool_node: ENTRY - dict message keys: {list(last_message.keys())}"
                            )
                            if "tool_calls" in last_message:
                                self._logger.error(
                                    f"QwenMoE debug_tool_node: ENTRY - dict tool_calls: {last_message['tool_calls']}"
                                )

                    # Convert dict messages to LangChain BaseMessages for ToolNode
                    converted_messages = []
                    for i, msg in enumerate(state.messages):
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Converting message {i}: type={type(msg)}"
                        )
                        if hasattr(msg, "type"):
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Message {i} has type field: {msg.type}"
                            )
                        if hasattr(msg, "tool_calls"):
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Message {i} has tool_calls field: {msg.tool_calls}"
                            )
                        converted_msg = coerce_to_lc_message(msg)
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Converted message {i} to: type={type(converted_msg)}"
                        )
                        if (
                            hasattr(converted_msg, "tool_calls")
                            and converted_msg.tool_calls
                        ):
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Message {i} has tool_calls: {len(converted_msg.tool_calls)}"
                            )
                        converted_messages.append(converted_msg)

                    converted_state = state.copy()
                    converted_state.messages = converted_messages

                    self._logger.error(
                        f"QwenMoE debug_tool_node: BEFORE TOOLNODE - last message type: {type(converted_state.messages[-1])}"
                    )

                    result = tool_node.invoke(converted_state)
                    self._logger.error(
                        f"QwenMoE debug_tool_node: SUCCESS - ToolNode executed"
                    )
                    self._logger.error(
                        f"QwenMoE debug_tool_node: Result type: {type(result)}"
                    )

                    # Check what's in the result
                    if isinstance(result, dict):
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Result dict keys: {list(result.keys())}"
                        )
                        if "messages" in result:
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Result has messages: {len(result['messages'])} messages"
                            )
                            for i, msg in enumerate(result["messages"]):
                                self._logger.error(
                                    f"QwenMoE debug_tool_node: Result message {i} type: {type(msg)}"
                                )
                    elif hasattr(result, "messages"):
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Result has messages attr: {len(result.messages)} messages"
                        )
                        for i, msg in enumerate(result.messages):
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Result message {i} type: {type(msg)}"
                            )
                    else:
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Result has no messages"
                        )

                    # Convert ToolMessage results back to dicts for state consistency
                    if (
                        isinstance(result, dict)
                        and "messages" in result
                        and result["messages"]
                    ):
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Converting {len(result['messages'])} result messages"
                        )
                        converted_messages = []
                        for i, msg in enumerate(result["messages"]):
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Converting result message {i} type: {type(msg)}"
                            )
                            converted_msg = coerce_to_langchain_message_dict(msg)
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Converted result message {i} to: {type(converted_msg)}"
                            )
                            converted_messages.append(converted_msg)

                        # Create a completely new result with all messages converted
                        new_result = {
                            "messages": state.messages
                            + converted_messages,  # Original messages + new tool messages
                        }
                        # Copy other attributes from result if any
                        for key, value in result.items():
                            if key != "messages":
                                new_result[key] = value

                        self._logger.error(
                            f"QwenMoE debug_tool_node: Final new_result messages types: {[type(m) for m in new_result['messages']]}"
                        )
                        self._logger.error(
                            f"QwenMoE debug_tool_node: Final new_result messages length: {len(new_result['messages'])}"
                        )
                        return new_result
                    elif hasattr(result, "messages"):
                        messages_attr = getattr(result, "messages", None)
                        if messages_attr:
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Converting {len(messages_attr)} result messages (attr)"
                            )
                            converted_messages = []
                            for i, msg in enumerate(messages_attr):
                                self._logger.error(
                                    f"QwenMoE debug_tool_node: Converting result message {i} type: {type(msg)}"
                                )
                                converted_msg = coerce_to_langchain_message_dict(msg)
                                self._logger.error(
                                    f"QwenMoE debug_tool_node: Converted result message {i} to: {type(converted_msg)}"
                                )
                                converted_messages.append(converted_msg)

                            # Create new state with combined messages
                            new_state = state.copy()
                            new_state.messages = state.messages + converted_messages
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Final new_state messages types: {[type(m) for m in new_state.messages]}"
                            )
                            self._logger.error(
                                f"QwenMoE debug_tool_node: Final new_state messages length: {len(new_state.messages)}"
                            )
                            return new_state

                    return result

                except Exception as e:
                    self._logger.error(f"QwenMoE debug_tool_node: FAILED - {e}")
                    # Let's also log the stack trace
                    import traceback

                    self._logger.error(
                        f"QwenMoE debug_tool_node: TRACEBACK - {traceback.format_exc()}"
                    )
                    raise

            workflow.add_node("tools", debug_tool_node)

            # Use custom tools_condition to handle our LangChainMessage format
            def custom_tools_condition(state: LangGraphState):
                """Check for tool calls in our LangChainMessage format."""
                self._logger.debug(
                    f"QwenMoE tools_condition: state type: {type(state)}"
                )
                self._logger.debug(
                    f"QwenMoE tools_condition: state has messages: {hasattr(state, 'messages')}"
                )

                if not hasattr(state, "messages") or not state.messages:
                    self._logger.debug(
                        "QwenMoE tools_condition: No messages in state, routing to END"
                    )
                    return END

                self._logger.debug(
                    f"QwenMoE tools_condition: state.messages length: {len(state.messages)}"
                )
                last_message = state.messages[-1]
                self._logger.debug(
                    f"QwenMoE tools_condition: last_message type: {type(last_message)}"
                )
                self._logger.debug(
                    f"QwenMoE tools_condition: last_message has tool_calls: {hasattr(last_message, 'tool_calls')}"
                )

                if hasattr(last_message, "tool_calls"):
                    self._logger.debug(
                        f"QwenMoE tools_condition: tool_calls value: {last_message.tool_calls}"
                    )
                    self._logger.debug(
                        f"QwenMoE tools_condition: tool_calls type: {type(last_message.tool_calls)}"
                    )
                    if last_message.tool_calls:
                        self._logger.info(
                            f"QwenMoE tools_condition: Found {len(last_message.tool_calls)} tool calls - routing to tools"
                        )
                        return "tools"

                self._logger.info(
                    "QwenMoE tools_condition: No tool calls found, routing to END"
                )
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

    async def _agent_node(self, state: LangGraphState, config=None) -> Dict[str, Any]:
        """Agent node with enhanced timeout protection, circuit breaker, and tool calling support."""
        _ = config  # Acknowledge unused parameter

        # Check iteration limits
        if state.current_iteration >= state.max_iterations:
            timeout_error = f"Maximum iterations ({state.max_iterations}) reached. Stopping to prevent infinite loops."
            lang_chain_message = LangChainMessage(
                content=timeout_error,
                type="ai",
                additional_kwargs={},
                response_metadata={},
            )
            return {
                "messages": [lang_chain_message],
                "current_iteration": state.current_iteration + 1,
            }

        try:
            # Initialize LLM if not done yet
            if self.llm is None:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path)

            # Build messages for LLM
            messages = build_lc_messages(state.messages)

            # DEBUG: Log the actual messages being sent to the LLM
            self._logger.info(f"DEBUG: Sending {len(messages)} messages to LLM:")
            for i, msg in enumerate(messages):
                if hasattr(msg, "type"):
                    msg_type = msg.type if msg.type else "unknown"
                elif hasattr(msg, "__class__"):
                    msg_type = msg.__class__.__name__
                else:
                    msg_type = str(type(msg))

                content = str(msg.content) if hasattr(msg, "content") else str(msg)
                content_preview = (
                    content[:200] + "..." if len(content) > 200 else content
                )
                self._logger.info(
                    f"  Message {i}: Type={msg_type}, Content={content_preview}"
                )

            # Stream with shared adaptive controls
            response = await self._stream_with_adaptive_controls(messages)

            # Extract content for tool call parsing
            response_content = (
                str(response.content) if hasattr(response, "content") else str(response)
            )

            # Parse tool calls from Qwen's JSON format
            tool_calls = self._parse_qwen_tool_calls(response_content)

            # Create final response with tool calls if found
            if tool_calls:
                # Remove tool call JSON from the visible content
                clean_content = self._clean_tool_calls_from_content(response_content)
                formatted_response = AIMessage(
                    content=(
                        clean_content
                        if clean_content
                        else "Let me search for that information."
                    ),
                    tool_calls=tool_calls,
                )
                self._logger.info(f"Qwen parsed {len(tool_calls)} tool calls")
                for i, tool_call in enumerate(tool_calls):
                    self._logger.debug(
                        f"Tool call {i}: {tool_call['name']} with args {tool_call['args']}"
                    )

                # DEBUG: Log the formatted message structure
                coerced_message = coerce_to_langchain_message_dict(formatted_response)
                self._logger.info(
                    f"DEBUG: Coerced message type: {coerced_message.get('type', 'unknown')}"
                )
                self._logger.info(
                    f"DEBUG: Coerced message has tool_calls: {'tool_calls' in coerced_message}"
                )
                if "tool_calls" in coerced_message:
                    self._logger.info(
                        f"DEBUG: Tool calls structure: {coerced_message['tool_calls']}"
                    )
                else:
                    self._logger.error(
                        "DEBUG: CRITICAL - tool_calls lost during coercion!"
                    )
            else:
                formatted_response = response

            # Return LangChainMessage directly for LangGraphState compatibility
            if isinstance(formatted_response, AIMessage):
                # Convert AIMessage to LangChainMessage format for LangGraph
                lang_chain_message = LangChainMessage(
                    content=(
                        str(formatted_response.content)
                        if formatted_response.content
                        else ""
                    ),
                    type="ai",
                    tool_calls=getattr(formatted_response, "tool_calls", None),
                    additional_kwargs=getattr(
                        formatted_response, "additional_kwargs", {}
                    ),
                    response_metadata=getattr(
                        formatted_response, "response_metadata", {}
                    ),
                )
            else:
                # Handle other response types
                lang_chain_message = LangChainMessage(
                    content=str(formatted_response),
                    type="ai",
                    additional_kwargs={},
                    response_metadata={},
                )

            self._logger.info(
                f"QwenMoE _agent_node: RETURNING AI message with tool_calls: {lang_chain_message.tool_calls}"
            )
            self._logger.info(
                f"QwenMoE _agent_node: Message type: {lang_chain_message.type}"
            )
            self._logger.info(
                f"QwenMoE _agent_node: Current state has {len(state.messages)} messages"
            )

            # Explicitly append to existing messages to ensure proper state accumulation
            all_messages = list(state.messages) + [lang_chain_message]
            self._logger.info(
                f"QwenMoE _agent_node: Will return {len(all_messages)} total messages"
            )

            return {
                "messages": all_messages,
                "current_iteration": state.current_iteration + 1,
            }

        except Exception as e:
            error_msg = f"Error in agent node: {str(e)}"
            self._logger.error(error_msg)
            # Return LangChainMessage directly for LangGraphState compatibility
            lang_chain_message = LangChainMessage(
                content=error_msg,
                type="ai",
                additional_kwargs={},
                response_metadata={},
            )
            return {
                "messages": [lang_chain_message],
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
