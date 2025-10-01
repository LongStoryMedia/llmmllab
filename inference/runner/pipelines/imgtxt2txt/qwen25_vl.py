"""
Qwen 2.5 VL pipeline - Simplified implementation following qwen3 pattern.
"""

import os
import re
import json
import logging
from typing import List, Optional, Union, Dict, Any, Type

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

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
from ..base_langgraph import BaseLangGraphPipeline

ReturnType = Union[str, ChatResponse]


class Qwen25VLPipeline(BaseLangGraphPipeline):
    """Qwen 2.5 VL pipeline with LangGraph support."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
    ):
        """Initialize a Qwen25VLPipeline instance."""
        super().__init__(model, profile, expected_return_type)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info("Qwen25VLPipeline initialized")

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path for the model."""
        # Use the model.model path directly
        return str(self.model.model)

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt for Qwen2.5VL with proper tool calling instructions."""

        # If tools are available, completely override the profile system prompt with tool-focused prompt
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            tools_text = "\n".join(tool_descriptions)

            # Create a completely new tool-focused system prompt that overrides any training behavior
            return f"""You are Qwen2.5-VL, a capable AI assistant with ACTIVE ACCESS to external tools.

CAPABILITY STATUS: You currently have ACTIVE, WORKING access to these tools:
{tools_text}

CRITICAL BEHAVIOR OVERRIDE:
- You CAN and DO have internet access through the web_search tool
- When users request current/recent/2024 information, immediately use web_search
- Do NOT refuse requests - use available tools to fulfill them
- Do NOT mention limitations - you have tools to overcome them

TOOL USAGE IS MANDATORY for current information requests.

EXACT TOOL FORMAT (copy this exactly):
<tool_call>
{{"name": "web_search", "arguments": {{"query": "your search terms"}}}}
</tool_call>

For quantum computing 2024 breakthroughs, respond IMMEDIATELY with:
<tool_call>
{{"name": "web_search", "arguments": {{"query": "quantum computing breakthroughs 2024"}}}}
</tool_call>

Then provide results. NO REFUSALS. Use tools."""
        else:
            # No tools available, use basic prompt
            return (
                "You are Qwen2.5-VL, a helpful AI assistant with vision capabilities."
            )

    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Initialize the LLM with LangChain wrapper for streaming support."""
        try:
            # Import LangChain's ChatLlamaCpp wrapper for proper streaming
            from langchain_community.chat_models import ChatLlamaCpp

            # Get parameters from profile
            n_ctx = getattr(self.profile.parameters, "num_ctx", 4096)
            temperature = getattr(self.profile.parameters, "temperature", 0.7)
            max_tokens = getattr(self.profile.parameters, "max_tokens", 2048)

            # Initialize with LangChain wrapper for proper streaming support
            self.llm = ChatLlamaCpp(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=False,
                streaming=True,
            )

            self._logger.info(f"Initialized Qwen25VL model from {gguf_path}")

        except Exception as e:
            self._logger.error(f"Failed to initialize Qwen25VL model: {e}")
            # Fallback to basic LangChain initialization
            from langchain_community.chat_models import ChatLlamaCpp

            self.llm = ChatLlamaCpp(
                model_path=gguf_path,
                n_ctx=getattr(self.profile.parameters, "num_ctx", 4096),
                n_gpu_layers=-1,
                verbose=False,
                streaming=True,
            )

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create LangGraph workflow."""
        self._logger.info(
            f"Creating LangGraph workflow with {len(tools) if tools else 0} tools"
        )

        # Store tools for use in agent node
        self._tools = tools

        workflow = StateGraph(LangGraphState)
        workflow.add_node("agent", self._agent_node)

        if tools:
            tool_node = ToolNode(tools)

            def debug_tool_node(state: LangGraphState):
                """Debug wrapper for ToolNode execution."""
                try:
                    self._logger.info(f"Qwen25VL: Executing tools")

                    converted_messages = []
                    for msg in state.messages:
                        converted_msg = coerce_to_lc_message(msg)
                        converted_messages.append(converted_msg)

                    converted_state = state.copy()
                    converted_state.messages = converted_messages

                    result = tool_node.invoke(converted_state)
                    self._logger.info(f"Qwen25VL: ToolNode executed successfully")

                    # Convert the result messages back to proper format
                    if "messages" in result:
                        converted_result_messages = []
                        for msg in result["messages"]:
                            if hasattr(msg, "dict"):
                                # Convert LangChain message to dict format
                                converted_msg = msg.dict()
                            else:
                                converted_msg = msg
                            converted_result_messages.append(converted_msg)
                        result["messages"] = converted_result_messages

                    return result

                except Exception as e:
                    self._logger.error(f"Qwen25VL: Tool execution failed: {e}")
                    error_msg = AIMessage(content=f"Error executing tools: {str(e)}")
                    return {"messages": [error_msg]}

            workflow.add_node("tools", debug_tool_node)

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges(
                "agent", tools_condition, {"tools": "tools", "__end__": END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge(START, "agent")
            workflow.add_edge("agent", END)

        return workflow.compile()

    async def _agent_node(self, state: LangGraphState, config=None) -> Dict[str, Any]:
        """Agent node with enhanced timeout protection and circuit breaker support."""
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
            self._logger.info(
                f"Qwen25VL: Processing agent node with {len(state.messages)} messages"
            )

            # Initialize LLM if not done yet
            if self.llm is None:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path)

            # Build messages for LLM using the base class method
            from utils.langgraph import build_lc_messages

            messages = build_lc_messages(state.messages)

            # Inject our custom system prompt at the beginning if tools are available
            if self._tools:
                from langchain_core.messages import SystemMessage

                custom_system_prompt = await self._create_system_prompt(self._tools)
                # Insert system message at the beginning
                system_msg = SystemMessage(content=custom_system_prompt)
                messages = [system_msg] + messages
                self._logger.info(
                    f"DEBUG: Injected custom system prompt with {len(self._tools)} tools"
                )

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

            # Use base class streaming with timeout and safety controls
            response = await self._stream_with_adaptive_controls(messages)

            # Extract content for tool call parsing
            response_content = (
                str(response.content) if hasattr(response, "content") else str(response)
            )

            self._logger.info(f"Generated response: {len(response_content)} characters")

            # Debug: Log the actual content to understand parsing issues
            self._logger.error(
                f"DEBUG: Response content preview (first 1000 chars):\n{response_content[:1000]}"
            )
            if len(response_content) > 1000:
                self._logger.error(
                    f"DEBUG: Response content suffix (last 500 chars):\n{response_content[-500:]}"
                )

            # Check for specific tool call patterns in the content
            if "<tool_call>" in response_content.lower():
                self._logger.error("DEBUG: Found <tool_call> pattern in content")
            if "web_search" in response_content.lower():
                self._logger.error("DEBUG: Found web_search pattern in content")
            if '"name":' in response_content:
                self._logger.error("DEBUG: Found JSON name pattern in content")

            # Debug: Check what system prompt was actually used
            if self._tools:
                system_prompt = await self._create_system_prompt(self._tools)
                self._logger.error(
                    f"DEBUG: Custom system prompt being used (first 500 chars): {system_prompt[:500]}"
                )
                self._logger.error(f"DEBUG: Tools count: {len(self._tools)}")
            else:
                self._logger.error(
                    "DEBUG: No tools available - using basic system prompt"
                )

            # Parse tool calls if present
            tool_calls = self._parse_qwen_tool_calls(response_content)

            # Create LangChain message with tool calls
            coerced_message = coerce_to_langchain_message_dict(response)
            lang_chain_message = LangChainMessage(**coerced_message)

            # Always set tool_calls as a list (empty if no calls found)
            lang_chain_message.tool_calls = tool_calls if tool_calls else []
            if tool_calls:
                self._logger.info(f"Found {len(tool_calls)} tool calls")
            else:
                self._logger.info("No tool calls found")

            # Build full message list for return
            all_messages = list(state.messages) + [lang_chain_message]

            self._logger.error(
                f"Qwen25VL _agent_node: RETURNING AI message with tool_calls: {lang_chain_message.tool_calls}"
            )
            self._logger.error(
                f"Qwen25VL _agent_node: Message type: {lang_chain_message.type}"
            )
            self._logger.error(
                f"Qwen25VL _agent_node: Current state has {len(state.messages)} messages"
            )
            self._logger.error(
                f"Qwen25VL _agent_node: Will return {len(all_messages)} total messages"
            )

            return {
                "messages": all_messages,
                "current_iteration": state.current_iteration + 1,
            }

        except Exception as e:
            self._logger.error(f"Error in Qwen25VL agent node: {e}")
            error_response = AIMessage(content=f"Error processing request: {str(e)}")
            coerced_message = coerce_to_langchain_message_dict(error_response)
            lang_chain_message = LangChainMessage(**coerced_message)

            all_messages = list(state.messages) + [lang_chain_message]

            return {
                "messages": all_messages,
                "current_iteration": state.current_iteration + 1,
            }

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

        # Pattern 2: Look for <tool_call> XML tags (custom format)
        if not tool_calls:
            tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
            matches = re.findall(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)

            for i, match in enumerate(matches):
                try:
                    # Parse the JSON content
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

        # Pattern 3: Look for mixed function_call tags
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

                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse mixed function_call: {match[:100]}... Error: {e}"
                    )
                    continue

        # Pattern 4: Look for raw JSON tool call (just the JSON structure without wrapper)
        if not tool_calls:
            # Look for standalone JSON at the start or as the main content
            raw_json_pattern = (
                r'^\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\s*\}\s*$'
            )
            raw_matches = re.findall(raw_json_pattern, content.strip(), re.MULTILINE)

            for i, (name, args_str) in enumerate(raw_matches):
                try:
                    args = json.loads(args_str)
                    formatted_call = {
                        "name": name,
                        "args": args,
                        "id": f"call_{i}_{name}",
                        "type": "tool_call",
                    }
                    tool_calls.append(formatted_call)
                    self._logger.debug(f"Parsed raw JSON tool call: {formatted_call}")
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(
                        f"Failed to parse raw JSON tool call: {name}, {args_str[:50]}... Error: {e}"
                    )
                    continue

        # Pattern 5: Look for complete JSON object that represents a tool call
        if not tool_calls:
            try:
                # Try to parse the entire content as JSON
                content_stripped = content.strip()
                if content_stripped.startswith("{") and content_stripped.endswith("}"):
                    tool_data = json.loads(content_stripped)
                    if "name" in tool_data and "arguments" in tool_data:
                        formatted_call = {
                            "name": tool_data["name"],
                            "args": tool_data["arguments"],
                            "id": f"call_0_{tool_data['name']}",
                            "type": "tool_call",
                        }
                        tool_calls.append(formatted_call)
                        self._logger.debug(
                            f"Parsed complete JSON tool call: {formatted_call}"
                        )

            except (json.JSONDecodeError, KeyError) as e:
                self._logger.debug(f"Content is not valid complete JSON tool call: {e}")

        return tool_calls
