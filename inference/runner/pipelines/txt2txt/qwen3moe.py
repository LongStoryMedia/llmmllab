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
        if content is None:
            return None
        
        # Ensure buffer is initialized
        if not hasattr(self, 'buffer') or self.buffer is None:
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
                if not hasattr(self, 'think_content') or self.think_content is None:
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
                thinking=None  # Thinking will be added in finalize_streaming
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

        # Add tool information if available with function calling format
        if tools:
            tool_descriptions = []
            for tool in tools:
                # Provide specific argument info for known tools
                tool_args = "query (string)"  # default
                if tool.name == "web_search":
                    tool_args = "query (string), limit (integer, optional)"
                elif tool.name == "memory_retrieval":
                    tool_args = "tool_input (list of embeddings)"
                elif tool.name == "summarization":
                    tool_args = "tool_input (list of messages)"
                
                tool_descriptions.append(f"- **{tool.name}**: {tool.description}\n  Arguments: {tool_args}")

            tool_info = "\n".join(tool_descriptions)
            base_prompt += f"""

## Available Tools:
{tool_info}

## How to Use Tools:
When you need to use a tool (like for web searches, calculations, or retrieving information), format your response with a JSON code block:

```json
{{
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ]
}}
```

## CRITICAL TOOL USAGE RULES:

1. **ALWAYS use tools for current information, web searches, or when you don't have specific knowledge**
2. **Format tool calls EXACTLY as shown below - no variations allowed**
3. **Use "arguments" not "args" in the JSON**
4. **Put the JSON in a code block with ```json```**

## EXACT FORMAT REQUIRED:
When you need to use a tool, respond with ONLY this format:

```json
{{
    "tool_calls": [
        {{
            "name": "web_search", 
            "arguments": {{
                "query": "your search query here"
            }}
        }}
    ]
}}
```

## MORE EXAMPLES:

For web search:
```json
{{
    "tool_calls": [
        {{
            "name": "web_search",
            "arguments": {{
                "query": "latest AI breakthroughs 2025",
                "limit": 5
            }}
        }}
    ]
}}
```

For memory retrieval:
```json
{{
    "tool_calls": [
        {{
            "name": "memory_retrieval",
            "arguments": {{
                "tool_input": []
            }}
        }}
    ]
}}
```

**DO NOT:**
- Write explanatory text before or after the JSON
- Use any format other than the exact JSON structure shown
- Say you cannot use tools - you CAN and MUST use them when needed
- Make up information when you should search for it

**DO:**
- Use web_search for any current information, links, or recent developments
- Follow the exact JSON format every time
- Put JSON in ```json code blocks"""

        return base_prompt

    def _parse_qwen_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse Qwen-style JSON tool calls from the generated content."""
        import json
        import re

        tool_calls = []
        
        # Look for JSON code blocks containing tool_calls
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if "tool_calls" in data:
                    for i, tool_call in enumerate(data["tool_calls"]):
                        if "name" in tool_call:
                            # Convert to LangGraph format
                            formatted_call = {
                                "name": tool_call["name"],
                                "args": tool_call.get("arguments", {}),
                                "id": f"call_{i}_{tool_call['name']}",
                                "type": "tool_call"
                            }
                            tool_calls.append(formatted_call)
                            self._logger.debug(f"Parsed Qwen tool call: {formatted_call}")
            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(f"Failed to parse Qwen tool call from: {match[:100]}... Error: {e}")
                continue
                
        # Also look for function call patterns outside of JSON blocks
        if not tool_calls:
            # Look for standalone function calls in the format {"name": "...", "arguments": {...}}
            func_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
            func_matches = re.findall(func_pattern, content)
            
            for i, (name, args_str) in enumerate(func_matches):
                try:
                    args = json.loads(args_str)
                    formatted_call = {
                        "name": name,
                        "args": args,
                        "id": f"call_{i}_{name}",
                        "type": "tool_call"
                    }
                    tool_calls.append(formatted_call)
                    self._logger.debug(f"Parsed Qwen function call: {formatted_call}")
                except json.JSONDecodeError as e:
                    self._logger.warning(f"Failed to parse function arguments: {args_str}. Error: {e}")
                    continue
        
        return tool_calls

    def _clean_tool_calls_from_content(self, content: str) -> str:
        """Remove tool call JSON blocks from the content to get clean user-facing text."""
        import re
        
        # Remove JSON code blocks that contain tool_calls
        json_pattern = r'```json\s*\{.*?"tool_calls".*?\}\s*```'
        content = re.sub(json_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove standalone function call patterns
        func_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        content = re.sub(func_pattern, '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()

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
        """Agent node with enhanced timeout protection, circuit breaker, and tool calling support."""
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
                    content=clean_content if clean_content.strip() else "Let me search for that information.",
                    tool_calls=tool_calls
                )
                self._logger.info(f"Qwen parsed {len(tool_calls)} tool calls")
                for i, tool_call in enumerate(tool_calls):
                    self._logger.debug(f"Tool call {i}: {tool_call['name']} with args {tool_call['args']}")
            else:
                formatted_response = response

            return {
                "messages": [coerce_to_langchain_message_dict(formatted_response)],
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
