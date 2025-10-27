"""
Streaming response state management for handling different types of content during chat streaming.
Provides clean state machine for routing content to appropriate ChatResponse properties.
"""

import re
import json
from enum import Enum
from typing import Optional, Dict, Any, List

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ToolCall,
    Thought,
)


class StreamingState(Enum):
    """States for streaming response content routing."""

    THINKING = "thinking"  # Between <think> and </think> tags
    PROCESSING = "processing"  # After tool execution, processing results
    EXECUTING = "executing"  # Between tool call tags, parsing arguments
    RESPONDING = "responding"  # Default state for main response content


class StreamingResponseState:
    """
    Manages the state of streaming chat responses and routes content to appropriate ChatResponse fields.

    Handles state transitions based on XML tags and routes content accordingly:
    - THINKING/PROCESSING: content goes to ChatResponse.thinking.text
    - RESPONDING: content goes to ChatResponse.message.content[].text
    - EXECUTING: content goes to ChatResponse.tool_calls[]
    """

    def __init__(self):
        self.state = StreamingState.RESPONDING
        self.thinking_buffer = ""
        self.tool_call_buffer = ""
        self.response_buffer = ""
        self.current_tool_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[ToolCall] = []
        self.accumulated_thinking = ""
        self.response_completed = False  # Track if response is finished
        self.json_buffer = ""  # Buffer for detecting JSON metadata
        self.in_json_block = False  # Track if we're in a JSON metadata block

        # Regex patterns for state detection
        self.think_start_pattern = re.compile(r"<think>")
        self.think_end_pattern = re.compile(r"</think>")
        self.tool_call_start_pattern = re.compile(r"<(?:tool|function)[-_]call>")
        self.tool_call_end_pattern = re.compile(r"</(?:tool|function)[-_]call>")

        # JSON metadata detection patterns
        self.json_start_pattern = re.compile(r'^\s*\{\s*"')
        self.json_block_pattern = re.compile(
            r'^\s*\{\s*"[^"]+"\s*:\s*\[?\{'
        )  # Detect structured JSON blocks

    def process_chunk(self, chunk: str) -> ChatResponse:
        """
        Process a streaming chunk and return appropriate ChatResponse based on current state.

        Args:
            chunk: Text chunk from streaming response

        Returns:
            ChatResponse with content routed to appropriate field based on current state
        """
        if not chunk:
            return self._create_empty_response()

        # Check if we've already completed the response
        if self.response_completed:
            return self._create_empty_response()

        # Check for state transitions first
        self._check_state_transitions(chunk)

        # Route content based on current state
        if self.state == StreamingState.THINKING:
            return self._handle_thinking_content(chunk)

        if self.state == StreamingState.PROCESSING:
            return self._handle_processing_content(chunk)

        if self.state == StreamingState.EXECUTING:
            return self._handle_executing_content(chunk)

        return self._handle_responding_content(chunk)

    def _check_state_transitions(self, chunk: str) -> None:
        """Check for state transition markers in the chunk."""

        # Check for thinking start
        if self.think_start_pattern.search(chunk):
            self.state = StreamingState.THINKING
            # Remove the tag from future processing
            chunk = self.think_start_pattern.sub("", chunk)

        # Check for thinking end
        if self.think_end_pattern.search(chunk):
            self.state = StreamingState.RESPONDING
            # Remove the tag from future processing
            chunk = self.think_end_pattern.sub("", chunk)

        # Check for tool call start (XML tags)
        if self.tool_call_start_pattern.search(chunk):
            self.state = StreamingState.EXECUTING
            self.current_tool_call = {
                "tool_name": "",
                "execution_id": f"call_{len(self.tool_calls)}",
                "success": True,
                "args": {},
                "result_data": {},
                "execution_time_ms": 0,
            }
            # Remove the tag from future processing
            chunk = self.tool_call_start_pattern.sub("", chunk)

        # Also check for function calls without XML wrappers
        elif self._detect_function_call_start(chunk):
            self.state = StreamingState.EXECUTING
            self.current_tool_call = {
                "tool_name": "",
                "execution_id": f"call_{len(self.tool_calls)}",
                "success": True,
                "args": {},
                "result_data": {},
                "execution_time_ms": 0,
            }

        # Check for tool call end (XML tags)
        if self.tool_call_end_pattern.search(chunk):
            if self.current_tool_call:
                # Parse accumulated tool call buffer as JSON
                try:
                    tool_data = json.loads(self.tool_call_buffer)
                    self.current_tool_call["tool_name"] = tool_data.get("name", "")
                    self.current_tool_call["args"] = tool_data.get(
                        "args", tool_data.get("arguments", {})
                    )

                    # Create ToolCall
                    tool_result = ToolCall(**self.current_tool_call)
                    self.tool_calls.append(tool_result)

                except (json.JSONDecodeError, Exception):
                    # If parsing fails, create a basic tool call entry
                    tool_result = ToolCall(
                        tool_name="unknown",
                        execution_id=self.current_tool_call.get("execution_id"),
                        success=False,
                        error_message="Failed to parse tool call arguments",
                        execution_time_ms=0,
                        message_id=0,
                    )
                    self.tool_calls.append(tool_result)

                # Reset buffers and transition to processing
                self.tool_call_buffer = ""
                self.current_tool_call = None
                self.state = StreamingState.PROCESSING

            # Remove the tag from future processing
            chunk = self.tool_call_end_pattern.sub("", chunk)

    def _handle_thinking_content(self, chunk: str) -> ChatResponse:
        """Handle content when in THINKING state."""
        # Clean chunk of XML tags
        clean_chunk = self._clean_xml_tags(chunk)
        if clean_chunk:
            self.thinking_buffer += clean_chunk
            self.accumulated_thinking += clean_chunk

        # Return ChatResponse with thinking content
        thoughts = [Thought(text=clean_chunk)] if clean_chunk else None
        message = Message(role=MessageRole.ASSISTANT, content=[], thoughts=thoughts)
        return ChatResponse(
            message=message,
            done=False,
        )

    def _handle_processing_content(self, chunk: str) -> ChatResponse:
        """Handle content when in PROCESSING state."""
        # Processing state also goes to thinking for now (as per requirements)
        clean_chunk = self._clean_xml_tags(chunk)
        if clean_chunk:
            self.thinking_buffer += clean_chunk
            self.accumulated_thinking += clean_chunk

        thoughts = [Thought(text=clean_chunk)] if clean_chunk else None
        message = Message(role=MessageRole.ASSISTANT, content=[], thoughts=thoughts)
        return ChatResponse(
            message=message,
            done=False,
        )

    def _handle_executing_content(self, chunk: str) -> ChatResponse:
        """Handle content when in EXECUTING state."""
        # Clean chunk and add to tool call buffer
        clean_chunk = self._clean_xml_tags(chunk)
        if clean_chunk:
            self.tool_call_buffer += clean_chunk

            # Try to detect and parse JSON function calls in the buffer
            self._try_parse_function_call()

        # Return ChatResponse with current tool calls
        tool_calls = self.tool_calls.copy() if self.tool_calls else None
        message = Message(role=MessageRole.ASSISTANT, content=[], tool_calls=tool_calls)
        return ChatResponse(
            message=message,
            done=False,
        )

    def _handle_responding_content(self, chunk: str) -> ChatResponse:
        """Handle content when in RESPONDING state (default)."""
        # Clean chunk and add to response buffer
        clean_chunk = self._clean_xml_tags(chunk)

        # Filter out JSON metadata blocks
        if clean_chunk:
            filtered_chunk = self._detect_json_block_boundaries(clean_chunk)
            # Only process non-JSON content
            if filtered_chunk and not self._is_json_metadata(filtered_chunk):
                self.response_buffer += filtered_chunk

                # Return ChatResponse with main message content
                content = [
                    MessageContent(type=MessageContentType.TEXT, text=filtered_chunk)
                ]
                return ChatResponse(
                    message=Message(role=MessageRole.ASSISTANT, content=content),
                    done=False,
                )

        # Return empty response if content was filtered out
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=[]), done=False
        )

    def _clean_xml_tags(self, chunk: str) -> str:
        """Remove XML tags from chunk."""
        if not chunk:
            return chunk

        # Remove XML tags only
        chunk = self.think_start_pattern.sub("", chunk)
        chunk = self.think_end_pattern.sub("", chunk)
        chunk = self.tool_call_start_pattern.sub("", chunk)
        chunk = self.tool_call_end_pattern.sub("", chunk)

        return chunk

    def _create_empty_response(self) -> ChatResponse:
        """Create an empty ChatResponse."""
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=[]), done=False
        )

    def _is_json_metadata(self, content: str) -> bool:
        """Check if content looks like JSON metadata that should be filtered."""
        stripped = content.strip()
        if not stripped:
            return False

        # Check for JSON object patterns
        if stripped.startswith("{") and '"' in stripped:
            # Look for common metadata patterns
            metadata_indicators = [
                '"intent":',
                '"analysis":',
                '"category":',
                '"classification":',
                '"reasoning":',
                '"metadata":',
                '"type":',
                '"complexity":',
                '"user_intent":',
                '"system_type":',
                '"session_context":',
            ]
            return any(indicator in stripped for indicator in metadata_indicators)

        return False

    def _detect_json_block_boundaries(self, content: str) -> str:
        """Detect and filter out JSON metadata blocks from streaming content."""
        if not content:
            return content

        # Add content to JSON buffer for analysis
        self.json_buffer += content

        lines = self.json_buffer.split("\n")
        filtered_lines = []

        for line in lines:
            stripped_line = line.strip()

            # Detect start of JSON block
            if not self.in_json_block and self.json_block_pattern.match(stripped_line):
                self.in_json_block = True
                continue

            # Skip content while in JSON block
            if self.in_json_block:
                # Check for end of JSON block (closing brace at start of line)
                if stripped_line == "}" or (
                    stripped_line.endswith("}") and not stripped_line.endswith('"}}')
                ):
                    self.in_json_block = False
                    continue
                else:
                    continue

            # Keep non-JSON content
            if not self.in_json_block:
                filtered_lines.append(line)

        # Update buffer with remaining content
        if filtered_lines:
            result = "\n".join(filtered_lines)
            self.json_buffer = ""  # Clear buffer after processing
            return result
        else:
            # Keep buffer if we're still in JSON block
            return ""

    def _try_parse_function_call(self) -> None:
        """Try to parse function calls from the tool call buffer."""
        if not self.tool_call_buffer.strip():
            return

        # Look for JSON function call patterns in the buffer
        buffer = self.tool_call_buffer.strip()

        # Try to find complete JSON objects that look like function calls
        json_start = buffer.find("{")
        if json_start == -1:
            return

        # Find the matching closing brace
        brace_count = 0
        json_end = -1

        for i in range(json_start, len(buffer)):
            if buffer[i] == "{":
                brace_count += 1
            elif buffer[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i
                    break

        if json_end != -1:
            # Extract and try to parse the JSON
            json_str = buffer[json_start : json_end + 1]
            try:
                function_data = json.loads(json_str)

                # Check if this looks like a function call
                if self._is_function_call_json(function_data):
                    # Create tool execution result
                    if not self.current_tool_call:
                        self.current_tool_call = {
                            "tool_name": "",
                            "execution_id": f"call_{len(self.tool_calls)}",
                            "success": True,
                            "args": {},
                            "result_data": {},
                            "execution_time_ms": 0,
                        }

                    # Extract function call details
                    self.current_tool_call["tool_name"] = function_data.get(
                        "name", function_data.get("function", "")
                    )
                    self.current_tool_call["args"] = function_data.get(
                        "args",
                        function_data.get(
                            "arguments", function_data.get("parameters", {})
                        ),
                    )

                    # Create and add tool execution result
                    tool_result = ToolCall(**self.current_tool_call)
                    self.tool_calls.append(tool_result)

                    # Clear the parsed portion from buffer
                    self.tool_call_buffer = buffer[json_end + 1 :]
                    self.current_tool_call = None

            except (json.JSONDecodeError, Exception) as e:
                # If parsing fails, keep accumulating content
                pass

    def _is_function_call_json(self, data: dict) -> bool:
        """Check if JSON data looks like a function call."""
        if not isinstance(data, dict):
            return False

        # Look for common function call indicators
        function_indicators = ["name", "function", "tool_name"]
        args_indicators = ["args", "arguments", "parameters"]

        has_function = any(key in data for key in function_indicators)
        has_args = any(key in data for key in args_indicators)

        return has_function and (has_args or len(data) >= 1)

    def _detect_function_call_start(self, chunk: str) -> bool:
        """Detect if chunk contains the start of a function call without XML wrappers."""
        stripped = chunk.strip()

        # Look for JSON patterns that might be function calls
        if "{" in stripped and '"' in stripped:
            # Check for common function call patterns
            function_patterns = [
                r'\{\s*"name"\s*:\s*"',
                r'\{\s*"function"\s*:\s*"',
                r'\{\s*"tool_name"\s*:\s*"',
                r'\{\s*"action"\s*:\s*"',
            ]

            for pattern in function_patterns:
                if re.search(pattern, stripped):
                    return True

        return False

    def get_final_response(self) -> ChatResponse:
        """
        Get the final consolidated ChatResponse with all accumulated content.
        Since content is already streamed token-by-token, this should only contain
        the final state markers and accumulated thinking/tool_calls.

        Returns:
            Final ChatResponse with done=True and accumulated thinking/tool_calls
        """
        # Don't include response_buffer content since it's already been streamed
        # Only include the final thinking and tool_calls
        thinking = (
            Thought(text=self.accumulated_thinking)
            if self.accumulated_thinking
            else None
        )

        thoughts = [thinking] if thinking else None
        tool_calls = self.tool_calls if self.tool_calls else None

        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[],  # Empty since content was already streamed
                thoughts=thoughts,
                tool_calls=tool_calls,
            ),
            done=True,
        )

    def reset(self) -> None:
        """Reset the state for a new streaming session."""
        self.state = StreamingState.RESPONDING
        self.thinking_buffer = ""
        self.tool_call_buffer = ""
        self.response_buffer = ""
        self.current_tool_call = None
        self.tool_calls = []
        self.accumulated_thinking = ""
        self.response_completed = False
        self.json_buffer = ""
        self.in_json_block = False
