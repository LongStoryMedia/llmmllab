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
    ToolExecutionResult,
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
        self.tool_calls: List[ToolExecutionResult] = []
        self.accumulated_thinking = ""
        self.response_completed = False  # Track if response is finished

        # Regex patterns for state detection
        self.think_start_pattern = re.compile(r"<think>")
        self.think_end_pattern = re.compile(r"</think>")
        self.tool_call_start_pattern = re.compile(r"<(?:tool|function)[-_]call>")
        self.tool_call_end_pattern = re.compile(r"</(?:tool|function)[-_]call>")

        # Remove completion patterns - these were symptom fixes

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

        # Check for tool call start
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

        # Check for tool call end
        if self.tool_call_end_pattern.search(chunk):
            if self.current_tool_call:
                # Parse accumulated tool call buffer as JSON
                try:
                    tool_data = json.loads(self.tool_call_buffer)
                    self.current_tool_call["tool_name"] = tool_data.get("name", "")
                    self.current_tool_call["args"] = tool_data.get(
                        "args", tool_data.get("arguments", {})
                    )

                    # Create ToolExecutionResult
                    tool_result = ToolExecutionResult(**self.current_tool_call)
                    self.tool_calls.append(tool_result)

                except (json.JSONDecodeError, Exception):
                    # If parsing fails, create a basic tool call entry
                    tool_result = ToolExecutionResult(
                        tool_name="unknown",
                        execution_id=self.current_tool_call["execution_id"],
                        success=False,
                        error_message="Failed to parse tool call arguments",
                        execution_time_ms=0,
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
        thinking = Thought(text=clean_chunk) if clean_chunk else None
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=[]),
            thinking=thinking,
            done=False,
        )

    def _handle_processing_content(self, chunk: str) -> ChatResponse:
        """Handle content when in PROCESSING state."""
        # Processing state also goes to thinking for now (as per requirements)
        clean_chunk = self._clean_xml_tags(chunk)
        if clean_chunk:
            self.thinking_buffer += clean_chunk
            self.accumulated_thinking += clean_chunk

        thinking = Thought(text=clean_chunk) if clean_chunk else None
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=[]),
            thinking=thinking,
            done=False,
        )

    def _handle_executing_content(self, chunk: str) -> ChatResponse:
        """Handle content when in EXECUTING state."""
        # Clean chunk and add to tool call buffer
        clean_chunk = self._clean_xml_tags(chunk)
        if clean_chunk:
            self.tool_call_buffer += clean_chunk

        # Return ChatResponse with current tool calls
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=[]),
            tool_calls=self.tool_calls.copy() if self.tool_calls else None,
            done=False,
        )

    def _handle_responding_content(self, chunk: str) -> ChatResponse:
        """Handle content when in RESPONDING state (default)."""
        # Clean chunk and add to response buffer
        clean_chunk = self._clean_xml_tags(chunk)
        if clean_chunk:
            self.response_buffer += clean_chunk

        # Return ChatResponse with main message content
        content = (
            [MessageContent(type=MessageContentType.TEXT, text=clean_chunk)]
            if clean_chunk
            else []
        )
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content), done=False
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

        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[],  # Empty since content was already streamed
            ),
            thinking=thinking,
            tool_calls=self.tool_calls if self.tool_calls else None,
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
