"""
Unit tests for consolidated message conversion utilities.

Tests all the unified conversion functions to ensure proper functionality
and catch issues like newline character problems.
"""

import pytest
import json
from typing import List
from datetime import datetime, timezone

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    ToolCall,
)
from utils.message_conversion import (
    extract_text_from_message,
    create_text_message_content,
    message_to_lc_message,
    lc_message_to_message,
    messages_to_lc_messages,
    lc_messages_to_messages,
    convert_message_content_to_langchain_format,
    convert_lc_message_content_to_message_format,
    get_most_recent_user_message_text,
    normalize_message_input,
)
from utils.tool_call_extraction import (
    extract_tool_calls_from_langchain_message,
    has_tool_calls_in_langchain_message,
    extract_tool_calls_from_streaming_chunks,
    extract_tool_calls_from_message_content,
    create_tool_call_message_content,
)
from utils.tool_call_types import (
    extract_tool_calls_as_models,
    has_tool_call_requests_as_models,
    extract_tool_call_requests,
    tool_call_request_to_execution_result,
    has_tool_calls,
    is_langchain_tool_call,
)


class TestTextExtraction:
    """Test text extraction functions."""

    def test_extract_text_from_message_simple(self):
        """Test extracting text from a simple Message object."""
        content = [
            MessageContent(type=MessageContentType.TEXT, text="Hello world", url=None)
        ]
        message = Message(
            role=MessageRole.USER,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = extract_text_from_message(message)
        assert result == "Hello world"
        assert "\n" not in result  # Should not have newlines for single text

    def test_extract_text_from_message_multiple_parts(self):
        """Test extracting text from Message with multiple text parts."""
        content = [
            MessageContent(type=MessageContentType.TEXT, text="Hello", url=None),
            MessageContent(type=MessageContentType.TEXT, text=" world", url=None),
        ]
        message = Message(
            role=MessageRole.USER,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = extract_text_from_message(message)
        assert result == "Hello world"
        # Should concatenate without extra newlines
        assert result.count("\n") == 0

    def test_extract_text_no_extra_newlines_bug(self):
        """Test that text extraction doesn't add newlines between characters."""
        content = [
            MessageContent(type=MessageContentType.TEXT, text="a", url=None),
            MessageContent(type=MessageContentType.TEXT, text="b", url=None),
            MessageContent(type=MessageContentType.TEXT, text="c", url=None),
        ]
        message = Message(
            role=MessageRole.USER,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = extract_text_from_message(message)
        assert result == "abc"
        # Critical: should NOT be "a\nb\nc" which would cause the bug
        assert "\n" not in result

    def test_extract_text_from_lc_message_simple(self):
        """Test extracting text from LangChain BaseMessage."""
        lc_message = HumanMessage(content="Hello LangChain")

        result = extract_text_from_message(lc_message)
        assert result == "Hello LangChain"

    def test_extract_text_from_lc_message_multimodal(self):
        """Test extracting text from LangChain multimodal message."""
        lc_message = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " multimodal"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                },
            ]
        )

        result = extract_text_from_message(lc_message)
        assert result == "Hello multimodal"
        # Should join without extra newlines
        assert result.count("\n") == 0

    def test_extract_text_preserves_spaces(self):
        """Test that text extraction preserves necessary spaces."""
        content = [
            MessageContent(type=MessageContentType.TEXT, text="Hello ", url=None),
            MessageContent(type=MessageContentType.TEXT, text="world", url=None),
        ]
        message = Message(
            role=MessageRole.USER,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = extract_text_from_message(message)
        assert result == "Hello world"

    def test_extract_text_mixed_content_types(self):
        """Test text extraction ignores non-text content."""
        content = [
            MessageContent(type=MessageContentType.TEXT, text="Hello", url=None),
            MessageContent(
                type=MessageContentType.IMAGE,
                text=None,
                url="http://example.com/image.jpg",
            ),
            MessageContent(type=MessageContentType.TEXT, text=" world", url=None),
        ]
        message = Message(
            role=MessageRole.USER,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = extract_text_from_message(message)
        assert result == "Hello world"


class TestMessageContentCreation:
    """Test message content creation functions."""

    def test_create_text_message_content(self):
        """Test creating text message content."""
        result = create_text_message_content("Test content")

        assert len(result) == 1
        assert result[0].type == MessageContentType.TEXT
        assert result[0].text == "Test content"
        assert result[0].url is None

    def test_create_text_message_content_empty(self):
        """Test creating empty text message content."""
        result = create_text_message_content("")

        assert len(result) == 1
        assert result[0].type == MessageContentType.TEXT
        assert result[0].text == ""


class TestMessageConversion:
    """Test message conversion between internal and LangChain formats."""

    def test_message_to_lc_message_user(self):
        """Test converting user message to LangChain format."""
        content = [MessageContent(type=MessageContentType.TEXT, text="Hello", url=None)]
        message = Message(
            role=MessageRole.USER,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = message_to_lc_message(message)

        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_message_to_lc_message_assistant(self):
        """Test converting assistant message to LangChain format."""
        content = [
            MessageContent(type=MessageContentType.TEXT, text="Hi there", url=None)
        ]
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

        result = message_to_lc_message(message)

        assert isinstance(result, AIMessage)
        assert result.content == "Hi there"

    def test_lc_message_to_message_conversion(self):
        """Test converting LangChain message to internal format."""
        lc_message = HumanMessage(content="Test message")

        result = lc_message_to_message(lc_message, conversation_id=123)

        assert result.role == MessageRole.USER
        assert len(result.content) == 1
        assert result.content[0].type == MessageContentType.TEXT
        assert result.content[0].text == "Test message"
        assert result.conversation_id == 123

    def test_messages_batch_conversion(self):
        """Test batch conversion of messages."""
        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="Hello", url=None)
                ],
                created_at=datetime.now(timezone.utc),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="Hi", url=None)
                ],
                created_at=datetime.now(timezone.utc),
            ),
        ]

        lc_messages = messages_to_lc_messages(messages)

        assert len(lc_messages) == 2
        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)

        # Test reverse conversion
        back_to_messages = lc_messages_to_messages(lc_messages, conversation_id=456)

        assert len(back_to_messages) == 2
        assert back_to_messages[0].role == MessageRole.USER
        assert back_to_messages[1].role == MessageRole.ASSISTANT


class TestToolCallHandling:
    """Test tool call extraction and processing."""

    def test_is_langchain_tool_call(self):
        """Test tool call identification."""
        valid_tool_call = {
            "name": "test_tool",
            "args": {"param": "value"},
            "id": "call_123",
        }

        assert is_langchain_tool_call(valid_tool_call) is True

        invalid_tool_call = {"name": "test_tool"}  # Missing args
        assert is_langchain_tool_call(invalid_tool_call) is False

    def test_has_tool_calls(self):
        """Test detecting tool calls in messages."""
        # Message with tool calls
        ai_message_with_tools = AIMessage(
            content="I'll use a tool",
            tool_calls=[
                {"name": "test_tool", "args": {"param": "value"}, "id": "call_123"}
            ],
        )

        assert has_tool_calls(ai_message_with_tools) is True

        # Message without tool calls
        ai_message_no_tools = AIMessage(content="Just a regular message")
        assert has_tool_calls(ai_message_no_tools) is False

    def test_extract_tool_calls_from_langchain_message(self):
        """Test extracting tool calls from LangChain messages."""
        ai_message = AIMessage(
            content="I'll use a tool",
            tool_calls=[
                {"name": "search", "args": {"query": "test"}, "id": "call_123"}
            ],
        )

        tool_calls = extract_tool_calls_from_langchain_message(ai_message)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].args == {"query": "test"}

    def test_create_tool_call_message_content(self):
        """Test creating message content from tool calls."""
        tool_call = ToolCall(
            name="test_tool",
            success=True,
            args={"param": "value"},
            result_data={"result": "success"},
            execution_id="call_123",
        )

        content = create_tool_call_message_content(tool_call)

        assert content.type == MessageContentType.TOOL_CALL
        assert content.text is not None

        # Verify JSON is valid
        parsed = json.loads(content.text)
        assert parsed["name"] == "test_tool"
        assert parsed["success"] is True
        assert parsed["args"] == {"param": "value"}


class TestNormalization:
    """Test message input normalization."""

    def test_normalize_string_input(self):
        """Test normalizing string input to messages."""
        result = normalize_message_input("Hello world")

        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert len(result[0].content) == 1
        assert result[0].content[0].text == "Hello world"

    def test_normalize_list_input(self):
        """Test normalizing list of strings."""
        result = normalize_message_input(["First message", "Second message"])

        assert len(result) == 2
        assert result[0].content[0].text == "First message"
        assert result[1].content[0].text == "Second message"

    def test_normalize_message_input(self):
        """Test normalizing Message object input."""
        original_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT, text="Already a message", url=None
                )
            ],
            created_at=datetime.now(timezone.utc),
        )

        result = normalize_message_input(original_message)

        assert len(result) == 1
        assert result[0] is original_message  # Should return the same object


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content_handling(self):
        """Test handling empty content."""
        message = Message(
            role=MessageRole.USER,
            content=[],
            created_at=datetime.now(timezone.utc),
        )

        result = extract_text_from_message(message)
        assert result == ""

    def test_empty_string_content_handling(self):
        """Test handling empty string content in LangChain messages."""
        lc_message = HumanMessage(content="")

        result = extract_text_from_message(lc_message)
        assert result == ""

    def test_malformed_tool_call_handling(self):
        """Test handling malformed tool call data."""
        content = [
            MessageContent(
                type=MessageContentType.TOOL_CALL, text="invalid json {", url=None
            )
        ]

        tool_calls = extract_tool_calls_from_message_content(content)
        assert len(tool_calls) == 0  # Should handle gracefully

    def test_recent_user_message_empty_list(self):
        """Test getting recent user message from empty list."""
        result = get_most_recent_user_message_text([])
        assert result == ""

    def test_recent_user_message_no_human_messages(self):
        """Test getting recent user message when no human messages exist."""
        messages = [
            AIMessage(content="I'm an AI"),
            SystemMessage(content="System message"),
        ]

        result = get_most_recent_user_message_text(messages)
        assert result == "System message"  # Should fall back to last message
