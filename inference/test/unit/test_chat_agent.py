"""
Unit tests for ChatAgent functionality.
Tests chat completion, streaming, message conversion, and tool integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from typing import List

from composer.agents.chat_agent import ChatAgent
from models import (
    ChatResponse,
    LangChainMessage,
    ModelProfile,
    PipelinePriority,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    NodeMetadata,
    CircuitBreakerConfig,
)
from composer.core.errors import NodeExecutionError


def create_test_model_profile() -> ModelProfile:
    """Create a test ModelProfile object."""
    from models.model_parameters import ModelParameters
    import uuid

    return ModelProfile(
        id=str(uuid.uuid4()),
        user_id="test-user",
        name="Test Profile",
        model_name="test-model",
        parameters=ModelParameters(
            temperature=0.7,
            num_predict=1000,
        ),
        system_prompt="Test system prompt",
        type=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def create_test_node_metadata() -> NodeMetadata:
    """Create a test NodeMetadata object."""
    import uuid
    import random

    return NodeMetadata(
        node_id=str(uuid.uuid4()),
        node_name="test-chat-node",
        node_type="ChatNode",
        user_id="test-user",
        conversation_id=random.randint(1, 100000),
        execution_time=datetime.now(timezone.utc),
        execution_context={
            "operation": "chat_completion",
            "streaming": False,
        },
    )


def create_test_chat_agent(
    pipeline_factory=None, profile=None, node_metadata=None, **kwargs
) -> ChatAgent:
    """Create a test ChatAgent with all required dependencies."""
    if pipeline_factory is None:
        pipeline_factory = Mock()
    if profile is None:
        profile = create_test_model_profile()
    if node_metadata is None:
        node_metadata = create_test_node_metadata()

    return ChatAgent(
        pipeline_factory=pipeline_factory,
        profile=profile,
        node_metadata=node_metadata,
        **kwargs
    )


def create_test_langchain_messages() -> List[LangChainMessage]:
    """Create test LangChain messages."""
    return [
        LangChainMessage(type="human", content="Hello, how are you?"),
        LangChainMessage(type="ai", content="I'm doing well, thank you!"),
        LangChainMessage(type="human", content="What can you help me with?"),
    ]


def create_test_chat_response(include_tool_calls: bool = False) -> ChatResponse:
    """Create a test ChatResponse object."""
    tool_calls = None
    if include_tool_calls:
        tool_calls = [
            {
                "id": "test-tool-call-1",
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": '{"param": "value"}',
                },
            }
        ]

    return ChatResponse(
        done=True,
        message=Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="This is a test response from the chat model.",
                )
            ],
            tool_calls=tool_calls,
        ),
        finish_reason="stop",
    )


class TestChatAgent:
    """Test suite for ChatAgent class."""

    def test_chat_agent_initialization(self):
        """Test ChatAgent initialization with dependencies."""
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        node_metadata = create_test_node_metadata()

        agent = ChatAgent(
            pipeline_factory=pipeline_factory,
            profile=profile,
            node_metadata=node_metadata,
            priority=PipelinePriority.HIGH,
            stream=True,
        )

        # Verify initialization
        assert agent.pipeline_factory == pipeline_factory
        assert agent.profile == profile
        assert agent._node_metadata == node_metadata
        assert agent.priority == PipelinePriority.HIGH
        assert agent.stream is True

        # Verify BaseAgent initialization
        assert hasattr(agent, "logger")
        assert hasattr(agent, "_node_metadata")

    def test_chat_agent_initialization_defaults(self):
        """Test ChatAgent initialization with default values."""
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        node_metadata = create_test_node_metadata()

        agent = ChatAgent(
            pipeline_factory=pipeline_factory,
            profile=profile,
            node_metadata=node_metadata,
        )

        # Verify defaults
        assert agent.priority == PipelinePriority.MEDIUM
        assert agent.stream is False

    @pytest.mark.asyncio
    @patch("runner.run_pipeline")
    async def test_chat_completion_non_streaming(self, mock_run_pipeline):
        """Test non-streaming chat completion."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        node_metadata = create_test_node_metadata()
        agent = ChatAgent(pipeline_factory, profile, node_metadata, stream=False)

        # Mock pipeline context manager
        mock_pipeline = Mock()
        pipeline_factory.pipeline.return_value.__enter__ = Mock(
            return_value=mock_pipeline
        )
        pipeline_factory.pipeline.return_value.__exit__ = Mock(return_value=None)

        test_response = create_test_chat_response()
        mock_run_pipeline.return_value = test_response

        # Test data
        messages = create_test_langchain_messages()
        user_id = "test-user"

        # Execute
        result = await agent.chat_completion(messages, user_id)

        # Verify
        assert result == test_response
        mock_run_pipeline.assert_called_once_with(messages, mock_pipeline, None)
        pipeline_factory.pipeline.assert_called_once_with(
            profile, ChatResponse, PipelinePriority.MEDIUM, None
        )

    @pytest.mark.asyncio
    @patch("runner.stream_pipeline")
    async def test_chat_completion_streaming(self, mock_stream_pipeline):
        """Test streaming chat completion."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        node_metadata = create_test_node_metadata()
        agent = ChatAgent(pipeline_factory, profile, node_metadata, stream=True)

        # Mock pipeline context manager
        mock_pipeline = Mock()
        pipeline_factory.pipeline.return_value.__enter__ = Mock(
            return_value=mock_pipeline
        )
        pipeline_factory.pipeline.return_value.__exit__ = Mock(return_value=None)

        # Mock streaming chunks
        chunk1 = ChatResponse(
            done=False,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello ")],
            ),
        )
        chunk2 = ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="world!")],
                tool_calls=[
                    {"id": "test", "type": "function", "function": {"name": "test"}}
                ],
            ),
        )

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_stream_pipeline.return_value = mock_stream()

        # Test data
        messages = create_test_langchain_messages()
        user_id = "test-user"

        # Execute
        result = await agent.chat_completion(messages, user_id)

        # Verify accumulated response
        assert result.done is True
        assert len(result.message.content) == 1
        assert result.message.content[0].text == "Hello world!"
        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self):
        """Test chat completion with tools."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Mock pipeline context manager
        mock_pipeline = Mock()
        pipeline_factory.pipeline.return_value.__enter__ = Mock(
            return_value=mock_pipeline
        )
        pipeline_factory.pipeline.return_value.__exit__ = Mock(return_value=None)

        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        tools = [mock_tool]

        with patch("runner.run_pipeline") as mock_run_pipeline:
            test_response = create_test_chat_response(include_tool_calls=True)
            mock_run_pipeline.return_value = test_response

            # Test data
            messages = create_test_langchain_messages()
            user_id = "test-user"

            # Execute
            result = await agent.chat_completion(messages, user_id, tools=tools)

            # Verify tools were passed
            mock_run_pipeline.assert_called_once_with(messages, mock_pipeline, tools)
            assert result.message.tool_calls is not None

    @pytest.mark.asyncio
    async def test_chat_completion_with_circuit_breaker(self):
        """Test chat completion with circuit breaker."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        circuit_breaker = CircuitBreakerConfig(
            failure_threshold=5, timeout_seconds=60, retry_delay_seconds=10
        )

        # Mock pipeline context manager
        mock_pipeline = Mock()
        pipeline_factory.pipeline.return_value.__enter__ = Mock(
            return_value=mock_pipeline
        )
        pipeline_factory.pipeline.return_value.__exit__ = Mock(return_value=None)

        with patch("runner.run_pipeline") as mock_run_pipeline:
            test_response = create_test_chat_response()
            mock_run_pipeline.return_value = test_response

            # Execute
            messages = create_test_langchain_messages()
            result = await agent.chat_completion(
                messages, "test-user", circuit_breaker=circuit_breaker
            )

            # Verify circuit breaker was passed to pipeline factory
            pipeline_factory.pipeline.assert_called_once_with(
                profile, ChatResponse, PipelinePriority.MEDIUM, circuit_breaker
            )

    @patch("composer.agents.chat_agent.message_to_langchain_message")
    def test_convert_to_langchain_message(self, mock_convert):
        """Test message conversion to LangChain format."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Mock conversion
        expected_message = LangChainMessage(type="ai", content="Test response")
        mock_convert.return_value = expected_message

        # Test data
        response = create_test_chat_response()
        user_id = "test-user"

        # Execute
        result = agent.convert_to_langchain_message(response, user_id)

        # Verify mock was called and result returned
        mock_convert.assert_called_once_with(response.message)
        assert result == expected_message
        mock_convert.assert_called_once_with(response.message)

    def test_convert_to_langchain_message_empty_response(self):
        """Test message conversion with empty response."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Test with None response
        result = agent.convert_to_langchain_message(None, "test-user")

        # Verify fallback message
        assert result.type == "ai"
        assert "No response generated" in result.content

    @pytest.mark.asyncio
    @patch("composer.agents.chat_agent.message_to_langchain_message")
    @patch("runner.run_pipeline")
    async def test_chat_completion_with_conversion(
        self, mock_run_pipeline, mock_convert
    ):
        """Test combined chat completion and conversion."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Mock pipeline context manager
        mock_pipeline = Mock()
        pipeline_factory.pipeline.return_value.__enter__ = Mock(
            return_value=mock_pipeline
        )
        pipeline_factory.pipeline.return_value.__exit__ = Mock(return_value=None)

        # Mock response and conversion
        test_response = create_test_chat_response()
        expected_message = LangChainMessage(type="ai", content="Converted message")

        mock_run_pipeline.return_value = test_response
        mock_convert.return_value = expected_message

        # Execute
        messages = create_test_langchain_messages()
        result = await agent.chat_completion_with_conversion(messages, "test-user")

        # Verify
        assert result == expected_message
        mock_run_pipeline.assert_called_once()
        mock_convert.assert_called_once_with(test_response.message)

    def test_extract_tool_calls(self):
        """Test tool call extraction from LangChain message."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Test message with tool calls
        tool_calls = [{"id": "test", "type": "function", "function": {"name": "test"}}]
        message_with_tools = LangChainMessage(
            type="ai", content="Test message", tool_calls=tool_calls
        )

        result = agent.extract_tool_calls(message_with_tools)
        assert result == tool_calls

        # Test message without tool calls
        message_without_tools = LangChainMessage(type="ai", content="Test message")
        result = agent.extract_tool_calls(message_without_tools)
        assert result is None

    def test_has_tool_calls(self):
        """Test tool call detection."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Test message with tool calls
        tool_calls = [{"id": "test", "type": "function", "function": {"name": "test"}}]
        message_with_tools = LangChainMessage(
            type="ai", content="Test message", tool_calls=tool_calls
        )

        assert agent.has_tool_calls(message_with_tools) is True

        # Test message without tool calls
        message_without_tools = LangChainMessage(type="ai", content="Test message")
        assert agent.has_tool_calls(message_without_tools) is False

    @pytest.mark.asyncio
    async def test_chat_completion_error_handling(self):
        """Test error handling in chat completion."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Mock pipeline to raise exception
        pipeline_factory.pipeline.side_effect = Exception("Pipeline error")

        # Execute and verify exception is handled
        messages = create_test_langchain_messages()
        with pytest.raises(NodeExecutionError):
            await agent.chat_completion(messages, "test-user")

    @pytest.mark.asyncio
    async def test_streaming_completion_error_handling(self):
        """Test error handling in streaming completion."""
        # Setup
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        node_metadata = create_test_node_metadata()
        agent = ChatAgent(pipeline_factory, profile, node_metadata, stream=True)

        # Mock pipeline context manager
        mock_pipeline = Mock()
        pipeline_factory.pipeline.return_value.__enter__ = Mock(
            return_value=mock_pipeline
        )
        pipeline_factory.pipeline.return_value.__exit__ = Mock(return_value=None)

        with patch("runner.stream_pipeline") as mock_stream:
            # Mock stream to raise exception
            mock_stream.side_effect = Exception("Stream error")

            # Execute and verify exception is handled
            messages = create_test_langchain_messages()
            with pytest.raises(NodeExecutionError):
                await agent.chat_completion(messages, "test-user")

    def test_node_metadata_injection(self):
        """Test node metadata injection functionality."""
        pipeline_factory = Mock()
        profile = create_test_model_profile()
        agent = create_test_chat_agent(pipeline_factory, profile)

        # Create test metadata
        metadata = NodeMetadata(
            node_name="TestChatAgent",
            node_id="test-123",
            node_type="ChatAgent",
            execution_time=datetime.now(timezone.utc),
            user_id="test-user",
            conversation_id=42,
        )

        # Inject metadata
        agent.inject_node_metadata(metadata)

        # Verify metadata is stored
        assert agent.get_node_metadata() == metadata

        # Verify execution context can be updated
        agent.update_execution_context(operation="test", status="success")
        context = agent.get_execution_context()
        assert context["operation"] == "test"
        assert context["status"] == "success"
