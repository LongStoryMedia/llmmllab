"""
Unit tests for ChatNode functionality.
Tests workflow integration, state management, and ChatAgent orchestration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from typing import List

from composer.nodes.agents.chat_node import ChatNode
from composer.agents.chat_agent import ChatAgent
from composer.graph.state import WorkflowState
from models import (
    LangChainMessage,
    UserConfig,
    ModelProfile,
    NodeMetadata,
    CircuitBreakerConfig,
    PipelinePriority,
)
from runner import PipelineFactory
from composer.core.errors import NodeExecutionError


def create_test_user_config() -> UserConfig:
    """Create a test UserConfig object using Mock for complex nested objects."""
    mock_config = Mock(spec=UserConfig)
    mock_config.user_id = "test-user"
    mock_config.circuit_breaker = CircuitBreakerConfig(
        failure_threshold=5, timeout_seconds=60, retry_delay_seconds=10
    )
    return mock_config


def create_test_workflow_state() -> WorkflowState:
    """Create a test WorkflowState object."""
    return WorkflowState(
        user_id="test-user",
        conversation_id=42,
        user_config=create_test_user_config(),
        messages=[
            LangChainMessage(type="human", content="Hello"),
            LangChainMessage(type="ai", content="Hi there!"),
        ],
        available_tools=[],
        tool_calls=None,
        selected_workflows=set(),
        node_metadata={},
        retrieved_memories=[],
        web_search_results=[],
    )


def create_mock_chat_agent():
    """Create a mock ChatAgent."""
    mock_agent = Mock(spec=ChatAgent)
    mock_agent.profile = Mock()
    mock_agent.profile.model_name = "test-model"
    mock_agent.profile.profile_type = "primary"  # String value instead of Mock
    mock_agent.priority = Mock()
    mock_agent.priority.value = "medium"
    mock_agent.stream = True
    mock_agent.chat_completion_with_conversion = AsyncMock()
    mock_agent.extract_tool_calls = Mock()
    mock_agent.inject_node_metadata = Mock()
    return mock_agent


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


def create_test_chat_node(
    pipeline_factory=None,
    profile=None,
    priority=PipelinePriority.MEDIUM,
    stream=False,
    node_name="ChatNode",
) -> ChatNode:
    """Create a test ChatNode with all required dependencies."""
    if pipeline_factory is None:
        pipeline_factory = Mock(spec=PipelineFactory)
    if profile is None:
        profile = create_test_model_profile()

    return ChatNode(
        pipeline_factory=pipeline_factory,
        profile=profile,
        priority=priority,
        stream=stream,
        node_name=node_name,
    )


class TestChatNode:
    """Test suite for ChatNode class."""

    def test_chat_node_initialization(self):
        """Test ChatNode initialization."""
        pipeline_factory = Mock(spec=PipelineFactory)
        profile = create_test_model_profile()
        node = create_test_chat_node(
            pipeline_factory=pipeline_factory, profile=profile, node_name="TestChatNode"
        )

        # Verify initialization
        assert node.pipeline_factory == pipeline_factory
        assert node.profile == profile
        assert node.node_name == "TestChatNode"
        assert node.priority == PipelinePriority.MEDIUM
        assert node.stream is False
        assert node.chat_agent is None  # Created during execution

        # Verify BaseNode initialization
        assert hasattr(node, "node_id")
        assert hasattr(node, "logger")

    def test_chat_node_initialization_default_name(self):
        """Test ChatNode initialization with default name."""
        node = create_test_chat_node()

        assert node.node_name == "ChatNode"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful chat node execution."""
        # Setup
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Mock responses
        response_message = LangChainMessage(
            type="ai",
            content="This is a test response",
            tool_calls=[
                {"id": "test", "type": "function", "function": {"name": "test"}}
            ],
        )
        mock_agent.chat_completion_with_conversion.return_value = response_message
        mock_agent.extract_tool_calls.return_value = [
            {"id": "test", "type": "function"}
        ]

        # Mock assemble_context_messages
        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Execute
            result_state = await node.execute(state)

            # Verify agent was called correctly
            mock_agent.inject_node_metadata.assert_called_once()
            mock_agent.chat_completion_with_conversion.assert_called_once_with(
                messages=[LangChainMessage(type="human", content="Test message")],
                user_id="test-user",
                tools=None,  # When available_tools is empty, it passes None
                circuit_breaker=state.user_config.circuit_breaker,
                stream=None,
            )

            # Verify state updates
            assert len(result_state.messages) == 3  # Original 2 + new response
            assert result_state.messages[-1] == response_message
            assert result_state.tool_calls == [{"id": "test", "type": "function"}]

    @pytest.mark.asyncio
    async def test_execute_with_tools(self):
        """Test chat node execution with tools."""
        # Setup
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Add tools to state
        from models.tool import Tool

        test_tool = Tool(
            name="test_tool", description="A test tool for testing purposes"
        )
        state.available_tools = [test_tool]

        # Mock responses
        response_message = LangChainMessage(type="ai", content="Response with tools")
        mock_agent.chat_completion_with_conversion.return_value = response_message
        mock_agent.extract_tool_calls.return_value = None

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Execute
            result_state = await node.execute(state)

            # Verify tools were passed correctly
            mock_agent.chat_completion_with_conversion.assert_called_once_with(
                messages=[LangChainMessage(type="human", content="Test message")],
                user_id="test-user",
                tools=[test_tool],
                circuit_breaker=state.user_config.circuit_breaker,
                stream=None,
            )

            # Verify no tool calls in result
            assert result_state.tool_calls is None

    @pytest.mark.asyncio
    async def test_execute_metadata_injection(self):
        """Test node metadata injection during execution."""
        # Setup
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Mock responses
        response_message = LangChainMessage(type="ai", content="Test response")
        mock_agent.chat_completion_with_conversion.return_value = response_message
        mock_agent.extract_tool_calls.return_value = None

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Execute
            await node.execute(state)

            # Verify metadata injection
            mock_agent.inject_node_metadata.assert_called_once()

            # Get the metadata that was injected
            call_args = mock_agent.inject_node_metadata.call_args[0]
            metadata = call_args[0]

            # Verify metadata structure
            assert isinstance(metadata, NodeMetadata)
            assert metadata.user_id == "test-user"
            assert metadata.conversation_id == 42
            assert metadata.node_name == "ChatNode"
            assert metadata.model_name == "test-model"
            assert metadata.streaming is True

    @pytest.mark.asyncio
    async def test_execute_missing_user_id(self):
        """Test execution with missing user ID."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()
        state.user_id = None

        # Execute and verify error
        with pytest.raises(NodeExecutionError, match="User ID required"):
            await node.execute(state)

    @pytest.mark.asyncio
    async def test_execute_missing_user_config(self):
        """Test execution with missing user config."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()
        state.user_config = None

        # Execute and verify error
        with pytest.raises(NodeExecutionError, match="User config required"):
            await node.execute(state)

    @pytest.mark.asyncio
    async def test_execute_no_context_messages(self):
        """Test execution with no context messages."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = []

            # Execute and verify error
            with pytest.raises(
                NodeExecutionError, match="No context messages available"
            ):
                await node.execute(state)

    @pytest.mark.asyncio
    async def test_execute_agent_error(self):
        """Test execution when agent raises error."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Mock agent to raise exception
        mock_agent.chat_completion_with_conversion.side_effect = Exception(
            "Agent error"
        )

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Execute and verify error handling
            with pytest.raises(NodeExecutionError, match="Chat execution failed"):
                await node.execute(state)

    @pytest.mark.asyncio
    async def test_call_method_delegates_to_execute(self):
        """Test __call__ method delegates to execute."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Mock responses
        response_message = LangChainMessage(type="ai", content="Test response")
        mock_agent.chat_completion_with_conversion.return_value = response_message
        mock_agent.extract_tool_calls.return_value = None

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Execute using __call__
            result_state = await node(state)

            # Verify same behavior as execute
            assert len(result_state.messages) == 3
            assert result_state.messages[-1] == response_message
            mock_agent.chat_completion_with_conversion.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_logging(self):
        """Test logging during execution."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Mock responses
        response_message = LangChainMessage(type="ai", content="Test response")
        mock_agent.chat_completion_with_conversion.return_value = response_message
        mock_agent.extract_tool_calls.return_value = [{"id": "test"}]

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Mock logger to verify calls
            node.logger = Mock()

            # Execute
            await node.execute(state)

            # Verify logging calls
            assert node.logger.info.call_count >= 2  # At least start and success logs

            # Check specific log messages
            log_calls = [call[0][0] for call in node.logger.info.call_args_list]
            assert any("Executing chat completion" in msg for msg in log_calls)
            assert any("Chat completion successful" in msg for msg in log_calls)

    @pytest.mark.asyncio
    async def test_execute_with_empty_tool_calls(self):
        """Test execution when agent returns empty tool calls."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node()
        state = create_test_workflow_state()

        # Mock responses
        response_message = LangChainMessage(type="ai", content="No tools used")
        mock_agent.chat_completion_with_conversion.return_value = response_message
        mock_agent.extract_tool_calls.return_value = []  # Empty list

        with patch(
            "composer.nodes.agents.chat_node.assemble_context_messages"
        ) as mock_assemble:
            mock_assemble.return_value = [
                LangChainMessage(type="human", content="Test message")
            ]

            # Execute
            result_state = await node.execute(state)

            # Verify empty tool calls are handled correctly
            assert result_state.tool_calls == []

    def test_integration_with_base_node(self):
        """Test integration with BaseNode functionality."""
        mock_agent = create_mock_chat_agent()
        node = create_test_chat_node(node_name="IntegrationTestNode")

        # Verify BaseNode methods are available
        assert hasattr(node, "create_node_metadata")
        assert hasattr(node, "store_node_metadata")
        assert hasattr(node, "_validate_user_id")
        assert hasattr(node, "_ensure_user_config_initialized")

        # Verify node properties
        assert node.node_name == "IntegrationTestNode"
        assert node.node_id is not None
        assert len(node.node_id) == 8  # UUID truncated to 8 chars
