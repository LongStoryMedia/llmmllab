"""
Unit tests for composer/agents/base.py.

Tests BaseAgent class providing common functionality for all workflow agents.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from composer.agents.base import BaseAgent, get_message_count
from composer.models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ModelProfile,
    NodeMetadata,
    ChatResponse,
)


class TestGetMessageCount:
    """Tests for get_message_count helper function."""

    def test_count_string_message(self):
        """Test counting string message."""
        count = get_message_count("Hello")
        assert count == 1

    def test_count_message_object(self):
        """Test counting Message object."""
        message = MagicMock(spec=Message)
        count = get_message_count(message)
        assert count == 1

    def test_count_message_list(self):
        """Test counting list of messages."""
        messages = [MagicMock(spec=Message), MagicMock(spec=Message)]
        count = get_message_count(messages)
        assert count == 2

    def test_count_unknown_type(self):
        """Test counting unknown type (fallback)."""
        count = get_message_count({"unknown": "type"})
        assert count == 1


class TestBaseAgentInitialization:
    """Tests for BaseAgent initialization."""

    def test_initialization_with_defaults(self):
        """Test agent initialization with defaults."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)

        agent = BaseAgent(model=mock_model, profile=mock_profile)

        assert agent.model == mock_model
        assert agent.profile == mock_profile
        assert agent.tools == []
        assert agent.middleware == []
        assert agent.logger is not None
        assert agent.agent_id is not None

    def test_initialization_with_component_name(self):
        """Test agent initialization with custom component name."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)

        agent = BaseAgent(model=mock_model, profile=mock_profile, component_name="CustomAgent")

        assert agent.logger is not None
        # Logger should be bound with custom component name
        assert "CustomAgent" in str(agent.logger)

    def test_initialization_with_tools(self):
        """Test agent initialization with tools."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        mock_tool = MagicMock(spec=BaseTool)

        agent = BaseAgent(model=mock_model, profile=mock_profile, tools=[mock_tool])

        assert agent.tools == [mock_tool]

    def test_initialization_with_middleware(self):
        """Test agent initialization with middleware."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        mock_middleware = MagicMock(spec=AgentMiddleware)

        agent = BaseAgent(model=mock_model, profile=mock_profile, middleware=[mock_middleware])

        assert agent.middleware == [mock_middleware]

    def test_initialization_sets_node_metadata(self):
        """Test agent initialization sets node metadata."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)

        agent = BaseAgent(model=mock_model, profile=mock_profile)

        assert agent._node_metadata is not None
        assert agent._node_metadata.node_name == "UNSET"
        assert agent._node_metadata.node_id == "UNSET"
        assert agent._node_metadata.node_type == "BaseAgent"


class TestBindNodeMetadata:
    """Tests for bind_node_metadata method."""

    def test_bind_metadata(self):
        """Test binding node metadata to agent."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        metadata = NodeMetadata(
            node_name="test_node",
            node_id="node-123",
            node_type="TestNode",
            user_id="user-123",
        )

        result = agent.bind_node_metadata(metadata)

        assert result is agent
        assert agent._node_metadata.node_name == "test_node"
        assert agent._node_metadata.node_id == "node-123"
        assert agent._node_metadata.node_type == "TestNode"
        assert agent._node_metadata.user_id == "user-123"

    def test_bind_metadata_returns_self(self):
        """Test bind_node_metadata returns self for chaining."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        metadata = NodeMetadata(
            node_name="test_node",
            node_id="node-123",
            node_type="TestNode",
        )

        result = agent.bind_node_metadata(metadata)
        assert result is agent


class TestGetOrCreateAgent:
    """Tests for _get_or_create_agent method."""

    @pytest.fixture
    def mock_create_agent(self, mocker):
        """Mock create_agent function."""
        return mocker.patch('composer.agents.base.create_agent', return_value=MagicMock())

    @pytest.mark.asyncio
    async def test_create_agent_basic(self, mock_create_agent):
        """Test basic agent creation."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        system_prompt = "You are a helpful assistant."

        result = await agent._get_or_create_agent(system_prompt)

        assert result is not None
        mock_create_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_agent_with_tools(self, mock_create_agent):
        """Test agent creation with tools."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)
        mock_tool = MagicMock(spec=BaseTool)

        await agent._get_or_create_agent("System prompt", tools=[mock_tool])

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["tools"] == [mock_tool]

    @pytest.mark.asyncio
    async def test_create_agent_with_grammar(self, mock_create_agent):
        """Test agent creation with grammar."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        class TestGrammar(BaseModel):
            field: str

        await agent._get_or_create_agent("System prompt", grammar=TestGrammar)

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["response_format"] is not None

    @pytest.mark.asyncio
    async def test_create_agent_with_middleware(self, mock_create_agent):
        """Test agent creation with middleware."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)
        mock_middleware = MagicMock(spec=AgentMiddleware)

        await agent._get_or_create_agent("System prompt", middleware=[mock_middleware])

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["middleware"] == [mock_middleware]

    @pytest.mark.asyncio
    async def test_create_agent_with_metadata(self, mock_create_agent):
        """Test agent creation with metadata."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        metadata = NodeMetadata(
            node_name="test_node",
            node_id="node-123",
            node_type="TestNode",
        )

        await agent._get_or_create_agent("System prompt", metadata=metadata)

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["name"] == "test_node"


class TestLogOperation:
    """Tests for logging methods."""

    @pytest.fixture
    def mock_logger_info(self, mocker):
        """Mock logger.info method."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)
        mock_info = mocker.patch.object(agent.logger, 'info')
        mock_error = mocker.patch.object(agent.logger, 'error')
        return agent, mock_info, mock_error

    @pytest.fixture
    def agent_with_metadata(self, mocker):
        """Create agent with bound metadata."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        metadata = NodeMetadata(
            node_name="test_node",
            node_id="node-123",
            node_type="TestNode",
            user_id="user-123",
        )
        agent.bind_node_metadata(metadata)
        return agent

    def test_log_operation_start(self, agent_with_metadata, mocker):
        """Test logging operation start."""
        agent = agent_with_metadata
        mock_info = mocker.patch.object(agent.logger, 'info')

        agent._log_operation_start("test_operation", extra="value")

        # Logger should have been called with operation context
        mock_info.assert_called_once()
        call_args = mock_info.call_args
        assert "Starting test_operation" in str(call_args)

    def test_log_operation_success(self, agent_with_metadata, mocker):
        """Test logging operation success."""
        agent = agent_with_metadata
        mock_info = mocker.patch.object(agent.logger, 'info')

        agent._log_operation_success("test_operation", result="success")

        mock_info.assert_called_once()
        call_args = mock_info.call_args
        assert "Completed test_operation" in str(call_args)

    def test_log_operation_error(self, agent_with_metadata, mocker):
        """Test logging operation error."""
        agent = agent_with_metadata
        mock_error = mocker.patch.object(agent.logger, 'error')

        error = ValueError("Test error")
        agent._log_operation_error("test_operation", error, extra="value")

        mock_error.assert_called_once()
        call_args = mock_error.call_args
        assert "Failed test_operation" in str(call_args)


class TestHandleNodeError:
    """Tests for _handle_node_error method."""

    @pytest.fixture
    def agent_with_metadata(self, mocker):
        """Create agent with bound metadata."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        metadata = NodeMetadata(
            node_name="test_node",
            node_id="node-123",
            node_type="TestNode",
            user_id="user-123",
        )
        agent.bind_node_metadata(metadata)
        return agent

    def test_handle_node_error(self, agent_with_metadata, mocker):
        """Test handling node error."""
        agent = agent_with_metadata
        mock_error = mocker.patch.object(agent.logger, 'error')

        error = ValueError("Test error")
        agent._handle_node_error("test_operation", error)

        # Should log the error
        mock_error.assert_called_once()


class TestSeparateSystemPrompt:
    """Tests for _separate_system_prompt method."""

    def test_separate_system_prompt_basic(self):
        """Test separating system prompt from messages."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        mock_profile.system_prompt = "Default system prompt"
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        message = MagicMock(spec=Message)
        message.role = MessageRole.USER
        message.content = [MagicMock(spec=MessageContent, text="Hello")]

        system_prompt, convo = agent._separate_system_prompt([message])

        assert "Default system prompt" in system_prompt
        assert "The current date is" in system_prompt
        assert len(convo) == 1

    def test_separate_system_prompt_with_system_message(self):
        """Test separating system prompt with system message."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        mock_profile.system_prompt = "Default system prompt"
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        system_msg = MagicMock(spec=Message)
        system_msg.role = MessageRole.SYSTEM
        mock_content = MagicMock(spec=MessageContent)
        mock_content.type = MessageContentType.TEXT
        mock_content.text = "Custom system"
        system_msg.content = [mock_content]

        user_msg = MagicMock(spec=Message)
        user_msg.role = MessageRole.USER
        mock_user_content = MagicMock(spec=MessageContent)
        mock_user_content.type = MessageContentType.TEXT
        mock_user_content.text = "Hello"
        user_msg.content = [mock_user_content]

        system_prompt, convo = agent._separate_system_prompt([system_msg, user_msg])

        assert "Default system prompt" in system_prompt
        assert "Custom system" in system_prompt
        assert len(convo) == 1


class TestRun:
    """Tests for run method."""

    @pytest.fixture
    def mock_agent_creation(self, mocker):
        """Mock agent creation."""
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Response")]})
        return mocker.patch.object(BaseAgent, '_get_or_create_agent', return_value=mock_agent)

    @pytest.mark.asyncio
    async def test_run_basic(self, mock_agent_creation):
        """Test basic agent run."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        response = await agent.run("Hello")

        assert response.done is True
        assert response.message is not None

    @pytest.mark.asyncio
    async def test_run_with_tools(self, mock_agent_creation):
        """Test agent run with tools."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)
        mock_tool = MagicMock(spec=BaseTool)

        response = await agent.run("Hello", tools=[mock_tool])

        assert response.done is True

    @pytest.mark.asyncio
    async def test_run_with_grammar(self, mock_agent_creation):
        """Test agent run with grammar."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        class TestGrammar(BaseModel):
            field: str

        response = await agent.run("Hello", grammar=TestGrammar)

        assert response.done is True

    @pytest.mark.asyncio
    async def test_run_handles_error(self, mocker):
        """Test agent run handles errors."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        mocker.patch.object(agent, '_get_or_create_agent', side_effect=Exception("Test error"))

        response = await agent.run("Hello")

        assert response.done is True
        assert "Error during agent execution" in response.message.content[0].text


class TestRunStructured:
    """Tests for run_structured method."""

    @pytest.mark.asyncio
    async def test_run_structured_basic(self, mocker):
        """Test basic structured agent run."""
        mock_model = MagicMock()
        mock_profile = MagicMock(spec=ModelProfile)
        mock_profile.system_prompt = "System prompt"
        agent = BaseAgent(model=mock_model, profile=mock_profile)

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content='{"field": "value"}')]})
        mocker.patch.object(agent, '_get_or_create_agent', return_value=mock_agent)
        mocker.patch('composer.agents.base.parse_structured_output', return_value=MagicMock())

        class TestGrammar(BaseModel):
            field: str

        response = await agent.run_structured("Hello", grammar=TestGrammar)

        assert response is not None