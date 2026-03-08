"""
Unit tests for composer/graph/executor.py.

Tests generic workflow execution for streaming CompiledStateGraph outputs.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import Optional, AsyncIterator
from datetime import datetime, timezone
from pydantic import BaseModel

from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig

from composer.graph.executor import (
    WorkflowExecutor,
    create_executor,
    stream_workflow,
)
from composer.models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ChatResponse,
    Thought,
    ToolCall,
    GenerationState,
)


class TestWorkflowExecutorInitialization:
    """Tests for WorkflowExecutor initialization."""

    def test_initialization_with_default_logger(self):
        """Test executor initialization with default logger."""
        executor = WorkflowExecutor()

        assert executor.logger is not None
        assert executor.default_context == "workflow_executor"
        assert executor.content_parser is not None

    def test_initialization_with_custom_logger(self):
        """Test executor initialization with custom logger."""
        mock_logger = MagicMock()
        executor = WorkflowExecutor(logger=mock_logger, default_context="custom")

        assert executor.logger == mock_logger
        assert executor.default_context == "custom"


class TestCreateThreadConfig:
    """Tests for create_thread_config method."""

    def test_create_thread_config_basic(self):
        """Test basic thread config creation."""
        executor = WorkflowExecutor()

        config = executor.create_thread_config("thread-123")

        assert config == {"configurable": {"thread_id": "thread-123"}}

    def test_create_thread_config_with_additional(self):
        """Test thread config with additional parameters."""
        executor = WorkflowExecutor()

        config = executor.create_thread_config(
            "thread-123",
            additional_config={"run_name": "test-run"}
        )

        assert config["configurable"]["thread_id"] == "thread-123"
        assert config["configurable"]["run_name"] == "test-run"


class TestMakeResponse:
    """Tests for _make_response method."""

    def test_make_response_basic(self):
        """Test basic response creation."""
        executor = WorkflowExecutor()

        response = executor._make_response(
            conversation_id=1,
            state=GenerationState.RESPONDING,
            prev_state=None
        )

        assert response.message.conversation_id == 1
        assert response.state == GenerationState.RESPONDING
        assert response.prev_state is None
        assert response.done is False
        assert response.message.role == MessageRole.ASSISTANT

    def test_make_response_with_message_kwargs(self):
        """Test response with message kwargs."""
        executor = WorkflowExecutor()

        response = executor._make_response(
            conversation_id=1,
            state=GenerationState.RESPONDING,
            prev_state=None,
            message_kwargs={"content": [MessageContent(type=MessageContentType.TEXT, text="test")]}
        )

        assert response.message.content[0].text == "test"


class TestStreamWorkflow:
    """Tests for stream_workflow method."""

    @pytest.fixture
    def mock_workflow(self):
        """Mock CompiledStateGraph."""
        return MagicMock(spec=CompiledStateGraph)

    @pytest.fixture
    def mock_initial_state(self):
        """Mock initial state with conversation_id."""
        state = MagicMock()
        state.conversation_id = 1
        return state

    @pytest.mark.asyncio
    async def test_stream_workflow_empty_events(self, mock_workflow, mock_initial_state):
        """Test streaming with no events."""
        mock_workflow.astream_events = AsyncMock(return_value=[])
        executor = WorkflowExecutor()

        results = []
        async for event in executor.stream_workflow(mock_workflow, mock_initial_state):
            results.append(event)

        # Should still yield final done event
        assert len(results) == 1
        assert results[0].done is True

    @pytest.mark.asyncio
    async def test_stream_workflow_chat_model_stream(self, mock_workflow, mock_initial_state, mocker):
        """Test streaming with chat model stream events."""
        mock_event = MagicMock()
        mock_event.event = "on_chat_model_stream"
        mock_event.data = {
            "chunk": AIMessage(content="Hello"),
            "output": None
        }
        mock_event.name = ""
        mock_event.run_id = "run-123"

        mock_workflow.astream_events = AsyncMock(return_value=[mock_event])
        executor = WorkflowExecutor()

        results = []
        async for event in executor.stream_workflow(mock_workflow, mock_initial_state):
            results.append(event)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_stream_workflow_chat_model_end(self, mock_workflow, mock_initial_state, mocker):
        """Test streaming with chat model end events."""
        mock_event = MagicMock()
        mock_event.event = "on_chat_model_end"
        mock_event.data = {
            "chunk": None,
            "output": AIMessage(content="Hello", response_metadata={"finish_reason": "stop"})
        }
        mock_event.name = ""
        mock_event.run_id = "run-123"

        mock_workflow.astream_events = AsyncMock(return_value=[mock_event])
        executor = WorkflowExecutor()

        results = []
        async for event in executor.stream_workflow(mock_workflow, mock_initial_state):
            results.append(event)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_stream_workflow_structured_output(self, mock_workflow, mock_initial_state, mocker):
        """Test streaming with structured output."""
        class TestOutput(BaseModel):
            field: str

        mock_event = MagicMock()
        mock_event.event = "on_chain_end"
        mock_event.name = "structured_agent"
        mock_event.data = {
            "chunk": None,
            "output": TestOutput(field="value")
        }
        mock_event.run_id = "run-123"

        mock_workflow.astream_events = AsyncMock(return_value=[mock_event])
        executor = WorkflowExecutor()

        results = []
        async for event in executor.stream_workflow(mock_workflow, mock_initial_state):
            results.append(event)

        assert len(results) >= 1
        # Find the structured output event
        structured_events = [r for r in results if hasattr(r, 'message') and r.message.structured_output]
        assert len(structured_events) >= 1

    @pytest.mark.asyncio
    async def test_stream_workflow_tool_calls(self, mock_workflow, mock_initial_state, mocker):
        """Test streaming with tool calls."""
        # First, stream a tool call
        mock_tool_event = MagicMock()
        mock_tool_event.event = "on_tool_start"
        mock_tool_event.name = "test_tool"
        mock_tool_event.data = {
            "chunk": None,
            "output": None,
            "input": {"arg": "value"}
        }
        mock_tool_event.run_id = "tool-run-123"

        # Then, stream the tool end
        mock_tool_end_event = MagicMock()
        mock_tool_end_event.event = "on_tool_end"
        mock_tool_end_event.name = "test_tool"
        mock_tool_end_event.data = {
            "chunk": None,
            "output": ToolMessage(content="result", tool_call_id="tool-run-123")
        }
        mock_tool_end_event.run_id = "tool-run-123"

        mock_workflow.astream_events = AsyncMock(return_value=[mock_tool_event, mock_tool_end_event])
        executor = WorkflowExecutor()

        results = []
        async for event in executor.stream_workflow(mock_workflow, mock_initial_state):
            results.append(event)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_stream_workflow_error_handling(self, mock_workflow, mock_initial_state, mocker):
        """Test streaming with error."""
        mock_workflow.astream_events = AsyncMock(side_effect=Exception("Test error"))
        executor = WorkflowExecutor()

        results = []
        async for event in executor.stream_workflow(mock_workflow, mock_initial_state):
            results.append(event)

        assert len(results) == 1
        assert results[0].done is True
        assert results[0].finish_reason == "error"
        assert "Sorry, I could not complete your request" in results[0].message.content[0].text


class TestCreateExecutor:
    """Tests for create_executor factory function."""

    def test_create_executor(self):
        """Test creating executor with factory."""
        executor = create_executor()

        assert isinstance(executor, WorkflowExecutor)

    def test_create_executor_with_custom_logger(self):
        """Test creating executor with custom logger."""
        mock_logger = MagicMock()
        executor = create_executor(logger=mock_logger, context="custom")

        assert executor.logger == mock_logger
        assert executor.default_context == "custom"


class TestStreamWorkflow:
    """Tests for stream_workflow convenience function."""

    @pytest.fixture
    def mock_workflow(self):
        """Mock CompiledStateGraph."""
        return MagicMock(spec=CompiledStateGraph)

    @pytest.fixture
    def mock_initial_state(self):
        """Mock initial state."""
        state = MagicMock()
        state.conversation_id = 1
        return state

    @pytest.mark.asyncio
    async def test_stream_workflow_function(self, mock_workflow, mock_initial_state, mocker):
        """Test stream_workflow function."""
        mock_executor = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__aiter__ = lambda self: iter([])
        mock_executor.stream_workflow = mock_stream
        mocker.patch('composer.graph.executor.create_executor', return_value=mock_executor)

        result = []
        async for event in stream_workflow(mock_initial_state, mock_workflow, thread_id="thread-1"):
            result.append(event)

        mock_executor.stream_workflow.assert_called_once()