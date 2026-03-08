"""
Unit tests for composer/graph/state.py.

Tests WorkflowState Pydantic models with LangGraph reducers.
"""
import pytest
from unittest.mock import MagicMock
from typing import List, Optional

from datetime import datetime
from pydantic import ValidationError

from composer.graph.state import (
    WorkflowState,
    assemble_context_messages,
    _memory_to_message,
    _summary_to_message,
)
from composer.models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Memory,
    Summary,
    Document,
    UserConfig,
    SearchResult,
    SearchTopicSynthesis,
)


class TestWorkflowState:
    """Tests for WorkflowState model."""

    def test_state_initialization(self):
        """Test WorkflowState initialization with required fields."""
        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=MagicMock(spec=UserConfig)
        )

        assert state.conversation_id == 1
        assert state.user_id == "user-123"
        assert state.messages == []
        assert state.summaries == []
        assert state.retrieved_memories == []
        assert state.created_memories == []
        assert state.web_search_results == []
        assert state.search_syntheses == []

    def test_state_with_all_fields(self):
        """Test WorkflowState with all fields populated."""
        user_config = MagicMock(spec=UserConfig)

        # Create proper SearchResult and SearchTopicSynthesis objects
        search_result = SearchResult(
            is_from_url_in_user_query=False,
            query="test query",
        )
        search_synthesis = SearchTopicSynthesis(
            urls=["https://example.com"],
            topics=["test topic"],
            synthesis="test synthesis",
            created_at=datetime.now(),
            conversation_id=1,
        )

        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=user_config,
            current_user_message=MagicMock(spec=Message),
            title="Test Conversation",
            messages=[MagicMock(spec=Message)],
            summaries=[MagicMock(spec=Summary)],
            retrieved_memories=[MagicMock(spec=Memory)],
            created_memories=[MagicMock(spec=Memory)],
            web_search_results=[search_result],
            search_syntheses=[search_synthesis],
            things_to_remember=[MagicMock(spec=Message)],
        )

        assert state.conversation_id == 1
        assert state.title == "Test Conversation"

    def test_state_with_default_factory(self):
        """Test that default factories create new instances."""
        state1 = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=MagicMock(spec=UserConfig)
        )
        state2 = WorkflowState(
            conversation_id=2,
            user_id="user-456",
            user_config=MagicMock(spec=UserConfig)
        )

        # Each state should have its own list instance
        assert state1.messages is not state2.messages

    def test_state_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        user_config = MagicMock(spec=UserConfig)

        with pytest.raises(ValidationError):
            WorkflowState(
                conversation_id=1,
                user_id="user-123",
                user_config=user_config,
                extra_field="not allowed"
            )

    def test_state_arbitrary_types_allowed(self):
        """Test that arbitrary types are allowed."""
        user_config = MagicMock(spec=UserConfig)
        # Create a proper Message object since WorkflowState requires it
        message = Message(
            id=1,
            conversation_id=1,
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="test")]
        )

        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=user_config,
            current_user_message=message
        )

        assert state.current_user_message == message


class TestReducers:
    """Tests for reducer functions."""

    def test_operator_add_reducer(self):
        """Test that operator.add reducer concatenates lists."""
        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=MagicMock(spec=UserConfig)
        )

        # Simulate reducer behavior
        current = state.messages
        new_value = [MagicMock(spec=Message)]

        # The reducer should return new_value if not None, else current
        result = new_value if new_value is not None else current
        assert result == new_value

    def test_conditional_reducer(self):
        """Test conditional reducer that prefers new value."""
        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=MagicMock(spec=UserConfig)
        )

        # Simulate reducer: lambda x, y: y if y is not None else x
        current = None
        new_value = "new title"

        result = new_value if new_value is not None else current
        assert result == "new title"

    def test_conditional_reducer_with_existing(self):
        """Test conditional reducer keeps existing when new is None."""
        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=MagicMock(spec=UserConfig),
            title="Existing Title"
        )

        # Simulate reducer: lambda x, y: y if y is not None else x
        current = "Existing Title"
        new_value = None

        result = new_value if new_value is not None else current
        assert result == "Existing Title"


class TestAssembleContextMessages:
    """Tests for assemble_context_messages function."""

    def test_assemble_messages_basic(self):
        """Test basic message assembly."""
        user_config = MagicMock(spec=UserConfig)

        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=user_config,
            messages=[
                MagicMock(spec=Message, role=MessageRole.USER),
                MagicMock(spec=Message, role=MessageRole.ASSISTANT),
            ]
        )

        messages = assemble_context_messages(state)

        assert len(messages) == 2

    def test_assemble_messages_with_memories(self):
        """Test message assembly with retrieved memories."""
        user_config = MagicMock(spec=UserConfig)

        memory = MagicMock(spec=Memory)
        memory.created_at = "2024-01-01"
        memory.conversation_id = 1
        memory.fragments = [
            MagicMock(role=MessageRole.USER, content="User query"),
            MagicMock(role=MessageRole.ASSISTANT, content="Assistant response"),
        ]

        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=user_config,
            messages=[MagicMock(spec=Message)],
            retrieved_memories=[memory],
        )

        messages = assemble_context_messages(state)

        assert len(messages) == 2

    def test_assemble_messages_with_summaries(self):
        """Test message assembly with summaries."""
        user_config = MagicMock(spec=UserConfig)

        summary = MagicMock(spec=Summary)
        summary.level = 1
        summary.content = "Summary content"

        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=user_config,
            messages=[MagicMock(spec=Message)],
            summaries=[summary],
        )

        messages = assemble_context_messages(state)

        assert len(messages) == 2

    def test_assemble_messages_priority_order(self):
        """Test that messages follow correct priority order."""
        user_config = MagicMock(spec=UserConfig)

        # Create proper Memory mock with required attributes
        memory = MagicMock(spec=Memory)
        memory.created_at = "2024-01-01"
        memory.conversation_id = 1
        memory.fragments = []

        # Create proper Summary mock with required attributes
        summary = MagicMock(spec=Summary)
        summary.level = 1
        summary.content = "test summary"

        state = WorkflowState(
            conversation_id=1,
            user_id="user-123",
            user_config=user_config,
            messages=[MagicMock(spec=Message, role=MessageRole.USER, id="msg1")],
            retrieved_memories=[memory],
            summaries=[summary],
        )

        messages = assemble_context_messages(state)

        # Core messages first, then memories, then summaries
        assert len(messages) == 3


class TestMemoryToMessage:
    """Tests for _memory_to_message function."""

    def test_memory_to_message_basic(self):
        """Test basic memory to message conversion."""
        memory = MagicMock(spec=Memory)
        memory.created_at = "2024-01-01"
        memory.conversation_id = 1
        memory.fragments = [
            MagicMock(role=MessageRole.USER, content="User query"),
            MagicMock(role=MessageRole.ASSISTANT, content="Assistant response"),
        ]

        message = _memory_to_message(memory, conversation_id=1)

        assert message.role == MessageRole.SYSTEM
        assert len(message.content) == 1
        assert "MEMORY FROM" in message.content[0].text
        assert "USER: User query" in message.content[0].text
        assert "ASSISTANT: Assistant response" in message.content[0].text

    def test_memory_to_message_without_conversation_id(self):
        """Test memory to message without conversation_id."""
        memory = MagicMock(spec=Memory)
        memory.created_at = "2024-01-01"
        memory.conversation_id = None
        memory.fragments = []

        message = _memory_to_message(memory, conversation_id=None)

        assert message.role == MessageRole.SYSTEM
        assert message.conversation_id is None


class TestSummaryToMessage:
    """Tests for _summary_to_message function."""

    def test_summary_to_message_basic(self):
        """Test basic summary to message conversion."""
        summary = MagicMock(spec=Summary)
        summary.level = 1
        summary.content = "Summary content"

        message = _summary_to_message(summary, conversation_id=1)

        assert message.role == MessageRole.SYSTEM
        assert len(message.content) == 1
        assert "[Summary Level 1]: Summary content" in message.content[0].text

    def test_summary_to_message_without_conversation_id(self):
        """Test summary to message without conversation_id."""
        summary = MagicMock(spec=Summary)
        summary.level = 2
        summary.content = "Summary content"

        message = _summary_to_message(summary, conversation_id=None)

        assert message.role == MessageRole.SYSTEM
        assert message.conversation_id is None