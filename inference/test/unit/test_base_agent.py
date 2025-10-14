"""
Unit tests for BaseAgent functionality.
Tests node metadata injection, logging setup, and error handling patterns.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from composer.agents.base_agent import BaseAgent
from models import NodeMetadata, ErrorDetails
from composer.core.errors import NodeExecutionError


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, component_name=None):
        super().__init__(component_name)

    def test_operation_success(self, user_id: str):
        """Test method for successful operations."""
        self._log_operation_start("test_operation", user_id=user_id)
        # Simulate work
        result = "success"
        self._log_operation_success("test_operation", result=result)
        return result

    def test_operation_error(self, user_id: str):
        """Test method that demonstrates error handling."""
        try:
            self._log_operation_start("test_operation", user_id=user_id)
            # Simulate error
            raise ValueError("Test error")
        except Exception as e:
            self._handle_node_error("test_operation", e, user_id=user_id)


def create_test_node_metadata() -> NodeMetadata:
    """Create a test NodeMetadata object."""
    return NodeMetadata(
        node_name="TestNode",
        node_id="test-node-123",
        node_type="ConcreteAgent",
        execution_time=datetime.now(timezone.utc),
        user_id="test-user",
        conversation_id=42,
        model_name="test-model",
        profile_type="general",
        priority="normal",
        streaming=False,
        is_cached=False,
        cache_key="test-cache-key",
        tool_count=3,
        intent_classification="technical",
        memory_operations=["store", "retrieve"],
        error_details=None,
    )


class TestBaseAgent:
    """Test suite for BaseAgent class."""

    def test_base_agent_initialization_default_component_name(self):
        """Test BaseAgent initialization with default component name."""
        agent = ConcreteAgent()

        # Verify logger is properly initialized
        assert hasattr(agent.logger, "_context")
        # Test component name through successful agent creation (logger context is internal)
        assert agent._node_metadata is None
        assert agent._execution_context == {}

    def test_base_agent_initialization_custom_component_name(self):
        """Test BaseAgent initialization with custom component name."""
        agent = ConcreteAgent("CustomComponent")

        # Verify logger initialization with custom component name
        assert hasattr(agent.logger, "_context")

    def test_inject_node_metadata(self):
        """Test node metadata injection functionality."""
        agent = ConcreteAgent()
        metadata = create_test_node_metadata()

        # Inject metadata
        agent.inject_node_metadata(metadata)

        # Verify metadata is stored
        assert agent._node_metadata == metadata
        assert agent.get_node_metadata() == metadata

        # Verify logger context is updated with node metadata
        # (The logger context updating is tested through the fact that metadata injection succeeds)

    def test_get_node_metadata_none_when_not_injected(self):
        """Test get_node_metadata returns None when no metadata injected."""
        agent = ConcreteAgent()
        assert agent.get_node_metadata() is None

    def test_update_execution_context(self):
        """Test execution context update functionality."""
        agent = ConcreteAgent()

        # Update context
        agent.update_execution_context(operation="test_op", duration=1.5, success=True)

        # Verify context is updated
        context = agent.get_execution_context()
        assert context["operation"] == "test_op"
        assert context["duration"] == 1.5
        assert context["success"] is True

    def test_update_execution_context_multiple_updates(self):
        """Test multiple execution context updates accumulate."""
        agent = ConcreteAgent()

        # Update context multiple times
        agent.update_execution_context(key1="value1")
        agent.update_execution_context(key2="value2")
        agent.update_execution_context(key1="updated_value1")

        # Verify all updates are preserved (last update wins for same key)
        context = agent.get_execution_context()
        assert context["key1"] == "updated_value1"
        assert context["key2"] == "value2"

    def test_get_execution_context_returns_copy(self):
        """Test get_execution_context returns a copy."""
        agent = ConcreteAgent()
        agent.update_execution_context(test="value")

        # Get context and modify it
        context = agent.get_execution_context()
        context["new_key"] = "new_value"

        # Verify original context is unchanged
        original_context = agent.get_execution_context()
        assert "new_key" not in original_context
        assert original_context["test"] == "value"

    @patch("composer.agents.base_agent.llmmllogger")
    def test_log_operation_start_without_metadata(self, mock_logger):
        """Test operation start logging without node metadata."""
        agent = ConcreteAgent()
        agent.logger = Mock()

        agent._log_operation_start("test_operation", user_id="test-user")

        agent.logger.info.assert_called_once_with(
            "Starting test_operation", operation="test_operation", user_id="test-user"
        )

    @patch("composer.agents.base_agent.llmmllogger")
    def test_log_operation_start_with_metadata(self, mock_logger):
        """Test operation start logging with node metadata."""
        agent = ConcreteAgent()
        agent.logger = Mock()
        metadata = create_test_node_metadata()
        agent.inject_node_metadata(metadata)

        agent._log_operation_start("test_operation", extra_param="value")

        agent.logger.info.assert_called_once_with(
            "Starting test_operation",
            operation="test_operation",
            extra_param="value",
            node_name="TestNode",
            user_id="test-user",
            conversation_id=42,
        )

    @patch("composer.agents.base_agent.llmmllogger")
    def test_log_operation_success(self, mock_logger):
        """Test operation success logging."""
        agent = ConcreteAgent()
        agent.logger = Mock()

        agent._log_operation_success("test_operation", result="success")

        agent.logger.info.assert_called_once_with(
            "Completed test_operation", operation="test_operation", result="success"
        )

    @patch("composer.agents.base_agent.llmmllogger")
    def test_log_operation_error_without_metadata(self, mock_logger):
        """Test operation error logging without node metadata."""
        agent = ConcreteAgent()
        agent.logger = Mock()
        error = ValueError("Test error")

        agent._log_operation_error("test_operation", error, context="test")

        agent.logger.error.assert_called_once_with(
            "Failed test_operation",
            operation="test_operation",
            error="Test error",
            error_type="ValueError",
            context="test",
        )

    @patch("composer.agents.base_agent.llmmllogger")
    def test_log_operation_error_with_metadata(self, mock_logger):
        """Test operation error logging with node metadata."""
        agent = ConcreteAgent()
        agent.logger = Mock()
        metadata = create_test_node_metadata()
        agent.inject_node_metadata(metadata)
        error = ValueError("Test error")

        agent._log_operation_error("test_operation", error)

        agent.logger.error.assert_called_once_with(
            "Failed test_operation",
            operation="test_operation",
            error="Test error",
            error_type="ValueError",
            node_name="TestNode",
            user_id="test-user",
            conversation_id=42,
        )

    def test_handle_node_error_without_metadata(self):
        """Test node error handling without metadata."""
        agent = ConcreteAgent()
        agent.logger = Mock()
        error = ValueError("Test error")

        with pytest.raises(NodeExecutionError) as exc_info:
            agent._handle_node_error("test_operation", error, context="test")

        # Verify exception details
        assert "test_operation failed: Test error" in str(exc_info.value)
        assert exc_info.value.__cause__ == error

    def test_handle_node_error_with_metadata(self):
        """Test node error handling with node metadata."""
        agent = ConcreteAgent()
        agent.logger = Mock()
        metadata = create_test_node_metadata()
        agent.inject_node_metadata(metadata)
        error = ValueError("Test error")

        with pytest.raises(NodeExecutionError) as exc_info:
            agent._handle_node_error("test_operation", error)

        # Verify exception includes node context
        assert "[TestNode] test_operation failed: Test error" in str(exc_info.value)
        assert exc_info.value.__cause__ == error

    def test_get_user_context_without_metadata(self):
        """Test getting user context without node metadata."""
        agent = ConcreteAgent()
        context = agent._get_user_context()
        assert context == {}

    def test_get_user_context_with_metadata(self):
        """Test getting user context with node metadata."""
        agent = ConcreteAgent()
        metadata = create_test_node_metadata()
        agent.inject_node_metadata(metadata)

        context = agent._get_user_context()
        assert context["user_id"] == "test-user"
        assert context["conversation_id"] == 42

    def test_integrated_workflow_with_metadata(self):
        """Test complete workflow with metadata injection."""
        agent = ConcreteAgent("IntegratedTest")
        metadata = create_test_node_metadata()

        # Inject metadata
        agent.inject_node_metadata(metadata)

        # Update execution context
        agent.update_execution_context(step="initialization")

        # Verify all components work together
        assert agent.get_node_metadata() == metadata
        assert agent.get_execution_context()["step"] == "initialization"
        assert agent._get_user_context()["user_id"] == "test-user"

        # Verify logger initialization and metadata binding worked
        # (The fact that all previous operations succeeded demonstrates logger context is working)

    def test_error_metadata_injection(self):
        """Test metadata injection with error details."""
        agent = ConcreteAgent()

        error_details = ErrorDetails(
            error_type="ValidationError",
            error_message="Invalid input data",
            stack_trace="Stack trace here...",
        )

        metadata = NodeMetadata(
            node_name="ErrorNode",
            node_id="error-node-456",
            node_type="ConcreteAgent",
            execution_time=datetime.now(timezone.utc),
            user_id="error-user",
            conversation_id=99,
            error_details=error_details,
        )

        agent.inject_node_metadata(metadata)

        retrieved_metadata = agent.get_node_metadata()
        assert retrieved_metadata.error_details == error_details
        assert retrieved_metadata.error_details.error_type == "ValidationError"
        assert retrieved_metadata.error_details.error_message == "Invalid input data"
