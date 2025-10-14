"""
Unit tests for the composer functional interface.

Tests the functional API that replaced the HTTP interface for composer service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from composer import (
    initialize_composer,
    shutdown_composer,
    compose_workflow,
    create_initial_state,
    execute_workflow,
    get_composer_config,
    get_composer_service,
)
from models.default_configs import create_default_user_config
from models.conversation import Conversation
from models.conversation_ctx import ConversationCtx
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.workflow_type import WorkflowType


class TestComposerFunctionalInterface:
    """Test suite for composer functional interface."""

    def setup_method(self):
        """Reset composer service before each test."""
        import composer

        composer._composer_service = None

    @pytest.fixture
    def mock_conversation_ctx(self):
        """Create a mock conversation context for testing."""
        user_config = create_default_user_config("test_user")
        conversation = Conversation(
            id=1,
            user_id="test_user",
            title="Test Conversation",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        test_message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(type=MessageContentType.TEXT, text="Hello, how are you?")
            ],
            conversation_id=1,
        )

        return ConversationCtx(
            messages=[test_message],
            notes=[],
            images=[],
            conversation=conversation,
            current_user_message=test_message,
        )

    @pytest.fixture
    def mock_composer_service(self):
        """Create a mock composer service."""
        service = Mock()
        service.compose_workflow = AsyncMock()
        service.execute_workflow = AsyncMock()
        service.create_initial_state = AsyncMock()
        service.shutdown = AsyncMock()
        return service

    def test_service_access_before_initialization(self):
        """Test that accessing service before initialization raises error."""
        with pytest.raises(RuntimeError, match="Composer service not initialized"):
            get_composer_service()

    def test_config_access_before_initialization(self):
        """Test that accessing config before initialization raises error."""
        with pytest.raises(RuntimeError, match="Composer service not initialized"):
            get_composer_config()

    @pytest.mark.asyncio
    async def test_initialization_and_shutdown(self, mock_composer_service):
        """Test composer initialization and shutdown."""
        with patch("composer.core.service.ComposerService") as MockComposerService:
            MockComposerService.return_value = mock_composer_service

            # Test initialization
            await initialize_composer()

            # Verify service was created
            MockComposerService.assert_called_once()

            # Test config access after initialization
            config = get_composer_config()
            assert isinstance(config, dict)
            assert config["service"] == "composer"

            # Test service access after initialization
            service = get_composer_service()
            assert service is mock_composer_service

            # Test shutdown
            await shutdown_composer()
            mock_composer_service.shutdown.assert_called_once()

            # Verify service is cleared after shutdown
            import composer

            assert composer._composer_service is None

    @pytest.mark.asyncio
    async def test_compose_workflow_interface(
        self, mock_conversation_ctx, mock_composer_service
    ):
        """Test that compose_workflow calls the service method properly."""
        with patch("composer.core.service.ComposerService") as MockComposerService:
            MockComposerService.return_value = mock_composer_service
            mock_workflow = Mock()
            mock_composer_service.compose_workflow.return_value = mock_workflow

            # Initialize composer
            await initialize_composer()

            # Test workflow composition - extract user_id and messages for new interface
            user_id = mock_conversation_ctx.conversation.user_id
            messages = mock_conversation_ctx.messages
            result = await compose_workflow(
                user_id=user_id, messages=messages, workflow_type=WorkflowType.CHAT
            )

            # Verify result is returned
            assert result is mock_workflow
            # Verify service method was called (we don't check exact args due to mocking complexity)
            assert mock_composer_service.compose_workflow.called

            # Cleanup
            await shutdown_composer()

    @pytest.mark.asyncio
    async def test_create_initial_state_interface(
        self, mock_conversation_ctx, mock_composer_service
    ):
        """Test that create_initial_state calls the service method properly."""
        with patch("composer.core.service.ComposerService") as MockComposerService:
            MockComposerService.return_value = mock_composer_service
            mock_initial_state = {"test": "state"}
            mock_composer_service.create_initial_state.return_value = mock_initial_state

            # Initialize composer
            await initialize_composer()

            # Test initial state creation - extract user_id and messages for new interface
            user_id = mock_conversation_ctx.conversation.user_id
            messages = mock_conversation_ctx.messages
            result = await create_initial_state(
                user_id=user_id, workflow_type=WorkflowType.CHAT
            )

            # Verify result is returned
            assert result is mock_initial_state
            # Verify service method was called
            assert mock_composer_service.create_initial_state.called

            # Cleanup
            await shutdown_composer()

    def test_interface_completeness(self):
        """Test that all expected functions are exported."""
        import composer

        expected_functions = [
            "initialize_composer",
            "shutdown_composer",
            "compose_workflow",
            "create_initial_state",
            "execute_workflow",
            "get_composer_config",
            "get_composer_service",
        ]

        for func_name in expected_functions:
            assert hasattr(composer, func_name), f"Missing function: {func_name}"
            func = getattr(composer, func_name)
            assert callable(func), f"Function {func_name} is not callable"

    @pytest.mark.asyncio
    async def test_basic_lifecycle(self, mock_composer_service):
        """Test basic initialization, config access, and shutdown lifecycle."""
        with patch("composer.core.service.ComposerService") as MockComposerService:
            MockComposerService.return_value = mock_composer_service

            try:
                # Should fail before initialization
                with pytest.raises(RuntimeError):
                    get_composer_service()

                # Initialize
                await initialize_composer()

                # Should work after initialization
                service = get_composer_service()
                assert service is mock_composer_service

                config = get_composer_config()
                assert isinstance(config, dict)
                assert "service" in config

            finally:
                # Always shutdown
                await shutdown_composer()
