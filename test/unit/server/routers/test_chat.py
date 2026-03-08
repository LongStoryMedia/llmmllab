"""
Unit tests for server/routers/chat.py.

Tests chat completion endpoint and file content transformation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from server.routers import chat
from server.models import Message, MessageContent, MessageContentType
from server.routers.chat import ChatCompletionBody


@pytest.fixture
def mock_composer(mocker):
    """Mock composer chat completion."""
    return mocker.patch('server.routers.chat.composer_chat_completion')


@pytest.fixture
def mock_storage(mocker):
    """Mock storage service."""
    # Create a mock storage service with proper get_service method
    storage = mocker.MagicMock()
    storage.message = mocker.MagicMock()

    # Create a mock service with async add_message
    mock_message_service = mocker.MagicMock()
    mock_message_service.add_message = AsyncMock(return_value=1)
    storage.get_service = mocker.MagicMock(return_value=mock_message_service)

    storage.initialized = True
    mocker.patch('server.routers.chat.storage', storage)
    return storage


@pytest.fixture
def mock_transform(mocker):
    """Mock file content transformation."""
    return mocker.patch('server.routers.chat.transform_file_content_to_documents')


@pytest.fixture
def mock_request(mocker):
    """Create a mock request with user ID and request ID."""
    request = MagicMock()
    request.headers = {}
    request.state = MagicMock()
    request.state.auth = {'user_id': 'test-user'}
    request.state.request_id = 'test-request-id'

    def get_user_id(req):
        return req.state.auth.get('user_id')

    def get_request_id(req):
        return req.state.request_id

    mocker.patch('server.routers.chat.get_user_id', get_user_id)
    mocker.patch('server.routers.chat.get_request_id', get_request_id)
    return request


@pytest.fixture
def chat_body():
    """Create a chat completion body."""
    return ChatCompletionBody(
        message=Message(
            id=1,
            conversation_id=1,
            role="user",
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello")]
        ),
        model_name=None,
        response_format=None
    )


class TestChatCompletion:
    """Tests for chat completion endpoint."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, mock_composer, mock_storage, mock_transform, mock_request, chat_body):
        """Test successful chat completion."""
        # Mock composer response - return an async iterator
        async def mock_chat_stream():
            yield '{"model": "test", "message": "Hello"}'

        mock_composer.return_value = mock_chat_stream()

        # Mock transformation
        mock_transform.return_value = chat_body.message

        # Import and call the endpoint
        from server.routers.chat import chat_completion

        response = await chat_completion(chat_body, mock_request)

        # Verify storage was called
        mock_storage.get_service.assert_called_once()
        mock_storage.get_service.return_value.add_message.assert_called_once()

        # Verify composer was called
        mock_composer.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_with_file_content(self, mock_composer, mock_storage, mock_transform, mock_request, chat_body):
        """Test chat completion with file content transformation."""
        # Mock transformation to return transformed message
        transformed = Message(
            id=1,
            conversation_id=1,
            role="user",
            content=[MessageContent(type=MessageContentType.TEXT, text="Transformed")]
        )
        mock_transform.return_value = transformed

        # Mock composer response
        async def mock_chat_stream():
            yield '{"model": "test", "message": "Transformed response"}'

        mock_composer.return_value = mock_chat_stream()

        from server.routers.chat import chat_completion

        response = await chat_completion(chat_body, mock_request)

        # Verify transformation was called
        mock_transform.assert_called_once()

class TestAdminEndpoint:
    """Tests for admin-only endpoints."""

    @pytest.mark.asyncio
    async def test_admin_endpoint_requires_admin(self, mock_composer, mock_storage, mock_request, chat_body):
        """Test admin endpoint requires admin access."""
        mock_request.state.auth = {'user_id': 'test-user', 'is_admin': False}

        from server.routers.chat import admin_only

        with pytest.raises(HTTPException) as exc_info:
            await admin_only(mock_request)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_endpoint_success(self, mock_composer, mock_storage, mock_request, chat_body):
        """Test admin endpoint succeeds with admin access."""
        mock_request.state.auth = {'user_id': 'admin', 'is_admin': True}

        from server.routers.chat import admin_only

        response = await admin_only(mock_request)

        assert response is not None
        assert response["status"] == "success"
        assert response["message"] == "Admin access granted"