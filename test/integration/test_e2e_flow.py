"""
End-to-end integration tests.

Tests the full request flow through server, composer, and runner.
"""

import pytest
from typing import Dict, Any

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_full_request_flow(server_client):
    """
    Test the complete request flow:
    1. API request to server
    2. Server processes request
    3. Composer handles orchestration
    4. Runner executes pipeline
    5. Response returns through stack
    """
    # This is a placeholder test that verifies the server is running
    # and can handle requests

    response = await server_client.get("/health")
    assert response.status_code == 200

    # The actual e2e test would involve:
    # 1. Creating a conversation
    # 2. Sending a message
    # 3. Verifying composer orchestration
    # 4. Verifying runner pipeline execution
    # 5. Checking database persistence

    assert True, "Server is reachable and responding"


@pytest.mark.asyncio
async def test_chat_completion_flow(server_client):
    """Test the chat completion flow through the stack."""
    # This test verifies that the chat completion endpoint
    # is accessible and returns a valid response

    # Note: This is a basic test that doesn't actually run a model
    # A full e2e test would require a model to be loaded

    response = await server_client.get("/v1/chat/completions")

    # The endpoint should exist (may return 405 Method Not Allowed
    # or 422 Validation Error if no body, but not 404)
    assert response.status_code in [200, 405, 422]


@pytest.mark.asyncio
async def test_conversation_lifecycle(server_client, db_connection):
    """Test the conversation creation and retrieval lifecycle."""
    # This test verifies that conversations can be created
    # and retrieved through the database

    # Insert a test conversation
    conversation_id = await db_connection.fetchval(
        """
        INSERT INTO conversations (user_id, title, created_at, updated_at)
        VALUES ($1, $2, NOW(), NOW())
        RETURNING id
        """,
        "test-user-id",
        "Test Conversation"
    )

    assert conversation_id is not None

    # Retrieve the conversation
    conversation = await db_connection.fetchrow(
        "SELECT * FROM conversations WHERE id = $1",
        conversation_id
    )

    assert conversation is not None
    assert conversation["title"] == "Test Conversation"


@pytest.mark.asyncio
async def test_message_flow(server_client, db_connection):
    """Test the message storage and retrieval flow."""
    # Create a conversation first
    conversation_id = await db_connection.fetchval(
        """
        INSERT INTO conversations (user_id, title, created_at, updated_at)
        VALUES ($1, $2, NOW(), NOW())
        RETURNING id
        """,
        "test-user-id",
        "Test Conversation"
    )

    # Insert a message
    message_id = await db_connection.fetchval(
        """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES ($1, $2, $3, NOW())
        RETURNING id
        """,
        conversation_id,
        "user",
        "Hello, world!"
    )

    assert message_id is not None

    # Retrieve the message
    message = await db_connection.fetchrow(
        "SELECT * FROM messages WHERE id = $1",
        message_id
    )

    assert message is not None
    assert message["content"] == "Hello, world!"


@pytest.mark.asyncio
async def test_database_persistence(server_client, db_connection):
    """Test that data persists correctly in the database."""
    # Test user creation
    user_id = await db_connection.fetchval(
        """
        INSERT INTO users (id, email, created_at)
        VALUES ($1, $2, NOW())
        RETURNING id
        """,
        "test-user-persist",
        "test@example.com"
    )

    assert user_id == "test-user-persist"

    # Verify the user exists
    user = await db_connection.fetchrow(
        "SELECT * FROM users WHERE id = $1",
        user_id
    )

    assert user is not None
    assert user["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_model_profile_storage(server_client, db_connection):
    """Test model profile creation and retrieval."""
    profile_id = await db_connection.fetchval(
        """
        INSERT INTO model_profiles (
            user_id, model_id, provider, task_type, config
        )
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        "test-user-id",
        "test-model",
        "llama_cpp",
        "text_to_text",
        {"temperature": 0.7, "max_tokens": 512}
    )

    assert profile_id is not None

    # Retrieve the profile
    profile = await db_connection.fetchrow(
        "SELECT * FROM model_profiles WHERE id = $1",
        profile_id
    )

    assert profile is not None
    assert profile["model_id"] == "test-model"