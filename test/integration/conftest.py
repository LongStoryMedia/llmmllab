"""
Pytest configuration for integration tests.

This module provides fixtures and configuration for integration testing
the server, composer, and runner components.
"""

import asyncio
import os
import pytest
import asyncpg
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient

# Set environment variables before imports
os.environ["DB_CONNECTION_STRING"] = os.environ.get(
    "DB_CONNECTION_STRING",
    "postgresql://postgres:postgres@localhost:5432/llmmll_test?sslmode=disable"
)
os.environ["TEST_MODE"] = "true"
os.environ["DISABLE_AUTH"] = "true"


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create a database connection pool for testing."""
    connection_string = os.environ.get(
        "DB_CONNECTION_STRING",
        "postgresql://postgres:postgres@localhost:5432/llmmll_test?sslmode=disable"
    )

    pool = await asyncpg.create_pool(connection_string)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture(scope="function")
async def db_connection(db_pool: asyncpg.Pool) -> AsyncGenerator[asyncpg.Connection, None]:
    """Create a database connection with transaction rollback."""
    conn = await db_pool.acquire()
    tran = conn.transaction()
    await tran.start()

    try:
        yield conn
    finally:
        await tran.rollback()
        await db_pool.release(conn)


@pytest.fixture(scope="function")
async def clean_database(db_connection: asyncpg.Connection) -> AsyncGenerator[None, None]:
    """Clean the database before and after each test."""
    # Clean the database before test
    await db_connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
    await db_connection.execute("CREATE SCHEMA public")

    yield

    # Clean after test
    await db_connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
    await db_connection.execute("CREATE SCHEMA public")


@pytest.fixture(scope="session")
async def server_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for server testing."""
    base_url = os.environ.get("SERVER_URL", "http://localhost:8000")

    async with AsyncClient(base_url=base_url, timeout=60.0) as client:
        yield client


@pytest.fixture
def test_user_id() -> str:
    """Get a test user ID."""
    return os.environ.get("TEST_USER_ID", "test-user-id-12345")


@pytest.fixture
def api_key() -> str:
    """Get the API key for testing."""
    return os.environ.get("SERVER_API_KEY", "test-api-key")


@pytest.fixture
def model_profile_config() -> Dict[str, Any]:
    """Get a default model profile configuration for testing."""
    return {
        "model_id": "test-model",
        "provider": "llama_cpp",
        "task_type": "text_to_text",
        "model_config": {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
        }
    }