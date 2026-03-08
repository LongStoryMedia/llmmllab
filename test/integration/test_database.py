"""
Database integration tests.

Tests the database connection and schema initialization.
"""

import pytest
import asyncpg
import os

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_database_connection(db_pool: asyncpg.Pool):
    """Test that we can connect to the database."""
    assert db_pool is not None
    assert db_pool._initialized


@pytest.mark.asyncio
async def test_database_extensions(db_connection: asyncpg.Connection):
    """Test that required database extensions are installed."""
    extensions = await db_connection.fetch(
        "SELECT extname FROM pg_extension WHERE extname IN ('timescaledb', 'vector')"
    )

    extension_names = {ext["extname"] for ext in extensions}
    assert "timescaledb" in extension_names, "timescaledb extension not found"
    assert "vector" in extension_names, "vector extension not found"


@pytest.mark.asyncio
async def test_database_schema(db_connection: asyncpg.Connection):
    """Test that required tables exist in the database."""
    tables = await db_connection.fetch(
        """
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
        """
    )

    table_names = {t["tablename"] for t in tables}

    # Check for required tables
    required_tables = [
        "users",
        "api_keys",
        "model_profiles",
        "conversations",
        "messages",
        "message_contents",
        "documents",
        "summaries",
        "memories",
        "images",
        "tools",
        "search_topic_synthesis",
        "thoughts",
        "analysis",
        "tool_calls",
    ]

    for table in required_tables:
        assert table in table_names, f"Required table {table} not found in database"


@pytest.mark.asyncio
async def test_database_connection_string():
    """Test that the database connection string is properly configured."""
    connection_string = os.environ.get("DB_CONNECTION_STRING")

    assert connection_string is not None, "DB_CONNECTION_STRING is not set"
    assert "postgresql://" in connection_string, "Invalid connection string format"
    assert "localhost" in connection_string or "db" in connection_string, \
        "Connection string should point to localhost or db"