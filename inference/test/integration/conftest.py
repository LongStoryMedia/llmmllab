"""Shared fixtures for database integration tests.

Spins up a TimescaleDB container via testcontainers, runs Alembic
migrations to ``head``, and provides an async engine / session factory
that every test module can use.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

# Ensure project root is on sys.path so db/ and models/ are importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

TIMESCALE_IMAGE = "timescale/timescaledb:latest-pg16"
TEST_DB = "testdb"
TEST_USER = "testuser"
TEST_PASSWORD = "testpass"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:  # type: ignore[misc]
    """Create a session-scoped event loop so all session-scoped async
    fixtures share the same loop (avoids 'fixture is not async-def' errors)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
def postgres_container():
    """Start a single TimescaleDB container for the entire test session."""
    with PostgresContainer(
        TIMESCALE_IMAGE,
        username=TEST_USER,
        password=TEST_PASSWORD,
        dbname=TEST_DB,
    ) as postgres:
        yield postgres


@pytest_asyncio.fixture(scope="session")
def sync_connection_string(postgres_container) -> str:
    """Build a psycopg2 connection string for Alembic (sync driver)."""
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    return f"postgresql+psycopg2://{TEST_USER}:{TEST_PASSWORD}@{host}:{port}/{TEST_DB}"


@pytest_asyncio.fixture(scope="session")
def async_connection_string(postgres_container) -> str:
    """Build an asyncpg connection string for the async engine."""
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    return f"postgresql+asyncpg://{TEST_USER}:{TEST_PASSWORD}@{host}:{port}/{TEST_DB}"


@pytest_asyncio.fixture(scope="session")
def _alembic_migrations(sync_connection_string: str) -> None:
    """Run Alembic migrations against the test container before any tests run."""
    import subprocess

    alembic_ini = os.path.join(_PROJECT_ROOT, "alembic.ini")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", alembic_ini, "upgrade", "head"],
        env={**os.environ, "DB_CONNECTION_STRING": sync_connection_string},
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Alembic upgrade failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


@pytest_asyncio.fixture(scope="session")
def engine(_alembic_migrations: None, async_connection_string: str) -> AsyncEngine:
    """Create a session-scoped async SQLAlchemy engine.

    Uses NullPool to avoid asyncpg's single-operation-per-connection
    limit causing 'another operation is in progress' errors when
    multiple fixtures share the same pooled connection.
    """
    eng = create_async_engine(
        async_connection_string,
        echo=False,
        poolclass=NullPool,
    )
    return eng


@pytest_asyncio.fixture(scope="session")
def session_factory(engine: AsyncEngine):
    """Create a session-scoped async session factory."""
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    return factory


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional session: each test rolls back so the
    database is clean for the next test."""
    async with session_factory() as s:
        yield s
        await s.rollback()


@pytest_asyncio.fixture
async def seed_test_user(session_factory) -> str:
    """Insert a minimal user row required by FK constraints (api_keys, memories).
    Returns the user_id."""
    from sqlalchemy import text

    user_id = "test-user-seed"
    async with session_factory() as s:
        await s.execute(
            text("INSERT INTO users(id) VALUES (:uid) ON CONFLICT (id) DO NOTHING"),
            {"uid": user_id},
        )
        await s.commit()
    return user_id


@pytest_asyncio.fixture
async def seed_conversation(seed_test_user: str, session_factory) -> int:
    """Insert a minimal conversation row required by messages/thoughts/tool_calls.
    Depends on seed_test_user so the user exists when the conversation trigger fires."""
    from sqlalchemy import text

    async with session_factory() as s:
        result = await s.execute(
            text("""
                INSERT INTO conversations(user_id, title)
                VALUES (:user_id, 'test-conversation')
                RETURNING id
            """),
            {"user_id": seed_test_user},
        )
        row = result.mappings().first()
        await s.commit()
    return row["id"] if row else 1


@pytest_asyncio.fixture
async def seed_message(seed_conversation: int, session_factory) -> int:
    """Insert a minimal message row required by triggers (thoughts, tool_calls).
    Returns the message_id."""
    from sqlalchemy import text

    async with session_factory() as s:
        result = await s.execute(
            text("""
                INSERT INTO messages(conversation_id, role)
                VALUES (:conv_id, 'user')
                RETURNING id
            """),
            {"conv_id": seed_conversation},
        )
        row = result.mappings().first()
        await s.commit()
    return row["id"] if row else 1


@pytest_asyncio.fixture(autouse=True)
async def _truncate_test_tables(session_factory):
    """Autouse fixture: truncate all test tables after each test to ensure
    isolation between tests."""
    from sqlalchemy import text

    yield

    async with session_factory() as s:
        # Truncate in dependency order (children first).
        # Only tables created by the Alembic migration are listed.
        tables = [
            "api_keys",
            "todos",
            "tool_calls",
            "thoughts",
            "message_contents",
            "documents",
            "images",
            "memories",
            "summaries",
            "messages",
            "conversations",
            "users",
        ]
        for tbl in tables:
            try:
                await s.execute(text(f"TRUNCATE TABLE {tbl} CASCADE"))
            except Exception:
                pass  # Table may not exist (e.g., skipped migration)
        await s.commit()


@pytest_asyncio.fixture
def api_key_storage(session_factory):
    from db.api_key_storage import ApiKeyStorage

    return ApiKeyStorage(session_factory)


@pytest_asyncio.fixture
def todo_storage(session_factory):
    from db.todo_storage import TodoStorage

    return TodoStorage(session_factory)


@pytest_asyncio.fixture
def thought_storage(session_factory):
    from db.thought_storage import ThoughtStorage

    return ThoughtStorage(session_factory)


@pytest_asyncio.fixture
def tool_call_storage(session_factory):
    from db.tool_call_storage import ToolCallStorage

    return ToolCallStorage(session_factory)


@pytest_asyncio.fixture
def message_content_storage(session_factory):
    from db.message_content_storage import MessageContentStorage

    return MessageContentStorage(session_factory)


@pytest_asyncio.fixture
def document_storage(session_factory):
    from db.document_storage import DocumentStorage

    return DocumentStorage(session_factory)


@pytest_asyncio.fixture
def image_storage(session_factory):
    from db.image_storage import ImageStorage

    return ImageStorage(session_factory)


@pytest_asyncio.fixture
def model_storage(session_factory):
    from db.model_storage import ModelStorage

    return ModelStorage(session_factory)


@pytest_asyncio.fixture
def summary_storage(session_factory):
    from db.summary_storage import SummaryStorage

    return SummaryStorage(session_factory)


@pytest_asyncio.fixture
def search_storage(session_factory):
    from db.search_storage import SearchStorage

    return SearchStorage(session_factory)


@pytest_asyncio.fixture
def memory_storage(session_factory):
    from db.memory_storage import MemoryStorage

    return MemoryStorage(session_factory)
