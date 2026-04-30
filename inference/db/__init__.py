"""
Database module that initializes all storage components and provides access to them.

Uses SQLAlchemy async engine + session factory instead of asyncpg.Pool.
Schema management is handled by Alembic (runs on startup).
"""

import os
from typing import Optional, Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from utils.logging import llmmllogger

from .cache_storage import cache_storage
from .engine import create_async_engine, create_session_factory, dispose_engine
from .userconfig_storage import UserConfigStorage
from .conversation_storage import ConversationStorage
from .message_storage import MessageStorage
from .image_storage import ImageStorage
from .model_storage import ModelStorage
from .summary_storage import SummaryStorage
from .memory_storage import MemoryStorage
from .search_storage import SearchStorage
from .thought_storage import ThoughtStorage
from .tool_call_storage import ToolCallStorage
from .message_content_storage import MessageContentStorage
from .document_storage import DocumentStorage
from .todo_storage import TodoStorage
from .checkpoint_storage import CheckpointStorage
from .api_key_storage import ApiKeyStorage
from .maintenance import maintenance_service

logger = llmmllogger.bind(component="db_init")


class Storage:
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self.user_config: Optional[UserConfigStorage] = None
        self.conversation: Optional[ConversationStorage] = None
        self.message: Optional[MessageStorage] = None
        self.image: Optional[ImageStorage] = None
        self.model: Optional[ModelStorage] = None
        self.summary: Optional[SummaryStorage] = None
        self.memory: Optional[MemoryStorage] = None
        self.search: Optional[SearchStorage] = None
        self.thought: Optional[ThoughtStorage] = None
        self.tool_call: Optional[ToolCallStorage] = None
        self.message_content: Optional[MessageContentStorage] = None
        self.document: Optional[DocumentStorage] = None
        self.todo: Optional[TodoStorage] = None
        self.checkpoint: Optional[CheckpointStorage] = None
        self.api_key: Optional[ApiKeyStorage] = None
        self.initialized = False

    async def initialize(self, connection_string: str):
        """Initialize the database engine, run Alembic migrations, and create storage components."""
        if self.initialized:
            return

        try:
            logger.info("Initializing SQLAlchemy database engine")
            self.engine = create_async_engine(connection_string)
            self.session_factory = create_session_factory(self.engine)
            logger.info("SQLAlchemy engine and session factory created")

            # Run Alembic migrations to ensure schema is up to date
            await self._run_alembic_upgrades(connection_string)

            # Initialize all storage components
            assert self.session_factory is not None
            factory = self.session_factory

            self.user_config = UserConfigStorage(factory)
            self.conversation = ConversationStorage(factory, self.user_config)
            self.image = ImageStorage(factory)
            self.model = ModelStorage(factory)
            self.summary = SummaryStorage(factory)
            self.memory = MemoryStorage(factory)
            self.search = SearchStorage(factory)
            self.thought = ThoughtStorage(factory)
            self.tool_call = ToolCallStorage(factory)
            self.message_content = MessageContentStorage(factory)
            self.document = DocumentStorage(factory)
            self.todo = TodoStorage(factory)
            self.checkpoint = CheckpointStorage(connection_string)
            self.api_key = ApiKeyStorage(factory)
            self.message = MessageStorage(
                factory,
                self.thought,
                self.tool_call,
                self.message_content,
                self.document,
            )

            self.initialized = True
            logger.info("Storage components initialized successfully")

            # Initialize and start the database maintenance service
            maintenance_interval = int(
                os.environ.get("DB_MAINTENANCE_INTERVAL_HOURS", "24")
            )
            assert self.engine is not None
            await maintenance_service.initialize(
                self.engine, factory, maintenance_interval
            )
            await maintenance_service.start_maintenance_schedule()
            logger.info("Database maintenance service started")

        except Exception as e:
            # Reset all components on failure
            self.engine = None
            self.session_factory = None
            self.user_config = None
            self.conversation = None
            self.message = None
            self.image = None
            self.model = None
            self.summary = None
            self.memory = None
            self.search = None
            self.thought = None
            self.tool_call = None
            self.message_content = None
            self.document = None
            self.todo = None
            self.initialized = False

            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close the database engine and its connection pool."""
        if self.engine:
            await dispose_engine()
            self.engine = None
            self.session_factory = None
            self.initialized = False
            logger.info("Database engine disposed")

    async def _run_alembic_upgrades(self, connection_string: str):
        """Run Alembic migrations to ensure schema is up to date."""
        from pathlib import Path  # pylint: disable=import-outside-toplevel

        alembic_ini = Path(__file__).resolve().parent.parent / "alembic.ini"
        if not alembic_ini.exists():
            logger.warning("alembic.ini not found, skipping migrations")
            return

        # Convert the async connection string to sync psycopg2 for Alembic
        sync_conn_str = connection_string.replace(
            "postgresql+asyncpg://", "postgresql://", 1
        )
        sync_conn_str = sync_conn_str.replace("postgres+asyncpg://", "postgresql://", 1)
        if sync_conn_str.startswith("postgresql://"):
            sync_conn_str = sync_conn_str.replace(
                "postgresql://", "postgresql+psycopg2://", 1
            )
        elif sync_conn_str.startswith("postgres://"):
            sync_conn_str = sync_conn_str.replace(
                "postgres://", "postgres+psycopg2://", 1
            )

        logger.info("Running Alembic migrations...")
        # Run Alembic as a subprocess to avoid thread-pool deadlocks
        # with asyncio.to_thread + Alembic's internal context managers.
        import asyncio as aio  # pylint: disable=import-outside-toplevel
        import sys as _sys  # pylint: disable=import-outside-toplevel

        project_root = str(alembic_ini.parent)
        try:
            proc = await aio.wait_for(
                aio.create_subprocess_exec(
                    _sys.executable,
                    "-m",
                    "alembic",
                    "-c",
                    str(alembic_ini),
                    "upgrade",
                    "head",
                    cwd=project_root,
                    env={**os.environ, "DB_CONNECTION_STRING": sync_conn_str},
                    stdout=aio.subprocess.PIPE,
                    stderr=aio.subprocess.PIPE,
                ),
                timeout=10,
            )
            stdout, stderr = await aio.wait_for(
                proc.communicate(),
                timeout=120,
            )
            if proc.returncode != 0:
                log_output = stderr.decode(errors="replace").strip()
                if not log_output:
                    log_output = stdout.decode(errors="replace").strip()
                logger.error(f"Alembic migrations failed: {log_output}")
                raise RuntimeError(
                    f"Alembic migrations failed (exit {proc.returncode}): {log_output}"
                )
            logger.info("Alembic migrations completed")
        except aio.TimeoutError:
            logger.error("Alembic migrations timed out after 120 seconds")
            raise TimeoutError(
                "Alembic migrations timed out after 120 seconds"
            ) from None

    def get_service[T](self, service: Optional[T]) -> T:
        """Get a storage service by name"""
        if not self.initialized:
            raise ValueError("Storage not initialized")

        if not service:
            raise ValueError(f"Unknown storage service: {service}")

        return service  # type: ignore[return-value]


# Create a singleton instance
storage = Storage()

__all__ = ["storage", "cache_storage"]
