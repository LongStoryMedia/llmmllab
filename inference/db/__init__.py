"""
Database module that initializes all storage components and provides access to them.
"""

import asyncpg
import os
from typing import Optional, Protocol, Any, Callable, cast

from asyncpg import Pool

from utils.logging import llmmllogger
from .cache_storage import cache_storage
from .userconfig_storage import UserConfigStorage
from .connection_recovery import init_recovery_manager
from .conversation_storage import ConversationStorage
from .message_storage import MessageStorage
from .image_storage import ImageStorage
from .model_profile_storage import ModelProfileStorage
from .model_storage import ModelStorage
from .summary_storage import SummaryStorage
from .memory_storage import MemoryStorage
from .search_storage import SearchStorage
from .dynamic_tool_storage import DynamicToolStorage
from .thought_storage import ThoughtStorage
from .analysis_storage import AnalysisStorage
from .tool_call_storage import ToolCallStorage
from .message_content_storage import MessageContentStorage
from .document_storage import DocumentStorage
from .todo_storage import TodoStorage
from .checkpoint_storage import CheckpointStorage
from .api_key_storage import ApiKeyStorage
from .queries import get_query
from .init_db import initialize_database
from .maintenance import maintenance_service

logger = llmmllogger.bind(component="db_init")


class StorageInterface(Protocol):
    """Protocol defining the interface for storage classes"""

    pool: Pool
    get_query: Callable[[str], str]

    def __init__(self, pool: Pool, get_query: Callable[[str], str]) -> None: ...


class Storage:
    def __init__(self):
        self.pool = None
        self.user_config = None
        self.conversation = None
        self.message = None
        self.image = None
        self.model_profile = None
        self.model = None
        self.summary = None
        self.memory = None
        self.search = None
        self.dynamic_tool = None
        self.thought = None
        self.analysis = None
        self.tool_call = None
        self.message_content = None
        self.document = None
        self.todo = None
        self.checkpoint = None
        self.api_key = None
        self.get_query = get_query
        self.initialized = False

    async def initialize(self, connection_string: str):
        """Initialize the database connection and storage components"""
        if self.initialized:
            return

        try:
            logger.info("Initializing database connection pool")
            # Avoid stale OID errors from server-side prepared statements by disabling or sizing the cache
            stmt_cache_size_str = os.environ.get("DB_STATEMENT_CACHE_SIZE", "0")
            try:
                stmt_cache_size = int(stmt_cache_size_str)
            except ValueError:
                stmt_cache_size = 0
            self.pool = await asyncpg.create_pool(
                connection_string, statement_cache_size=stmt_cache_size
            )
            logger.info(
                f"Database pool created (statement_cache_size={stmt_cache_size})"
            )

            # Initialize connection recovery manager
            init_recovery_manager(self.pool)

            # Proactively clear any stale connection state after pool creation
            await self._clear_stale_connection_state()

            # Initialize all storage components
            self.user_config = UserConfigStorage(self.pool, get_query)
            self.conversation = ConversationStorage(
                self.pool, get_query, self.user_config
            )
            self.image = ImageStorage(self.pool, get_query)
            self.model_profile = ModelProfileStorage(self.pool, get_query)
            self.model = ModelStorage(self.pool, get_query)
            self.summary = SummaryStorage(self.pool, get_query)
            self.memory = MemoryStorage(self.pool, get_query)
            self.search = SearchStorage(self.pool, get_query)
            self.dynamic_tool = DynamicToolStorage(self.pool, get_query)
            self.thought = ThoughtStorage(self.pool, get_query)
            self.analysis = AnalysisStorage(self.pool, get_query)
            self.tool_call = ToolCallStorage(self.pool, get_query)
            self.message_content = MessageContentStorage(self.pool, get_query)
            self.document = DocumentStorage(self.pool, get_query)
            self.todo = TodoStorage(self.pool, get_query)
            self.checkpoint = CheckpointStorage(self.pool, get_query)
            self.api_key = ApiKeyStorage(self.pool, get_query)
            self.message = MessageStorage(
                self.pool,
                get_query,
                self.thought,
                self.tool_call,
                self.message_content,
                self.analysis,
                self.document,
            )

            # Initialize checkpoint storage
            await self.checkpoint.initialize(connection_string)

            self.initialized = True
            logger.info("Storage components initialized successfully")

            await initialize_database(self.pool)

            # Initialize and start the database maintenance service
            maintenance_interval = int(
                os.environ.get("DB_MAINTENANCE_INTERVAL_HOURS", "24")
            )
            await maintenance_service.initialize(self.pool, maintenance_interval)
            await maintenance_service.start_maintenance_schedule()
            logger.info("Database maintenance service started")
            await self.model_profile.upsert_default_model_profiles()
            logger.info("Default model profiles ensured in database")

        except Exception as e:
            # Reset all components to None to ensure they're not partially initialized
            self.pool = None
            self.user_config = None
            self.conversation = None
            self.message = None
            self.image = None
            self.model_profile = None
            self.model = None
            self.summary = None
            self.memory = None
            self.search = None
            self.thought = None
            self.analysis = None
            self.tool_call = None
            self.message_content = None
            self.todo = None
            self.initialized = False

            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            self.initialized = False
            logger.info("Database connection pool closed")

    async def _clear_stale_connection_state(self):
        """Proactively clear any stale connection state on startup."""
        if not self.pool:
            return

        try:
            logger.info("Clearing stale connection state on startup...")
            pool = cast(asyncpg.Pool, self.pool)

            # Get one connection and clear its state
            async with pool.acquire() as conn:
                c = cast(asyncpg.Connection, conn)
                await c.execute("DISCARD ALL;")
                await c.reload_schema_state()

            logger.info("✅ Stale connection state cleared successfully")

        except Exception as e:
            logger.warning(
                f"Failed to clear stale connection state (non-critical): {e}"
            )

    def get_service[T](self, service: Optional[T]) -> T:
        """Get a storage service by name"""
        if not self.initialized:
            raise ValueError("Storage not initialized")

        if not service:
            raise ValueError(f"Unknown storage service: {service}")

        return cast(T, service)


# Create a singleton instance
storage = Storage()

__all__ = ["storage", "cache_storage"]
