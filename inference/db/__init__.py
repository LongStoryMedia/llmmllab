"""
Database module that initializes all storage components and provides access to them.
"""

import asyncpg
import os
from utils.logging import llmmllogger

from asyncpg import Pool

from .cache_storage import cache_storage
from .userconfig_storage import UserConfigStorage
from .conversation_storage import ConversationStorage
from .message_storage import MessageStorage
from .image_storage import ImageStorage
from .model_profile_storage import ModelProfileStorage
from .model_storage import ModelStorage
from .summary_storage import SummaryStorage
from .memory_storage import MemoryStorage
from .search_storage import SearchStorage
from .dynamic_tool_storage import DynamicToolStorage
from .queries import get_query
from typing import Optional, Protocol, Any, Callable, cast

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

            # Initialize all storage components
            self.user_config = UserConfigStorage(self.pool, get_query)
            self.conversation = ConversationStorage(self.pool, get_query)
            self.message = MessageStorage(self.pool, get_query)
            self.image = ImageStorage(self.pool, get_query)
            self.model_profile = ModelProfileStorage(self.pool, get_query)
            self.model = ModelStorage(self.pool, get_query)
            self.summary = SummaryStorage(self.pool, get_query)
            self.memory = MemoryStorage(self.pool, get_query)
            self.search = SearchStorage(self.pool, get_query)
            self.dynamic_tool = DynamicToolStorage(self.pool, get_query)

            self.initialized = True
            logger.info("Storage components initialized successfully")
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
            self.initialized = False

            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            self.initialized = False
            logger.info("Database connection pool closed")

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
