"""
Server module that initializes all storage components and provides access to them.
This serves as the gateway layer between composer and the database.
"""

import asyncpg
import os
from typing import Optional, Protocol, Any, Callable, cast

from asyncpg import Pool

from composer.utils.logging import llmmllogger
from .cache import cache
from .userconfig import UserConfig
from .connection_recovery import init_recovery_manager
from .conversation import Conversation
from .message import Message
from .image import Image
from .model_profile import ModelProfile
from .model import Model
from .summary import Summary
from .memory import Memory
from .search import Search
from .dynamic_tool import DynamicTool
from .thought import Thought
from .analysis import Analysis
from .tool_call import ToolCall
from .message_content import MessageContent
from .document import Document
from .todo import Todo
from .checkpoint import Checkpoint
from .api_key import ApiKey
from .queries import get_query
from .init_db import initialize_database
from .maintenance import maintenance_service

logger = llmmllogger.bind(component="server_init")


class ServerInterface(Protocol):
    """Protocol defining the interface for server storage classes"""

    pool: Pool
    get_query: Callable[[str], str]

    def __init__(self, pool: Pool, get_query: Callable[[str], str]) -> None:


class Server:
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
            self.user_config = UserConfig(self.pool, get_query)
            self.conversation = Conversation(
                self.pool, get_query, self.user_config
            )
            self.image = Image(self.pool, get_query)
            self.model_profile = ModelProfile(self.pool, get_query)
            self.model = Model(self.pool, get_query)
            self.summary = Summary(self.pool, get_query)
            self.memory = Memory(self.pool, get_query)
            self.search = Search(self.pool, get_query)
            self.dynamic_tool = DynamicTool(self.pool, get_query)
            self.thought = Thought(self.pool, get_query)
            self.analysis = Analysis(self.pool, get_query)
            self.tool_call = ToolCall(self.pool, get_query)
            self.message_content = MessageContent(self.pool, get_query)
            self.document = Document(self.pool, get_query)
            self.todo = Todo(self.pool, get_query)
            self.checkpoint = Checkpoint(self.pool, get_query)
            self.api_key = ApiKey(self.pool, get_query)
            self.message = Message(
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

            logger.info(" Stale connection state cleared successfully")

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
server = Server()

__all__ = ["server", "cache"]