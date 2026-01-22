"""
Simplified checkpoint storage service using LangGraph AsyncPostgresSaver.
Provides clean factory methods for creating checkpointers without unnecessary abstraction.
"""

from typing import Optional
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="checkpoint_storage")


class CheckpointStorage:
    """
    Simplified checkpoint storage following LangGraph best practices.

    Provides factory methods for creating AsyncPostgresSaver instances
    without over-engineering the abstraction layer.
    """

    def __init__(self, pool=None, get_query=None):
        """
        Initialize checkpoint storage.

        Note: pool and get_query parameters maintained for compatibility
        but not used since AsyncPostgresSaver manages its own connections.
        """
        self.logger = llmmllogger.bind(component="checkpoint_storage_instance")
        self._connection_string: Optional[str] = None
        self._initialized = False

    async def initialize(self, connection_string: str) -> None:
        """
        Initialize checkpoint storage by setting up tables.

        Args:
            connection_string: PostgreSQL connection string
        """
        try:
            self._connection_string = connection_string

            # Setup tables using LangGraph's standard approach
            # This creates the checkpoints and checkpoint_writes tables automatically
            async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
                await saver.setup()

            self._initialized = True
            self.logger.info(
                "✅ Checkpoint storage initialized using LangGraph's standard tables"
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize checkpoint storage: {e}")
            raise

    @asynccontextmanager
    async def create_checkpointer(self):
        """
        Create a new AsyncPostgresSaver instance for workflow compilation.

        This follows LangGraph's standard pattern for production usage.
        Use this method when compiling graphs that need persistence.

        Usage:
            async with checkpoint_storage.create_checkpointer() as checkpointer:
                graph = builder.compile(checkpointer=checkpointer)
        """
        if not self._initialized or not self._connection_string:
            raise RuntimeError(
                "CheckpointStorage not initialized - call initialize() first"
            )

        async with AsyncPostgresSaver.from_conn_string(
            self._connection_string
        ) as saver:
            yield saver

    def create_saver_for_workflow(self):
        """
        Create an AsyncPostgresSaver for workflow compilation.

        Returns the context manager which can be used with async context.
        This follows LangGraph's recommended production pattern.

        Returns:
            AsyncPostgresSaver context manager
        """
        if not self._initialized or not self._connection_string:
            raise RuntimeError(
                "CheckpointStorage not initialized - call initialize() first"
            )

        # Return the context manager - this is the standard LangGraph pattern
        return AsyncPostgresSaver.from_conn_string(self._connection_string)

    def get_connection_string(self) -> Optional[str]:
        """Get the connection string for external checkpointer creation."""
        return self._connection_string if self._initialized else None

    def is_initialized(self) -> bool:
        """Check if checkpoint storage has been initialized."""
        return self._initialized
