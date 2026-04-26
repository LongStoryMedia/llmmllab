"""
Simplified checkpoint storage service using LangGraph AsyncPostgresSaver.
Provides clean factory methods for creating checkpointers without unnecessary abstraction.
"""

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

    def __init__(self, connection_string: str):
        """
        Initialize checkpoint storage and set up tables.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.logger = llmmllogger.bind(component="checkpoint_storage_instance")
        self._connection_string = connection_string
        self._initialized = False
        self._setup()

    def _setup(self) -> None:
        """Set up checkpoint tables using LangGraph's standard approach."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're inside an async context; schedule setup
            asyncio.ensure_future(self._async_setup())
        else:
            # No event loop; create one and run setup synchronously
            asyncio.run(self._async_setup())

    async def _async_setup(self) -> None:
        """Async helper to set up the checkpoint tables."""
        try:
            async with AsyncPostgresSaver.from_conn_string(
                self._connection_string
            ) as saver:
                await saver.setup()

            self._initialized = True
            self.logger.info(
                "Checkpoint storage initialized using LangGraph's standard tables"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize checkpoint storage: {e}")
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
        if not self._initialized:
            raise RuntimeError(
                "CheckpointStorage not initialized - initialization failed"
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
        if not self._initialized:
            raise RuntimeError(
                "CheckpointStorage not initialized - initialization failed"
            )

        # Return the context manager - this is the standard LangGraph pattern
        return AsyncPostgresSaver.from_conn_string(self._connection_string)

    def get_connection_string(self):
        """Get the connection string for external checkpointer creation."""
        return self._connection_string if self._initialized else None

    def is_initialized(self) -> bool:
        """Check if checkpoint storage has been initialized."""
        return self._initialized