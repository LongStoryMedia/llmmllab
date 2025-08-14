"""
Utility functions and classes for database operations with proper type hints.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import (
    AsyncGenerator,
    TypeVar,
    Generic,
    Callable,
    Any,
    Optional,
    Dict,
    Union,
    List,
    cast,
)

import asyncpg
from asyncpg.protocol import Record

T = TypeVar("T")


class TypedConnection:
    """
    A wrapper around asyncpg.Connection that provides proper type hints.
    """

    def __init__(self, connection: asyncpg.Connection):
        self._connection = connection

    async def fetchrow(self, query: str, *args, **kwargs) -> Optional[Record]:
        """Fetch a single row from the database."""
        result = await self._connection.fetchrow(query, *args, **kwargs)
        return cast(Optional[Record], result)

    async def fetch(self, query: str, *args, **kwargs) -> List[Record]:
        """Fetch multiple rows from the database."""
        result = await self._connection.fetch(query, *args, **kwargs)
        return cast(List[Record], result)

    async def execute(self, query: str, *args, **kwargs) -> str:
        """Execute a query and return a status string."""
        result = await self._connection.execute(query, *args, **kwargs)
        return cast(str, result)

    # Add any other methods you need from asyncpg.Connection


class TypedPool:
    """
    A wrapper around asyncpg.Pool that provides proper type hints for acquired connections.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[TypedConnection, None]:
        """
        Acquire a connection from the pool with proper type hints.

        Usage:

        ```python
        async with typed_pool.acquire() as conn:
            result = await conn.fetchrow("SELECT * FROM table")
        ```
        """
        conn = await self._pool.acquire()
        try:
            yield TypedConnection(conn)
        finally:
            await self._pool.release(conn)


def typed_pool(pool: asyncpg.Pool) -> TypedPool:
    """
    Create a typed wrapper around an asyncpg.Pool.

    Usage:

    ```python
    # When initializing your storage class:
    self.typed_pool = typed_pool(pool)

    # When using the pool:
    async with self.typed_pool.acquire() as conn:
        result = await conn.fetchrow("SELECT * FROM table")
    ```
    """
    return TypedPool(pool)
