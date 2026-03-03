"""
Database Connection Recovery - handles stale OID errors and connection state issues.
"""

import asyncio
import asyncpg
from typing import Optional, Any
from server.utils.logging import llmmllogger

logger = llmmllogger.bind(component="db_connection_recovery")


class ConnectionRecoveryManager:
    """Manages database connection recovery from stale OID errors."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def is_stale_oid_error(self, error: Exception) -> bool:
        """Check if the error is caused by a stale OID reference or related transaction issues."""
        error_message = str(error).lower()
        return (
            "could not open relation with oid" in error_message
            or (
                "relation with oid" in error_message
                and "does not exist" in error_message
            )
            or (
                "current transaction is aborted" in error_message
                and "recovery" in str(error_message)
            )
        )

    async def recover_from_stale_oid(self, error: Exception) -> bool:
        """
        Attempt to recover from stale OID errors by refreshing connection pool state.
        For transaction abort errors, we skip connection flushing to avoid interference.

        Returns:
            bool: True if recovery was attempted, False otherwise
        """
        if not await self.is_stale_oid_error(error):
            return False

        logger.warning(f"Detected stale OID or transaction error: {error}")

        # If this is a transaction abort error, we don't flush connections
        # as it would interfere with transaction handling
        if "current transaction is aborted" in str(error).lower():
            logger.info(
                "Transaction abort detected - skipping connection flush, transaction will be retried"
            )
            return True

        logger.info("Attempting connection pool recovery...")

        try:
            # Clear all cached prepared statements and schema state
            await self._flush_pool_connections()

            # Test connection after recovery
            await self._test_connection()

            logger.info("✅ Connection pool recovery completed successfully")
            return True

        except Exception as recovery_error:
            logger.error(f"❌ Connection pool recovery failed: {recovery_error}")
            return False

    async def _flush_pool_connections(self) -> None:
        """
        Flush all connections in the pool to clear cached prepared statements and schema state.
        """
        connections = []

        try:
            # Acquire all available connections
            max_connections = getattr(self.pool, "_maxsize", 10)

            for i in range(max_connections):
                try:
                    conn = await asyncio.wait_for(self.pool.acquire(), timeout=1.0)
                    connections.append(conn)
                except (asyncio.TimeoutError, Exception):
                    # No more connections available or timeout
                    break

            logger.info(f"Acquired {len(connections)} connections for cache flush")

            # Flush each connection's cached state
            for i, conn in enumerate(connections):
                try:
                    # Clear prepared statements cache
                    await conn.execute("DISCARD ALL;")

                    # Reload schema state to clear cached OID mappings
                    await conn.reload_schema_state()

                    logger.debug(f"Flushed connection {i+1}/{len(connections)}")

                except Exception as e:
                    logger.warning(f"Failed to flush connection {i+1}: {e}")

        finally:
            # Release all connections back to the pool
            for conn in connections:
                try:
                    await self.pool.release(conn)
                except Exception as e:
                    logger.warning(f"Failed to release connection: {e}")

    async def _test_connection(self) -> None:
        """Test that the pool can successfully execute a simple query."""
        async with self.pool.acquire() as conn:
            await conn.fetchrow("SELECT 1 as test")

    async def execute_with_recovery(
        self, operation: callable, *args, max_retries: int = 2, **kwargs
    ) -> Any:
        """
        Execute a database operation with automatic recovery from stale OID errors.

        Args:
            operation: Async function to execute
            *args: Arguments for the operation
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation

        Raises:
            The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if this is a stale OID error that we can recover from
                if await self.is_stale_oid_error(e):
                    if attempt < max_retries:
                        logger.warning(
                            f"Stale OID error on attempt {attempt + 1}/{max_retries + 1}: {e}"
                        )

                        # Attempt recovery
                        recovery_success = await self.recover_from_stale_oid(e)

                        if recovery_success:
                            logger.info(
                                f"Retrying operation after recovery (attempt {attempt + 2})"
                            )
                            continue
                        else:
                            logger.error("Recovery failed, will not retry")
                            break
                    else:
                        logger.error(
                            f"Max retries ({max_retries}) reached for stale OID error"
                        )
                        break
                else:
                    # Not a stale OID error, don't retry
                    break

        # If we get here, all attempts failed
        raise last_exception


# Create a global instance to be used by storage classes
recovery_manager: Optional[ConnectionRecoveryManager] = None


def init_recovery_manager(pool: asyncpg.Pool) -> None:
    """Initialize the global recovery manager with a connection pool."""
    global recovery_manager
    recovery_manager = ConnectionRecoveryManager(pool)
    logger.info("Database connection recovery manager initialized")


def get_recovery_manager() -> Optional[ConnectionRecoveryManager]:
    """Get the global recovery manager instance."""
    return recovery_manager


async def execute_with_recovery(operation: callable, *args, **kwargs) -> Any:
    """
    Convenience function to execute operations with recovery.

    Usage:
        result = await execute_with_recovery(
            lambda: conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        )
    """
    if recovery_manager is None:
        # No recovery manager available, execute directly
        return await operation(*args, **kwargs)

    return await recovery_manager.execute_with_recovery(operation, *args, **kwargs)
