"""
Database maintenance utilities for periodic optimization tasks.
"""

import asyncio
import logging
import asyncpg
from typing import Optional
import datetime
import contextlib
import os

logger = logging.getLogger(__name__)


class DatabaseMaintenanceService:
    """Service to perform periodic database maintenance tasks"""

    def __init__(self):
        self.pool = None
        self._maintenance_task = None
        self._interval_hours = 24  # Default to running once per day
        self._is_running = False
        self._last_run = None

    async def initialize(self, pool: asyncpg.Pool, interval_hours: int = 24):
        """Initialize the maintenance service with a connection pool and interval"""
        self.pool = pool
        self._interval_hours = interval_hours
        logger.info(
            f"Database maintenance service initialized with {interval_hours} hour interval"
        )

    async def perform_maintenance(self) -> bool:
        """
        Perform database maintenance tasks like VACUUM ANALYZE, REINDEX, and policy refresh.
        Similar to the Go implementation's PerformDatabaseMaintenance function.

        Returns:
            bool: True if maintenance completed successfully, False otherwise
        """
        if not self.pool:
            logger.error("Cannot perform maintenance: database pool not initialized")
            return False

        logger.info("Starting database maintenance tasks...")
        success = True

        try:
            # Get a connection from the pool
            async with self.pool.acquire() as conn:
                # 1. Vacuum analyze for better query planning
                logger.info("Running VACUUM ANALYZE...")
                try:
                    await conn.execute("VACUUM ANALYZE")
                    logger.info("VACUUM ANALYZE completed successfully")
                except Exception as e:
                    logger.error(f"Failed to run VACUUM ANALYZE: {str(e)}")
                    success = False

                # 1b. Align sequences to prevent ID drift causing duplicates on restore/migration
                logger.info(
                    "Aligning sequences for hypertables (messages, message_contents)..."
                )
                try:
                    # Never move sequence backwards: use GREATEST with current last_value
                    await conn.execute(
                        """
                        SELECT setval(
                            'messages_id_seq',
                            GREATEST(
                                COALESCE((SELECT MAX(id) FROM messages), 0),
                                (SELECT last_value FROM messages_id_seq)
                            ),
                            true
                        );
                        """
                    )
                    await conn.execute(
                        """
                        SELECT setval(
                            'message_contents_id_seq',
                            GREATEST(
                                COALESCE((SELECT MAX(id) FROM message_contents), 0),
                                (SELECT last_value FROM message_contents_id_seq)
                            ),
                            true
                        );
                        """
                    )
                    logger.info("Sequence alignment completed successfully")
                except Exception as e:
                    logger.error(f"Failed to align sequences: {str(e)}")
                    success = False

                # 2. Optional REINDEX (off by default to avoid stale OID plan errors during traffic)
                reindex_enabled = os.environ.get(
                    "DB_REINDEX_ON_MAINTENANCE", "false"
                ).lower() in ("1", "true", "yes")
                if reindex_enabled:
                    logger.info(
                        "Running REINDEX on database (DB_REINDEX_ON_MAINTENANCE=true)..."
                    )
                    try:
                        # Note: We use the current database name rather than hardcoding
                        db_name_row = await conn.fetchrow(
                            "SELECT current_database() as db_name"
                        )
                        db_name = db_name_row["db_name"]

                        # Run reindex concurrently
                        await conn.execute(
                            f"REINDEX (VERBOSE, CONCURRENTLY) DATABASE {db_name}"
                        )
                        logger.info(
                            f"REINDEX completed successfully on database '{db_name}'"
                        )
                    except Exception as e:
                        logger.error(f"Failed to run REINDEX: {str(e)}")
                        success = False

                    # Flush statement caches across pool connections to avoid stale OID references
                    try:
                        await self._flush_pool_caches()
                        logger.info("Flushed statement caches across pool connections")
                    except Exception as e:
                        logger.warning(
                            f"Failed to flush some connection caches (will recover on reconnect): {str(e)}"
                        )
                else:
                    logger.info(
                        "Skipping REINDEX (DB_REINDEX_ON_MAINTENANCE not enabled)"
                    )

                # 3. Run TimescaleDB-specific maintenance
                logger.info("Running TimescaleDB policy refresh...")
                try:
                    result = await conn.fetch(
                        "SELECT run_job(j.id) FROM timescaledb_information.jobs j WHERE j.proc_name = 'policy_refresh'"
                    )
                    if result:
                        logger.info(
                            f"TimescaleDB policy refresh completed successfully: {len(result)} jobs processed"
                        )
                    else:
                        logger.info(
                            "TimescaleDB policy refresh completed (no jobs found)"
                        )
                except Exception as e:
                    logger.warning(
                        f"Note: TimescaleDB policy refresh failed (may be normal if no jobs): {str(e)}"
                    )
                    # This is not considered a failure as it's expected in some cases

            self._last_run = datetime.datetime.now()
            logger.info(
                "Database maintenance tasks completed successfully"
                if success
                else "Database maintenance completed with some errors"
            )
            return success

        except Exception as e:
            logger.error(f"Unexpected error during database maintenance: {str(e)}")
            return False

    async def start_maintenance_schedule(self):
        """Start the scheduled maintenance task"""
        if self._maintenance_task is not None:
            logger.warning("Maintenance schedule is already running")
            return

        self._is_running = True
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info(
            f"Database maintenance schedule started with {self._interval_hours} hour interval"
        )

    async def stop_maintenance_schedule(self):
        """Stop the scheduled maintenance task"""
        if self._maintenance_task is None:
            logger.warning("No maintenance schedule is running")
            return

        self._is_running = False
        self._maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._maintenance_task
        self._maintenance_task = None
        logger.info("Database maintenance schedule stopped")

    async def _maintenance_loop(self):
        """Internal loop that runs maintenance at the specified interval"""
        try:
            while self._is_running:
                # Optional initial delay to avoid racing with app traffic on startup
                try:
                    initial_delay = int(
                        os.environ.get("DB_MAINTENANCE_INITIAL_DELAY_SECONDS", "300")
                    )
                except ValueError:
                    initial_delay = 300
                if initial_delay > 0 and self._last_run is None:
                    await asyncio.sleep(initial_delay)

                await self.perform_maintenance()

                # Wait for the specified interval before running again
                await asyncio.sleep(
                    self._interval_hours * 3600
                )  # Convert hours to seconds
        except asyncio.CancelledError:
            logger.info("Maintenance loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in maintenance loop: {str(e)}")
            # Try to restart the loop if an unexpected error occurs
            if self._is_running:
                asyncio.create_task(self._maintenance_loop())

    @property
    def last_run(self) -> Optional[datetime.datetime]:
        """Get the timestamp of the last maintenance run"""
        return self._last_run

    @property
    def next_run(self) -> Optional[datetime.datetime]:
        """Get the estimated timestamp of the next scheduled maintenance run"""
        if self._last_run is None or not self._is_running:
            return None
        return self._last_run + datetime.timedelta(hours=self._interval_hours)

    @property
    def status(self) -> dict:
        """Get the current status of the maintenance service"""
        return {
            "is_running": self._is_running,
            "interval_hours": self._interval_hours,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
        }

    async def align_sequences(self) -> bool:
        """Align sequences to the current MAX(id) without moving backwards.

        Returns True if alignment commands executed without error.
        """
        if not self.pool:
            logger.error("Cannot align sequences: database pool not initialized")
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    SELECT setval(
                        'messages_id_seq',
                        GREATEST(
                            COALESCE((SELECT MAX(id) FROM messages), 0),
                            (SELECT last_value FROM messages_id_seq)
                        ),
                        true
                    );
                    """
                )
                await conn.execute(
                    """
                    SELECT setval(
                        'message_contents_id_seq',
                        GREATEST(
                            COALESCE((SELECT MAX(id) FROM message_contents), 0),
                            (SELECT last_value FROM message_contents_id_seq)
                        ),
                        true
                    );
                    """
                )
            logger.info("Sequence alignment executed successfully")
            return True
        except Exception as e:
            logger.error(f"Error aligning sequences: {str(e)}")
            return False

    async def _flush_pool_caches(self) -> None:
        """Cycle through pool connections and clear prepared statements/schema cache.

        This prevents errors like 'could not open relation with OID ...' after REINDEX or DDL.
        """
        if not self.pool:
            return

        conns = []
        try:
            # Acquire as many connections as the pool allows
            max_to_acquire = getattr(self.pool, "_maxsize", 10)
            for _ in range(max_to_acquire):
                try:
                    c = await self.pool.acquire(timeout=0.5)
                except Exception:
                    break
                if not c:
                    break
                conns.append(c)

            # Flush each connection's state
            for c in conns:
                try:
                    await c.execute("DISCARD ALL;")
                except Exception:
                    # Some connections may not have any statements; ignore
                    pass
                try:
                    await c.reload_schema_state()
                except Exception:
                    pass
        finally:
            # Release back to pool
            for c in conns:
                with contextlib.suppress(Exception):
                    await self.pool.release(c)


# Create singleton instance
maintenance_service = DatabaseMaintenanceService()
