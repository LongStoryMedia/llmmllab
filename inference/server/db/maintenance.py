"""
Database maintenance utilities for periodic optimization tasks.
"""

import asyncio
import logging
import asyncpg
from typing import Optional
import datetime
import contextlib
from server.db.queries import get_query

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

                # 2. Reindex tables to optimize indexes
                logger.info("Running REINDEX on database...")
                try:
                    # Note: We use the current database name rather than hardcoding "ollama"
                    # Get current database name
                    db_name_row = await conn.fetchrow(
                        "SELECT current_database() as db_name"
                    )
                    db_name = db_name_row["db_name"]

                    # Run reindex
                    await conn.execute(
                        f"REINDEX (VERBOSE, CONCURRENTLY) DATABASE {db_name}"
                    )
                    logger.info(
                        f"REINDEX completed successfully on database '{db_name}'"
                    )
                except Exception as e:
                    logger.error(f"Failed to run REINDEX: {str(e)}")
                    success = False

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
                # Run maintenance immediately at startup
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


# Create singleton instance
maintenance_service = DatabaseMaintenanceService()
