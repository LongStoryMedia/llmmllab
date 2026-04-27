"""
Database maintenance utilities for periodic optimization tasks.
"""

import asyncio
import datetime
import contextlib
import os

from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="db.maintenance")


class DatabaseMaintenanceService:
    """Service to perform periodic database maintenance tasks"""

    def __init__(self):
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self.engine: Optional[AsyncEngine] = None
        self._maintenance_task = None
        self._interval_hours = 24  # Default to running once per day
        self._is_running = False
        self._last_run = None

    async def initialize(
        self, engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession], interval_hours: int = 24
    ):
        """Initialize the maintenance service with a SQLAlchemy async engine and session factory"""
        self.engine = engine
        self.session_factory = session_factory
        self._interval_hours = interval_hours
        logger.info(
            f"Database maintenance service initialized with {interval_hours} hour interval"
        )

    async def perform_maintenance(self) -> bool:
        """
        Perform database maintenance tasks like VACUUM ANALYZE, REINDEX, and policy refresh.
        Similar to the Go implementation's PerformDatabaseMaintenance function.

        Each step runs in its own session/connection so a failure in one step
        does not cascade to the others.

        Returns:
            bool: True if maintenance completed successfully, False otherwise
        """
        if not self.session_factory or not self.engine:
            logger.error("Cannot perform maintenance: engine/session not initialized")
            return False

        logger.info("Starting database maintenance tasks...")
        success = True

        # 1. Vacuum analyze for better query planning
        # VACUUM cannot run inside a transaction block, so we use a raw
        # engine connection (no session) which runs in autocommit mode.
        await self._run_vacuum_analyze()

        # 1b. Align sequences to prevent ID drift causing duplicates on restore/migration
        if not await self._align_sequences():
            success = False

        # 2. Optional REINDEX (off by default to avoid stale OID plan errors during traffic)
        reindex_enabled = os.environ.get(
            "DB_REINDEX_ON_MAINTENANCE", "false"
        ).lower() in ("1", "true", "yes")
        if reindex_enabled:
            if not await self._run_reindex():
                success = False
        else:
            logger.info("Skipping REINDEX (DB_REINDEX_ON_MAINTENANCE not enabled)")

        # 3. Run TimescaleDB-specific maintenance
        await self._run_timescaledb_policy_refresh()

        self._last_run = datetime.datetime.now()
        logger.info(
            "Database maintenance tasks completed successfully"
            if success
            else "Database maintenance completed with some errors"
        )
        return success

    async def _run_vacuum_analyze(self) -> None:
        """Run VACUUM ANALYZE on a raw connection (no transaction)."""
        logger.info("Running VACUUM ANALYZE...")
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("VACUUM ANALYZE"))
            logger.info("VACUUM ANALYZE completed successfully")
        except Exception as e:
            logger.error(f"Failed to run VACUUM ANALYZE: {str(e)}")

    async def _align_sequences(self) -> bool:
        """Align sequences using a dedicated session."""
        logger.info(
            "Aligning sequences for hypertables (messages, message_contents)..."
        )
        try:
            async with self.session_factory() as session:
                await session.execute(
                    text("""
                        SELECT setval(
                            'messages_id_seq',
                            GREATEST(
                                COALESCE((SELECT MAX(id) FROM messages), 0),
                                (SELECT last_value FROM messages_id_seq)
                            ),
                            true
                        );
                    """)
                )
                await session.execute(
                    text("""
                        SELECT setval(
                            'message_contents_id_seq',
                            GREATEST(
                                COALESCE((SELECT MAX(id) FROM message_contents), 0),
                                (SELECT last_value FROM message_contents_id_seq)
                            ),
                            true
                        );
                    """)
                )
                await session.commit()
            logger.info("Sequence alignment completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to align sequences: {str(e)}")
            return False

    async def _run_reindex(self) -> bool:
        """Run REINDEX using a dedicated session."""
        logger.info(
            "Running REINDEX on database (DB_REINDEX_ON_MAINTENANCE=true)..."
        )
        try:
            async with self.session_factory() as session:
                db_name_row = await session.execute(
                    text("SELECT current_database() as db_name")
                )
                db_name = db_name_row.mappings()["db_name"]  # type: ignore[index]

                await session.execute(
                    text(f"REINDEX (VERBOSE, CONCURRENTLY) DATABASE {db_name}")
                )
                await session.commit()
            logger.info(f"REINDEX completed successfully on database '{db_name}'")

            # Flush statement caches across pool connections
            try:
                await self._flush_pool_caches()
                logger.info("Flushed statement caches across pool connections")
            except Exception as e:
                logger.warning(
                    f"Failed to flush some connection caches (will recover on reconnect): {str(e)}"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to run REINDEX: {str(e)}")
            return False

    async def _run_timescaledb_policy_refresh(self) -> None:
        """Run TimescaleDB policy refresh using a dedicated session."""
        logger.info("Running TimescaleDB policy refresh...")
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(
                        "SELECT run_job(j.id) FROM timescaledb_information.jobs j WHERE j.proc_name = 'policy_refresh'"
                    )
                )
                rows = list(result)
                if rows:
                    logger.info(
                        f"TimescaleDB policy refresh completed: {len(rows)} jobs processed"
                    )
                else:
                    logger.info("TimescaleDB policy refresh completed (no jobs found)")
        except Exception as e:
            logger.warning(
                f"TimescaleDB policy refresh failed (may be normal if no jobs): {str(e)}"
            )

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
        if not self.session_factory:
            logger.error("Cannot align sequences: session factory not initialized")
            return False

        try:
            async with self.session_factory() as conn:
                await conn.execute(
                    text("""
                    SELECT setval(
                        'messages_id_seq',
                        GREATEST(
                            COALESCE((SELECT MAX(id) FROM messages), 0),
                            (SELECT last_value FROM messages_id_seq)
                        ),
                        true
                    );
                    """)
                )
                await conn.execute(
                    text("""
                    SELECT setval(
                        'message_contents_id_seq',
                        GREATEST(
                            COALESCE((SELECT MAX(id) FROM message_contents), 0),
                            (SELECT last_value FROM message_contents_id_seq)
                        ),
                        true
                    );
                    """)
                )
                await conn.commit()
            logger.info("Sequence alignment executed successfully")
            return True
        except Exception as e:
            logger.error(f"Error aligning sequences: {str(e)}")
            return False

    async def _flush_pool_caches(self) -> None:
        """Cycle through session connections and clear prepared statements/schema cache.

        This prevents errors like 'could not open relation with OID ...' after REINDEX or DDL.
        """
        if not self.engine:
            return

        try:
            # Use DISCARD ALL on raw connections from the underlying pool
            async with self.engine.connect() as conn:
                await conn.execute(text("DISCARD ALL;"))
                await conn.commit()
        except Exception:
            # Some connections may not have any statements; ignore
            pass


# Create singleton instance
maintenance_service = DatabaseMaintenanceService()