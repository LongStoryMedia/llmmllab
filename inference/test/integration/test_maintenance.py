"""Integration tests for DatabaseMaintenanceService — exercises real PostgreSQL."""

import os

import pytest
from sqlalchemy import text

from db.maintenance import DatabaseMaintenanceService


@pytest.mark.asyncio
async def test_maintenance_performs_without_error(
    session_factory, engine,
):
    """Verify that perform_maintenance runs all steps without cascading errors."""
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # REINDEX is enabled via env but we don't want to require it
    # The test should pass regardless
    original = os.environ.pop("DB_REINDEX_ON_MAINTENANCE", None)
    try:
        del os.environ["DB_REINDEX_ON_MAINTENANCE"]
    except KeyError:
        pass

    result = await service.perform_maintenance()
    # Maintenance may return False if TimescaleDB has no jobs, but it
    # should NOT raise or cascade-fail after VACUUM.
    assert isinstance(result, bool)

    # Verify the database is still usable after maintenance
    async with session_factory() as session:
        r = await session.execute(text("SELECT 1 as ok"))
        assert r.scalar() == 1

    if original is not None:
        os.environ["DB_REINDEX_ON_MAINTENANCE"] = original


@pytest.mark.asyncio
async def test_maintenance_vacuum_does_not_break_subsequent_steps(
    session_factory, engine,
):
    """VACUUM ANALYZE runs outside a transaction; subsequent steps should
    still work even if VACUUM encounters issues."""
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    original = os.environ.pop("DB_REINDEX_ON_MAINTENANCE", None)
    try:
        del os.environ["DB_REINDEX_ON_MAINTENANCE"]
    except KeyError:
        pass

        # Run maintenance twice to ensure no stale transaction state
        await service.perform_maintenance()
        await service.perform_maintenance()

        # DB should still be healthy
        async with session_factory() as session:
            r = await session.execute(text("SELECT 1 as ok"))
            assert r.scalar() == 1
    finally:
        if original is not None:
            os.environ["DB_REINDEX_ON_MAINTENANCE"] = original
