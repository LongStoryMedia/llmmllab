"""Integration tests for Alembic migrations — exercises real PostgreSQL.

Verifies that:
1. The initial migration (0001) creates all expected tables.
2. Incremental migrations apply correctly on top of existing schema.
"""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Generator

import pytest
from sqlalchemy import text


# ─── Phase 1: verify initial schema ─────────────────────────────────────────

EXPECTED_TABLES = {
    "users",
    "api_keys",
    "conversations",
    "messages",
    "message_contents",
    "documents",
    "summaries",
    "search_topic_syntheses",
    "memories",
    "images",
    "research_tasks",
    "research_subtasks",
    "thoughts",
    "tool_calls",
    "todos",
}


@pytest.mark.asyncio
async def test_alembic_initial_migration_creates_tables(session_factory):
    """The session-scoped _alembic_migrations fixture runs `alembic upgrade head`.

    This test verifies all expected tables exist and that alembic_version
    is stamped at the head revision.
    """
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
        )
        tables = {row[0] for row in result}

        missing = EXPECTED_TABLES - tables
        assert not missing, f"Missing tables after migration: {missing}"

        # Check alembic_version is stamped
        version = await session.execute(
            text("SELECT version_num FROM alembic_version")
        )
        rev = version.scalar()
        assert rev is not None, "alembic_version table is empty"
        assert rev == "0001", f"Expected head revision '0001', got '{rev}'"


# ─── Phase 2: verify incremental migration applies ──────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# Migration source for the incremental 0002 revision
_INCREMENTAL_MIGRATION_SRC = dedent('''\
    """add tags column to todos table

    Revision ID: 0002
    Revises: 0001
    Create Date: 2026-01-01 00:00:00.000000
    """

    from alembic import op
    import sqlalchemy as sa

    revision = "0002"
    down_revision = "0001"
    branch_labels = None
    depends_on = None

    def upgrade() -> None:
        op.add_column("todos", sa.Column("tags", sa.JSON(), nullable=True))

    def downgrade() -> None:
        op.drop_column("todos", "tags")
''')


@pytest.fixture
def temp_migration_dir() -> Generator[Path, None, None]:
    """Create a temporary Alembic config with an incremental migration (0002).

    The migration adds a `tags` JSON column to the `todos` table.
    """
    root = _project_root()
    src_versions = root / "alembic" / "versions"

    tmp = Path(tempfile.mkdtemp(prefix="alembic_test_"))

    # Copy versions/ directory
    dst_versions = tmp / "versions"
    shutil.copytree(src_versions, dst_versions)

    # Write incremental migration file
    migration_file = dst_versions / "0002_add_todo_tags.py"
    migration_file.write_text(_INCREMENTAL_MIGRATION_SRC)

    # Write a minimal env.py that mirrors the real one's logic
    env_py = tmp / "env.py"
    env_py.write_text(dedent(f'''\
        import os
        import sys
        sys.path.insert(0, {str(root)!r})

        from alembic import context
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool
        from db.models import Base

        config = context.config

        connection_string = os.environ.get("DB_CONNECTION_STRING", "")
        if connection_string:
            connection_string = connection_string.replace(
                "postgresql+asyncpg://", "postgresql+psycopg2://", 1
            )
            connection_string = connection_string.replace(
                "postgres+asyncpg://", "postgresql+psycopg2://", 1
            )
            if connection_string.startswith("postgresql://"):
                connection_string = connection_string.replace(
                    "postgresql://", "postgresql+psycopg2://", 1
                )
            elif connection_string.startswith("postgres://"):
                connection_string = connection_string.replace(
                    "postgres://", "postgresql+psycopg2://", 1
                )

        config.set_main_option("sqlalchemy.url", connection_string)
        target_metadata = Base.metadata

        def run_migrations_online() -> None:
            connectable = create_engine(
                config.get_main_option("sqlalchemy.url"),
                poolclass=NullPool,
                echo=False,
                isolation_level="AUTOCOMMIT",
            )
            with connectable.connect() as connection:
                context.configure(
                    connection=connection,
                    target_metadata=target_metadata,
                )
                with context.begin_transaction():
                    context.run_migrations()

        if context.is_offline_mode():
            pass
        else:
            run_migrations_online()
    '''))

    # Write a minimal alembic.ini pointing at the temp script_location
    alembic_ini = tmp / "alembic.ini"
    alembic_ini.write_text(
        dedent(f"""\
            [alembic]
            script_location = {tmp}
            sqlalchemy.url = postgresql+psycopg2://localhost:5432/db
        """)
    )

    yield tmp

    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_alembic_incremental_migration_applies(
    session_factory,
    sync_connection_string: str,
    temp_migration_dir: Path,
):
    """Apply a new migration (0002) on top of the existing schema and verify
    the column is added."""

    # Verify the column does NOT exist yet (proves migration hasn't run)
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'todos'
                  AND column_name = 'tags'
            """)
        )
        assert result.scalar() is None, (
            "tags column exists before migration — test setup issue"
        )

    # Run Alembic upgrade using the temp migration directory
    proc = await asyncio.wait_for(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(temp_migration_dir / "alembic.ini"),
            "upgrade",
            "head",
            cwd=str(temp_migration_dir),
            env={**os.environ, "DB_CONNECTION_STRING": sync_connection_string},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        ),
        timeout=30,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        out = stdout.decode(errors="replace").strip()
        raise RuntimeError(
            f"Alembic incremental migration failed (exit {proc.returncode}):\n"
            f"STDOUT: {out}\nSTDERR: {err}"
        )

    # Verify the `tags` column now exists on `todos`
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'todos'
                  AND column_name = 'tags'
            """)
        )
        row = result.mappings().one_or_none()
        assert row is not None, "tags column not found after migration"
        assert row.data_type in ("json", "jsonb"), (
            f"Expected json/jsonb, got {row.data_type}"
        )

        # Verify alembic_version was advanced
        version = await session.execute(
            text("SELECT version_num FROM alembic_version")
        )
        rev = version.scalar()
        assert rev == "0002", (
            f"Expected revision '0002' after incremental migration, got '{rev}'"
        )

        # Smoke test: insert a row with the new column
        await session.execute(
            text("""
                INSERT INTO todos (user_id, title, status, priority, tags)
                VALUES (:uid, 'migration test', 'not-started', 'low', '["auto", "test"]')
                RETURNING id
            """),
            {"uid": "migration-test-user"},
        )
        await session.commit()

    # Clean up: delete the test row
    async with session_factory() as session:
        await session.execute(
            text("DELETE FROM todos WHERE user_id = 'migration-test-user'")
        )
        await session.commit()
