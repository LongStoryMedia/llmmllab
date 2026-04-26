"""Alembic environment configuration.

Uses a synchronous SQLAlchemy engine with psycopg2 for migrations.
The db/__init__.py caller ensures the URL uses the psycopg2 driver.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the sqlalchemy.url from environment variable (sync driver)
connection_string = os.environ.get("DB_CONNECTION_STRING", "")
if connection_string:
    # Strip async prefix, use psycopg2 sync driver
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

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using a sync engine.

    TimescaleDB DDL (create_hypertable, add_compression_policy, etc.) requires
    autocommit — it cannot run inside a user transaction. We set autocommit on
    the connection so each statement commits immediately, and skip the outer
    transaction wrapper.
    """
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=NullPool,
        echo=True,
    )

    with connectable.connect().execution_options(autocommit=True) as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
