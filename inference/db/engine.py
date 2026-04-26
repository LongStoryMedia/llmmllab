"""
SQLAlchemy async engine and session factory.

Replaces the hand-rolled asyncpg.Pool with SQLAlchemy's AsyncEngine
and async_sessionmaker for first-class async ORM support.
"""

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="db_engine")

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def create_async_engine(connection_string: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine using asyncpg as the DBAPI."""
    from sqlalchemy.ext.asyncio import create_async_engine

    # Ensure the connection string uses the asyncpg driver
    if connection_string.startswith("postgresql://") and not connection_string.startswith("postgresql+asyncpg://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif connection_string.startswith("postgres://") and not connection_string.startswith("postgres+asyncpg://"):
        connection_string = connection_string.replace("postgres://", "postgres+asyncpg://", 1)

    # asyncpg doesn't accept query string parameters as kwargs (e.g., ?sslmode=disable).
    # Strip the query string and handle sslmode via connect_args instead.
    import urllib.parse

    parsed = urllib.parse.urlparse(connection_string)
    query_params = urllib.parse.parse_qs(parsed.query)

    connect_args = {}
    if "sslmode" in query_params:
        sslmode = query_params["sslmode"][0]
        if sslmode == "disable":
            connect_args["ssl"] = False
        elif sslmode != "allow":
            import ssl as ssl_module

            connect_args["ssl"] = ssl_module.create_default_context()

    # Rebuild URL without query string (asyncpg rejects ?sslmode as kwarg)
    connection_string = parsed._replace(query="").geturl()

    engine = create_async_engine(
        connection_string,
        echo=False,
        connect_args=connect_args if connect_args else {},
        # Match previous asyncpg pool behavior — pre_ping detects stale connections
        # (replaces our hand-rolled connection_recovery module)
        pool_pre_ping=True,
    )

    global _engine
    _engine = engine
    logger.info("SQLAlchemy async engine created")
    return engine


def get_engine() -> AsyncEngine:
    """Get the global async engine instance."""
    if _engine is None:
        raise RuntimeError(
            "Database engine not initialized. Call create_async_engine() first."
        )
    return _engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine."""
    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    global _session_factory
    _session_factory = factory
    logger.info("SQLAlchemy session factory created")
    return factory


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the global session factory instance."""
    if _session_factory is None:
        raise RuntimeError(
            "Session factory not initialized. Call create_session_factory() first."
        )
    return _session_factory


async def dispose_engine() -> None:
    """Dispose the engine and its connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("SQLAlchemy engine disposed")
