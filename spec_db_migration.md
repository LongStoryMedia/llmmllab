# Spec: asyncpg → SQLAlchemy 2.x + Alembic Migration

## Goal

Replace the hand-rolled asyncpg data layer with SQLAlchemy 2.x async ORM and Alembic for schema management, reducing boilerplate, improving type safety, and enabling idiomatic migration tooling.

## Current Architecture

### Database Access

- **Driver**: `asyncpg` (raw PostgreSQL driver)
- **Pool**: A single `asyncpg.Pool` created at startup, stored on a global `Storage` singleton (`db/__init__.py`)
- **Typed wrappers**: `TypedPool`/`TypedConnection` in `db_utils.py` — thin wrappers around asyncpg for better type hints
- **Connection recovery**: `connection_recovery.py` — detects stale OID errors, flushes pool with `DISCARD ALL`

### Schema Management

- **File**: `db/init_db.py` — `initialize_database(pool)` acquires a connection and runs ~14 initialization steps in dependency order
- **SQL files**: ~130 `.sql` files in `db/sql/` organized by entity (user, conversation, message, memory, etc.)
- **Loader**: `db/queries.py` — `SQLLoader` walks `db/sql/` at startup, builds a `Dict[str, str]` mapping query keys to SQL strings
- **Idempotency**: Achieved via `CREATE IF NOT EXISTS` and silently ignoring "already exists" errors

### Storage Modules (14 total)

Each module is a class constructed with `(pool: asyncpg.Pool, get_query: Callable[[str], str])` and optionally dependencies on other storage modules.

| Module | Table(s) | Complexity | Notes |
|--------|----------|------------|-------|
| `userconfig_storage.py` | `users` | Low | Multi-tier cache (memory → Redis → DB) |
| `conversation_storage.py` | `conversations` (hypertable) | Medium | Redis caching, recovery manager |
| `message_storage.py` | `messages` (hypertable) | **High** | Orchestrates inserts across contents/tool_calls/thoughts/documents in transactions |
| `message_content_storage.py` | `message_contents` (hypertable) | Low | Supports transactional connections |
| `summary_storage.py` | `summaries` (hypertable) | Medium | Redis caching |
| `memory_storage.py` | `memories` (hypertable, pgvector) | Medium | Vector similarity search, dimension normalization |
| `image_storage.py` | `images` | Low | Simple CRUD |
| `model_storage.py` | `models` | Low | Simple CRUD, JSON details |
| `search_storage.py` | `search_topic_syntheses` (hypertable) | Low | Simple CRUD |
| `thought_storage.py` | `thoughts` | Low | Supports transactional connections |
| `tool_call_storage.py` | `tool_calls` | Medium | JSON serialization safety |
| `document_storage.py` | `documents` (hypertable) | Low | Simple CRUD |
| `todo_storage.py` | `todos` | Low | Simple CRUD with filters |
| `api_key_storage.py` | `api_keys` | Medium | SHA-256 hashing, supports transactional connections |
| `checkpoint_storage.py` | `checkpoints`/`checkpoint_writes` | Special | Wraps LangGraph's `AsyncPostgresSaver` — not a direct storage module |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `cache_storage.py` | Redis-based caching singleton (Messages, Summaries, Conversations, UserConfigs) |
| `serialization.py` | JSON serialization utilities for complex objects |
| `maintenance.py` | Periodic VACUUM ANALYZE, sequence alignment, TimescaleDB policy refresh |
| `interfaces.py` | 6 ABCs (`MessageStore`, `ConversationStore`, etc.) — **not used as base classes** |

### Consumption Surface

- **11 server files** import `from db import storage` (routers: api_key, chat, users, todos, conversation, config, documents; middleware: db_init, auth; app.py)
- **8 composer files** import `from db import storage` or individual storage classes
- All access goes through the `storage` singleton: `storage.conversation.create_conversation(...)`, `storage.message.add_message(...)`, etc.

### Database Features

- **TimescaleDB**: Hypertables on conversations, messages, message_contents, summaries, memories, documents, search_topic_syntheses. Compression and retention policies on most hypertables.
- **pgvector**: 768-dimensional vector embeddings in `memories` table with cosine similarity search (`<->` operator)
- **Cascade deletes**: Triggers propagate deletes from user → conversations → messages → contents/thoughts/tool_calls/documents
- **Extensions**: `timescaledb`, `vector` (pgvector)

## Target Architecture

### SQLAlchemy 2.x Async ORM

- **Engine**: `sqlalchemy.ext.asyncio.create_async_engine()` with `asyncpg` as the DBAPI (same underlying driver, higher-level API)
- **Session**: `async_sessionmaker` bound to the async engine, providing request-scoped sessions
- **Models**: Declarative base classes for each table (~14 model classes)
- **Queries**: Mix of ORM queries (for simple CRUD) and `text()` raw SQL (for complex queries, TimescaleDB functions, pgvector operations)
- **pgvector**: Use the `pgvector` Python package's SQLAlchemy integration for `VECTOR(768)` columns and similarity operators

### Alembic for Schema Management

- **Configuration**: `alembic.ini` + `alembic/` directory in `inference/`
- **Initial migration**: Generate from current `db/sql/` DDL files, organized into versioned `V####_description.sql`-style Alembic migration scripts
- **Autogenerate**: Future schema changes can use `alembic revision --autogenerate` against SQLAlchemy models
- **Startup**: Alembic `upgrade head` runs during app lifespan startup, replacing `initialize_database(pool)`
- **TimescaleDB DDL**: Hypertable creation, compression, retention policies stay as raw SQL within Alembic migrations (they don't map to SQLAlchemy ORM)

### Session Lifecycle

- Replace the global `asyncpg.Pool` with a global `AsyncEngine` + `async_sessionmaker`
- Each request gets a session via a FastAPI dependency (`Depends(get_db_session)`)
- The `storage` singleton evolves to hold the `async_sessionmaker` and provides session-scoped storage operations

### What Stays the Same

- **Redis caching** (`cache_storage.py`) — unchanged, still used for read-through caching
- **Multi-tier caching** — unchanged
- **Serialization utilities** (`serialization.py`) — unchanged
- **Maintenance service** (`maintenance.py`) — adapted to use SQLAlchemy engine instead of asyncpg pool
- **LangGraph checkpoint storage** (`checkpoint_storage.py`) — `AsyncPostgresSaver` works with SQLAlchemy's connection string, minimal changes
- **Cascade delete triggers** — defined in Alembic migrations, behavior unchanged
- **The `storage` singleton pattern** — consumer code (`storage.conversation.create_conversation(...)`) stays the same at the call site

## Migration Scope

### In Scope

1. SQLAlchemy async engine + session factory replacing `asyncpg.Pool`
2. SQLAlchemy declarative models for all 14 tables
3. Alembic configuration + initial migration(s) from existing `db/sql/` DDL
4. Rewriting `db/__init__.py` — `Storage` class uses `async_sessionmaker` instead of `asyncpg.Pool`
5. Updating all 14 storage modules to use SQLAlchemy sessions
6. Updating `init_db.py` → Alembic startup call in `app.py` lifespan
7. Updating consumption points (server routers, composer) to use session injection
8. Updating test fixtures
9. Removing dead code: `queries.py` (SQLLoader), `db_utils.py` (TypedPool/TypedConnection), `connection_recovery.py`, unused `interfaces.py` ABCs

### Out of Scope

- Changing the database schema (table structures, column types)
- Changing Redis caching behavior
- Changing the public API surface of storage methods
- Modifying LangGraph's `AsyncPostgresSaver` internals
- Changing business logic in any storage method

## Key Design Decisions

### 1. ORM vs Raw SQL per Module (Decision: Pragmatic, No New Dependencies Beyond SQLAlchemy + Alembic)

Not all queries need to become ORM queries. The approach is pragmatic and dependency-minimal:

- **Simple CRUD** (api_keys, models, todos, images, documents) → full ORM
- **Complex queries with joins/subqueries** (messages, conversations) → `text()` with SQLAlchemy session — no extra packages
- **pgvector operations** (memory similarity search) → `text()` with raw SQL — skips the `pgvector` Python package, uses `ARRAY` type for embeddings and `<->` in raw SQL. Zero additional dependency.
- **TimescaleDB functions** (create_hypertable, add_compression_policy) → always raw SQL in Alembic migrations

**Dependencies added**: `sqlalchemy[asyncio]` and `alembic`. That's it. No `pgvector`, no `sqlalchemy-utils`, no `databases`. `asyncpg` stays as the DBAPI driver underneath SQLAlchemy.

### 2. Session Injection Strategy (Decision: Session Factory on Modules)

Storage modules hold the `async_sessionmaker` and create scoped sessions internally via `async with session_factory() as session:`. Simplest approach — existing call sites (`storage.conversation.create_conversation(...)`) don't need to change. Transaction boundaries are explicit inside methods that need them (message_storage's composite inserts).
