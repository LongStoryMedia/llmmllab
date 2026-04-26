# Implementation Plan: asyncpg → SQLAlchemy 2.x + Alembic

## Dependencies Added

- `sqlalchemy[asyncio]>=2.0,<3.0` — ORM + async support
- `alembic>=1.14,<2.0` — migration tooling

`asyncpg` stays as the DBAPI driver underneath SQLAlchemy. No `pgvector` package — vector ops use raw SQL via `text()`.

---

## Phase 1 — Infrastructure: Engine, Session, Alembic Bootstrap

### Step 1.1: Add dependencies to `pyproject.toml`

**File**: `inference/pyproject.toml`

Add `sqlalchemy[asyncio]` and `alembic` to dependencies. `asyncpg` remains (now used as SQLAlchemy's DBAPI rather than directly).

### Step 1.2: Create SQLAlchemy engine + session factory module

**New file**: `inference/db/engine.py`

```
- create_async_engine(connection_string) → AsyncEngine
  - Uses asyncpg as driver: f"{connection_string}" (asyncpg is in extras_require)
  - Pool settings matching current behavior (no statement caching issues)
- async_sessionmaker bound to the engine
- get_session_factory() → async_sessionmaker[AsyncSession] (singleton accessor)
- dispose_engine() → close engine pool
```

### Step 1.3: Initialize Alembic

**New files**:
- `inference/alembic.ini` — standard config, sets `sqlalchemy.url` from env var
- `inference/alembic/env.py` — configures Alembic to use the async engine and read models from `db.models`
- `inference/alembic/versions/` — empty directory for migration scripts

`env.py` key configuration:
- Reads `DB_CONNECTION_STRING` from environment
- Uses `asyncpg` driver
- Sets `target_metadata` from SQLAlchemy model base so `autogenerate` works

### Step 1.4: Create SQLAlchemy declarative models

**New file**: `inference/db/models.py`

One declarative model per table. ~14 model classes. These map 1:1 to existing tables — no schema changes.

```python
class Users(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_primary_key
    username: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    config: Mapped[Optional[dict]] = mapped_column(JSON)

class Conversations(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_primary_key(autoincrement=True)
    user_id: Mapped[str]
    title: Mapped[str] = mapped_column(server_default=text("'New conversation'"))
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    # Composite PK (id, created_at) for TimescaleDB hypertable

# ... similar for: Messages, MessageContents, Summaries, Memories, Images,
#                 ToolCalls, Thoughts, Documents, Todos, ApiKeys,
#                 SearchTopicSyntheses, ResearchTasks, ResearchSubtasks
```

Key type mappings:
- `timestamptz` → `DateTime(timezone=True)`
- `jsonb` → `JSON`
- `vector(768)` → `String` (stored as text, queried via raw SQL) — avoids `pgvector` dependency
- `numeric(10,3)` → `Numeric(10, 3)`
- Composite PKs `(id, created_at)` on hypertables — represented but not enforced as FK targets

Relationships defined on models for ORM-accessible tables (users ↔ conversations ↔ messages) but **not required** — most queries will use `text()` or explicit joins.

### Step 1.5: Create initial Alembic migration from existing SQL

**New file**: `inference/alembic/versions/0001_initial_schema.py`

Rather than relying on `autogenerate` (which won't capture TimescaleDB hypertable/compression/retention DDL), create the initial migration manually by consolidating the existing `db/sql/` DDL files into a single `upgrade()` function.

Process:
1. Collect all DDL from `db/sql/` files in the order specified by `init_db.py`'s `initialization_steps`
2. Combine into one `upgrade()` function with `await op.execute(...)` calls
3. Extension creation (`CREATE EXTENSION`) goes first
4. Table creation second
5. TimescaleDB hypertable/compression/retention third
6. Triggers and indexes last

The `downgrade()` drops tables in reverse order.

This migration is a **1:1 translation** of the current schema — no changes.

---

## Phase 2 — Rewrite `db/__init__.py` (Storage Singleton)

### Step 2.1: Replace pool with session factory in Storage class

**File**: `inference/db/__init__.py`

Changes to `Storage`:
- `self.pool` → `self.engine: AsyncEngine` + `self.session_factory: async_sessionmaker`
- `initialize(connection_string)`:
  - Creates engine via `db.engine.create_async_engine()`
  - Runs Alembic `upgrade head` via `alembic command.upgrade_control` API
  - Creates `async_sessionmaker`
  - Passes `session_factory` (not `pool` + `get_query`) to all storage module constructors
- `close()`: disposes engine
- Remove `_clear_stale_connection_state()` — SQLAlchemy handles connection invalidation
- Remove `init_recovery_manager()` call — no longer needed

### Step 2.2: Remove dead modules

Delete:
- `inference/db/queries.py` — SQLLoader, `get_query()` — replaced by Alembic + inline `text()` queries
- `inference/db/db_utils.py` — `TypedPool`, `TypedConnection`, `typed_pool()` — SQLAlchemy sessions are typed natively
- `inference/db/connection_recovery.py` — stale OID recovery — SQLAlchemy's connection pool handles this
- `inference/db/init_db.py` — replaced by Alembic
- `inference/db/interfaces.py` — unused ABCs

---

## Phase 3 — Migrate Storage Modules

Strategy: Each module's constructor changes from `(pool, get_query)` to `(session_factory)`. Methods use `async with session_factory() as session:` internally. SQL queries move from `get_query("key")` to `text("""...""")` inline, or to ORM operations.

**Order**: Simple first, complex last. Each module is independently testable.

### Step 3.1: Model Storage (simplest — no dependencies, 4 methods)

**File**: `inference/db/model_storage.py`

- Constructor: `(session_factory)`
- `list_models()` → `select(Model)`
- `get_model(model_id)` → `select(Model).where(Model.id == model_id)`
- `create_model(model)` → `session.add(model); await session.commit()`
- `delete_model(model_id)` → `await session.delete(model)`
- Full ORM, no raw SQL needed

### Step 3.2: Image Storage (simple CRUD, 5 methods)

**File**: `inference/db/image_storage.py`

- Constructor: `(session_factory)`
- All methods → ORM (`select`, `session.add`, `session.delete`)

### Step 3.3: Todo Storage (simple CRUD with filters, 7 methods)

**File**: `inference/db/todo_storage.py`

- Constructor: `(session_factory)`
- All methods → ORM with `where()` clauses

### Step 3.4: Document Storage (simple CRUD, 4 methods)

**File**: `inference/db/document_storage.py`

- Constructor: `(session_factory)`
- All methods → ORM

### Step 3.5: Search Storage (simple CRUD, 2 methods)

**File**: `inference/db/search_storage.py`

- Constructor: `(session_factory)`
- All methods → ORM

### Step 3.6: API Key Storage (medium — SHA-256 hashing, 8 methods)

**File**: `inference/db/api_key_storage.py`

- Constructor: `(session_factory)`
- Hashing/generation static methods — unchanged
- CRUD methods → ORM
- Remove `ConnectionRecoveryManager` import

### Step 3.7: Thought Storage (low — 3 methods, supports transactional conn)

**File**: `inference/db/thought_storage.py`

- Constructor: `(session_factory)`
- Methods → ORM
- `conn` parameter on `add_thought` becomes internal: caller passes nothing, module creates session+transaction

### Step 3.8: Message Content Storage (low — 4 methods, transactional)

**File**: `inference/db/message_content_storage.py`

- Constructor: `(session_factory)`
- Methods → ORM
- Same transactional pattern as thought_storage

### Step 3.9: User Config Storage (low — with multi-tier cache)

**File**: `inference/db/userconfig_storage.py`

- Constructor: `(session_factory)`
- DB read/write → ORM via `text()` (the config queries are complex JSONB operations)
- Redis/memory caching layers — unchanged

### Step 3.10: Conversation Storage (medium — caching, recovery manager)

**File**: `inference/db/conversation_storage.py`

- Constructor: `(session_factory, user_config_storage)`
- `get_query("conversation.<name>")` → inline `text()` queries
- Remove `ConnectionRecoveryManager` and `typed_pool` imports
- Redis caching via `cache_storage` — unchanged

### Step 3.11: Summary Storage (medium — caching, 5 methods)

**File**: `inference/db/summary_storage.py`

- Constructor: `(session_factory)`
- CRUD → ORM with `select()` and `where()`
- Redis caching — unchanged

### Step 3.12: Memory Storage (medium — pgvector, 6 public methods)

**File**: `inference/db/memory_storage.py`

- Constructor: `(session_factory)`
- `store_memory()` → ORM insert (`session.add(Memory(...))`)
- `delete_memory()` / `delete_all_user_memories()` → ORM delete
- `search_similarity()` → `text()` with raw SQL for `<->` vector operator and `vector(768)` type. This is the one query that benefits from raw SQL — the cosine similarity expression is cleaner in PostgreSQL than through ORM.
- Embedding normalization/padding helpers — unchanged
- Remove `typed_pool` import

### Step 3.13: Tool Call Storage (medium — JSON serialization, 4 methods)

**File**: `inference/db/tool_call_storage.py`

- Constructor: `(session_factory)`
- Methods → ORM. JSON fields use SQLAlchemy `JSON` type which handles serialization.
- Remove `TypedConnection`, `typed_pool` imports

### Step 3.14: Message Storage (highest complexity — 8 public methods, 20+ private, orchestrates sub-storages)

**File**: `inference/db/message_storage.py`

- Constructor: `(session_factory, thought_storage, tool_call_storage, message_content_storage, document_storage)`
- Sub-storage calls (thought/tool_call/content/document) happen within the **same session** created by message_storage, ensuring a single transaction boundary
- `_add_message`: Creates one session, inserts message + delegates to sub-storages which use the same session via `session_factory()`. Use `session.begin()` for explicit transaction.
- `_get_message` + helpers → Mix of ORM `select()` for message row and `text()` for the composite queries that fetch related data
- Remove `TypedConnection`, `typed_pool`, `recovery_manager` imports
- The `conn: Optional[TypedConnection]` parameters on public methods are removed — sessions are managed internally

### Step 3.15: Checkpoint Storage (special — wraps LangGraph's AsyncPostgresSaver)

**File**: `inference/db/checkpoint_storage.py`

- Minimal changes. `AsyncPostgresSaver.from_conn_string()` works with the same connection string
- Constructor accepts `connection_string` instead of `pool`/`get_query`
- `initialize()` → unchanged (LangGraph creates its own tables)

---

## Phase 4 — Update Consumption Points

### Step 4.1: Update `server/app.py` lifespan

**File**: `inference/server/app.py`

- `await storage.initialize(DB_CONNECTION_STRING)` — same call, same signature
- Internally it now creates an SQLAlchemy engine and runs Alembic instead of creating an asyncpg pool and running `initialize_database()`
- Shutdown: `await storage.close()` disposes the engine

### Step 4.2: Update `server/middleware/db_init_middleware.py`

**File**: `inference/server/middleware/db_init_middleware.py`

- No changes to call pattern: `storage.initialized` flag + `await storage.initialize(connection_string)` stay the same

### Step 4.3: Update all server routers

**Files**: `server/routers/{api_key,chat,users,todos,conversation,config,documents}.py`

- No changes needed. All routers access `storage.<service>.<method>()` which has the same signature. The internal implementation changes from asyncpg to SQLAlchemy, but the public API is unchanged.

### Step 4.4: Update composer imports

**Files**: `composer/graph/workflows/base.py`, `composer/tools/static/memory_retrieval_tool.py`, `composer/graph/nodes/memory/{store,search}.py`

- `composer/graph/workflows/base.py` imports individual storage classes (`UserConfigStorage`, `ConversationStorage`, etc.) for type hints. These classes still exist with the same names, just with different constructor signatures. Update imports if needed.
- Files using `from db import storage` — no changes (same singleton, same attribute access pattern)

### Step 4.5: Update maintenance service

**File**: `inference/db/maintenance.py`

- Constructor/`initialize()` takes `AsyncEngine` instead of `asyncpg.Pool`
- `perform_maintenance()`: VACUUM ANALYZE, sequence alignment, and TimescaleDB policy refresh run via `await session.execute(text("VACUUM ANALYZE ..."))` using the session factory
- Remove `asyncpg`-specific imports

---

## Phase 5 — Tests & Cleanup

### Step 5.1: Update test fixtures

**Files**: Any test files that mock `asyncpg`, `db.db_utils.TypedPool`, or `db.connection_recovery`

- Current test surface is small (no dedicated storage tests on this branch)
- Tests that mock `asyncpg` → mock `sqlalchemy.ext.asyncio` instead
- Tests that use `db.connection_recovery` → remove (no longer needed)

### Step 5.2: Remove `db/sql/` directory

The `db/sql/` directory (~130 files) is no longer needed. Its DDL content lives in the Alembic initial migration. Its query content lives inline in storage modules as `text()` strings or ORM operations.

### Step 5.3: Verify and validate

1. `cd inference && python -c "from db.models import Base; print('Models OK')"` — models import cleanly
2. `cd inference && alembic current` — Alembic reads the migration state
3. `cd inference && pytest test/` — existing tests pass
4. `make validate` — Pyright type check passes

---

## File Manifest

### New Files (5)
| File | Purpose |
|------|---------|
| `inference/db/engine.py` | SQLAlchemy async engine + session factory |
| `inference/db/models.py` | Declarative ORM models (~14 classes) |
| `inference/alembic.ini` | Alembic configuration |
| `inference/alembic/env.py` | Alembic environment setup |
| `inference/alembic/versions/0001_initial_schema.py` | Initial migration from existing SQL |

### Modified Files (18)
| File | Change |
|------|--------|
| `inference/pyproject.toml` | Add sqlalchemy[asyncio], alembic |
| `inference/db/__init__.py` | Pool → engine + session_factory |
| `inference/db/model_storage.py` | asyncpg → ORM |
| `inference/db/image_storage.py` | asyncpg → ORM |
| `inference/db/todo_storage.py` | asyncpg → ORM |
| `inference/db/document_storage.py` | asyncpg → ORM |
| `inference/db/search_storage.py` | asyncpg → ORM |
| `inference/db/api_key_storage.py` | asyncpg → ORM |
| `inference/db/thought_storage.py` | asyncpg → ORM |
| `inference/db/message_content_storage.py` | asyncpg → ORM |
| `inference/db/userconfig_storage.py` | asyncpg → text() ORM |
| `inference/db/conversation_storage.py` | asyncpg → text() ORM |
| `inference/db/summary_storage.py` | asyncpg → ORM |
| `inference/db/memory_storage.py` | asyncpg → ORM + text() for vector |
| `inference/db/tool_call_storage.py` | asyncpg → ORM |
| `inference/db/message_storage.py` | asyncpg → ORM + text() |
| `inference/db/checkpoint_storage.py` | Minor constructor change |
| `inference/db/maintenance.py` | asyncpg.Pool → AsyncEngine |

### Deleted Files (6)
| File | Reason |
|------|--------|
| `inference/db/queries.py` | SQLLoader replaced by Alembic + inline text() |
| `inference/db/db_utils.py` | TypedPool/TypedConnection replaced by SQLAlchemy sessions |
| `inference/db/connection_recovery.py` | SQLAlchemy handles connection recovery |
| `inference/db/init_db.py` | Replaced by Alembic |
| `inference/db/interfaces.py` | Unused ABCs |
| `inference/db/sql/` (directory, ~130 files) | DDL in Alembic migration; queries inline in storage modules |

### Unchanged (consumption layer — same call patterns)
- `inference/server/app.py` — same `storage.initialize()` / `storage.close()` calls
- `inference/server/middleware/db_init_middleware.py` — same `storage.initialized` / `storage.initialize()`
- All server routers — same `storage.<service>.<method>()` calls
- All composer files — same `storage` singleton access
- `inference/db/cache_storage.py` — Redis caching unchanged
- `inference/db/serialization.py` — JSON utilities unchanged

---

## Risk & Rollback

### Risks

1. **Connection recovery behavior change**: SQLAlchemy's pool handles stale connections differently than our hand-rolled `DISCARD ALL` + `reload_schema()` approach. Monitor for stale OID errors after migration. Mitigation: SQLAlchemy's `pool_pre_ping=True` detects stale connections before use.

2. **TimescaleDB hypertable operations**: These are Timescale-specific and don't map to SQLAlchemy. Keeping them as raw SQL in Alembic is safe but means schema changes to hypertables require manual Alembic migrations (can't use `autogenerate`).

3. **Transaction boundaries in message_storage**: Currently sub-storage modules (thought, tool_call, etc.) receive an optional connection. Moving to internal session factory means message_storage needs to ensure sub-storages share its session. Solution: pass the session factory, and for composite operations, message_storage creates the session and passes it to sub-storage methods that accept an optional session parameter.

4. **Performance**: SQLAlchemy adds a layer of indirection. The `text()` path for complex queries should have near-identical performance to asyncpg. ORM paths add object construction overhead but are for simple CRUD where it's negligible.

### Rollback

The branch is on `simplify/runner` — a clean `git reset --hard` to pre-migration state rolls back everything. The migration doesn't change the database schema, so there's no data migration to reverse.

### Execution Order

Phases must execute in order (1 → 2 → 3 → 4 → 5). Within Phase 3, steps 3.1–3.14 should execute in the listed order (simplest first), but they are independently testable — each module can be verified in isolation before moving to the next.
