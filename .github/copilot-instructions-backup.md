# LLM ML Lab - AI Agent Instructions

## Project Architecture

LLM ML Lab is a multi-modal language model platform with microservice architecture:

- **inference/**: Python services (evaluation, server, runner, composer) with isolated virtual environments
     - **evaluation/**: Model benchmarking and fine-tuning tools
     - **server/**: FastAPI REST services for model interaction (calls runner for execution)
     - **runner/**: Model execution pipelines with dynamic tool integration
     - **composer/**: LangGraph-based workflow orchestration and agentic system runtime (NEW)
- **ui/**: React TypeScript frontend with Material UI Joy
- **schemas/**: YAML schema definitions for type safety across services (generates code via `./regenerate_models.sh`)

## Configuration Architecture

The platform uses a **hierarchical configuration system**:

- **System Config**: Service settings (host, port, database) - not user configurable
- **User Config**: Workflow & tool preferences - customizable per user via UI
- **Schema-Driven**: YAML schemas generate Python/TypeScript models automatically

**Key Files:**
- `schemas/config.yaml` - System service settings that holds system settings and default values for user configuration
- `schemas/user_config.yaml` - Complete user configuration schema

## Environment Variable Management

The platform uses environment variables for infrastructure configuration with a **hierarchical override system**:

### Environment Variable Hierarchy
1. **Kubernetes Deployment** (`k8s/deployment.yaml`) - Production defaults
2. **Runtime Configuration** - Dynamic user preferences via API

### Environment Variable Best Practices

**When Adding New Environment Variables:**
1. **Add to Schema First**: Update relevant YAML schema in `schemas/`
1. **Add to Kubernetes**: Include in `k8s/deployment.yaml` with production defaults
1. **Document**: Add to `docs/k8s_environment_variables.md`
1. **Validate**: Test with `debug/test_k8s_env_vars.py`

### Database & Infrastructure Variables

**Required for Service Startup:**
```bash
# Database connectivity
DATABASE_URL=postgresql://user:pass@host:port/db
DB_HOST=192.168.0.71
DB_PORT=32345
DB_USER=lsm
DB_PASSWORD=<from_secret>
DB_NAME=llmmll

# Redis caching
REDIS_HOST=192.168.0.71  
REDIS_PORT=32346
REDIS_DB=0

# Cross-module imports
PYTHONPATH=/app
```

**Development vs Production:**
- **Local Development**: Use `.env` files or direct shell exports
- **Kubernetes Deployment**: Use deployment.yaml with secrets for sensitive values
- **Testing**: Use validation scripts to ensure proper configuration

## Key Development Workflows

### Environment Setup

```bash
# Kubernetes pod commands (production/staging)
POD_NAME=$(k get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
k exec -it -n ollama $POD_NAME -- /app/v.sh server python -m uvicorn app:app --port 8000
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python -m uvicorn app:app --port 8001
k exec -it -n ollama $POD_NAME -- /app/v.sh runner python -c "import torch; print(torch.cuda.is_available())"
```

inference does not generally run locally due to hardware needs. use `inference/sync-code.sh` to sync code to remote cluster.
However, when spot checking code and imports, make sure to always source `./.venv/bin/activate`
the ui is fully local and connects to remote inference services.


### Code Generation
```bash
# Generate Python and TypeScript models from YAML schemas
./regenerate_models.sh

# Language-specific generation
./regenerate_models.sh python     # Generate only Python models
./regenerate_models.sh typescript # Generate only TypeScript models

# Generate a new model from a schema (creates both Python model and updates __init__.py)
schema2code schemas/[name].yaml -l python -o inference/models/[name].py

# Generate TypeScript types
schema2code schemas/[name].yaml -l typescript -o ui/src/types/[name].ts
```

### Database Development Workflow

**Adding New Storage Services:**
1. **Schema First**: Create YAML schema in `schemas/[entity].yaml`
2. **Generate Models**: Run `./regenerate_models.sh` to create Pydantic models
3. **SQL Files**: Create SQL files in `db/sql/[entity]/` directory
4. **Storage Service**: Implement `[entity]_storage.py` following established patterns
5. **Integration**: Add to `init_db.py` and `db/__init__.py`
6. **Testing**: Add unit tests and integration validation

**SQL Development Rules:**
- All SQL files must be idempotent (can be run multiple times safely)
- Use proper parameter binding (`$1`, `$2`) never string formatting
- Include appropriate indexes for query performance
- Handle TimescaleDB and pgvector extensions gracefully
- Follow consistent naming conventions (plural table names)

**Database Testing:**
```bash
# Test database initialization
k exec -it -n ollama $POD_NAME -- /app/v.sh server python -c "from db.init_db import initialize_database; import asyncio; asyncio.run(initialize_database(pool))"

# Test storage service
k exec -it -n ollama $POD_NAME -- /app/v.sh server python -c "from db import storage; print('✅ Storage initialized' if storage.initialized else '❌ Not initialized')"

# Validate SQL queries load correctly
k exec -it -n ollama $POD_NAME -- /app/v.sh server python -c "from db.queries import get_query; print(get_query('[entity].[operation]'))"
```

## Critical Patterns

### Multi-Environment Architecture
The inference service uses **four isolated Python environments**:
- `evaluation/`: Benchmarking and fine-tuning (separate deps from serving)
- `server/`: FastAPI REST services 
- `runner/`: Model execution pipelines
- `composer/`: LangGraph-based workflow orchestration and agentic system runtime

Always use `/app/v.sh {service}` when executing commands in Kubernetes pods.

### Shared Code Architecture

The inference system has shared components that all services can access:

- **`models/`**: Generated Pydantic models from YAML schemas (shared data contracts)
- **`utils/`**: Shared utility functions (serialization, response handling, etc.)
- **`db/`**: Database interfaces, storage classes, and SQL queries (MOVED from server/db/)
- **`test/`**: Shared test utilities and fixtures
- **`debug/`**: Validation and debugging scripts

**Shared Component Access Rules:**
- ✅ All services can import from `models.*`, `utils.*`, `db.*`
- ✅ Services can use shared database interfaces and storage classes
- ✅ Shared utilities provide common functionality without coupling
- ❌ Shared components cannot import from specific services (server, runner, composer)
- ❌ Business logic must not be placed in shared components

**Import Organization Rules:**
- ✅ **Top-level imports preferred:** Place all imports at the top of the file unless there is a specific technical reason not to
- ✅ **Standard import order:** Standard library → Third-party packages → Local imports (use `isort` format)
- ✅ **Explicit imports:** Use explicit imports (`from module import Class`) rather than star imports (`from module import *`)
- ❌ **Lazy imports only when necessary:** Only use function-level imports for circular dependencies, optional dependencies, or significant performance reasons
- ❌ **Avoid nested imports:** Don't place imports inside classes or deeply nested functions unless required for functionality

**Valid reasons for non-top-level imports:**
- **Circular dependency resolution:** When modules have mutual dependencies
- **Optional dependencies:** When imports are conditionally available (`try/except ImportError`)
- **Heavy/expensive imports:** When import significantly impacts startup time and is rarely used
- **Dynamic imports:** When import path is determined at runtime
- **Plugin systems:** When loading modules dynamically based on configuration

### Component Interface Layers

Each service component defines its **public API boundary** through its `__init__.py` file:

**Composer Interface (`composer/__init__.py`):**
- `initialize_composer()`, `compose_workflow()`, `execute_workflow()`
- Abstracts LangGraph workflow construction and execution
- Provides clean API for workflow orchestration without internal coupling
- Enables other services to use composer functionality via Protocol interfaces

**Runner Interface (`runner/__init__.py`):**
- `pipeline_factory()`, `run_pipeline()`, `stream_pipeline()`, `embed_pipeline()`
- Abstracts model execution pipelines and streaming functionality
- Provides clean API for pipeline execution without internal coupling
- Enables other services to use runner functionality via Protocol interfaces

**Interface Layer Principles:**
- ✅ Define clean API boundaries between components
- ✅ Abstract internal implementation details from external consumers
- ✅ Enable Protocol-based dependency injection patterns
- ✅ Document all public functions with clear architectural roles
- ❌ Expose internal classes or implementation details
- ❌ Create dependencies on other service components

### Schema-Driven Development
YAML schemas in `schemas/` define the data contracts. When modifying APIs:
1. Update relevant YAML schema first
2. Run `./regenerate_models.sh` to generate Python models and TypeScript types
3. Generated files: `inference/models/*.py`, `ui/src/types/*.ts`

**Adding New Schemas:**
1. Create new YAML schema in `schemas/[name].yaml`
2. Generate Python model: `schema2code schemas/[name].yaml -l python -o inference/models/[name].py`
3. Generate TypeScript types: `schema2code schemas/[name].yaml -l typescript -o ui/src/types/[name].ts`
4. The schema2code tool automatically updates `__init__.py` with exports

**Schema Design Rules:**
- **Avoid Duplication**: If an enum or structure is used in multiple schemas, extract it to a separate schema file
- **Use $ref**: Reference shared schemas using `$ref: "shared_schema.yaml"` instead of copying definitions
- **Single Source of Truth**: Each data structure should be defined exactly once
- **Example**: Instead of duplicating computational requirements enum, create `computational_requirement.yaml` and reference it

### Memory Management
The platform implements sophisticated memory optimization:
- Models loaded on-demand and unloaded after use
- GPU memory tracking and automatic resource management
- Multiple memory optimization strategies based on available VRAM

## Service Communication

```
UI (React) ←→ REST API (FastAPI) ←→ Model Runner
                        ↓
                 GPU Resources
```

## Development Commands

```bash
# Validate all code before commits
make validate  # TypeScript + Python syntax + Pyright type checking

# Start development environment
make start  # Parallel: inference-dev + UI dev server

# Sync code to remote cluster during development
./inference/sync-code.sh -w  # Watch mode with auto-sync
```
avoid commands that I need to manually approve. 
avoid overly complex or long commands as they often fail due to timeouts. Instead, write a script and call that.
always try `inference/sync-code.sh` when syncing code (it will almost always work the second time if it doesn't the first)

## File Conventions

- **API endpoints**: REST in `server/`
- **Configuration**: Centralized in `schemas/config.yaml` with component-specific refs
- **Kubernetes**: Deployments in `{service}/k8s/` with `apply.sh` automation
- **Container startup**: `inference/startup.sh` orchestrates multi-service containers

## Context Extension System

The platform includes a sophisticated context extension system (see `docs/context_extension.md`):
- Extends LLM context windows beyond token limitations
- Semantic memory retrieval from conversation history
- External search integration for real-time knowledge
- Hierarchical summarization for context compression

When working with chat/completion features, consider how changes affect context window management.

## Database Layer Architecture

The platform uses **PostgreSQL with TimescaleDB** for persistent storage, following established patterns for storage services, SQL organization, and database initialization.

### Database Access Patterns

**Direct Database Access:**
```bash
k exec -it psql-0 -n psql -- psql -h localhost -U lsm -d llmmll -v "ON_ERROR_STOP=1" -c "<SQL_COMMAND>"
```

**Storage Service Pattern:**
- All database interactions use storage services in `db/`
- Storage services follow consistent naming: `[entity]_storage.py` (e.g., `message_storage.py`, `conversation_storage.py`)
- Services use `asyncpg.Pool`, `typed_pool()`, and `get_query()` function for SQL loading

### SQL File Organization

**Directory Structure:**
```
db/sql/
├── message/          # Message-related SQL files
├── conversation/     # Conversation-related SQL files
├── memory/          # Memory storage SQL files  
├── research/        # Research task SQL files
└── [entity]/        # Each entity gets its own directory
```

**SQL File Types:**
- **Schema Files**: `create_[table]_table.sql` - Table creation (idempotent)
- **Index Files**: `create_[table]_indexes.sql` - Index creation (idempotent)
- **Operation Files**: `create_[entity].sql`, `get_[entity].sql`, `update_[entity].sql` - CRUD operations
- **Extension Files**: `create_[table]_hypertable.sql` - TimescaleDB hypertables
- **Migration Files**: Named with clear purpose, all idempotent

### SQL File Requirements

**Idempotency Rules:**
- ✅ Use `CREATE TABLE IF NOT EXISTS`
- ✅ Use `CREATE INDEX IF NOT EXISTS`
- ✅ Use `SELECT create_hypertable('[table]', '[column]', if_not_exists => TRUE)`
- ✅ Handle conflicts gracefully with `ON CONFLICT` or conditional logic
- ❌ Never use `CREATE TABLE` without `IF NOT EXISTS`
- ❌ Never write SQL that fails if run multiple times

**Query Organization:**
- **Namespaced Keys**: Use `[entity].[operation]` (e.g., `message.get_content`, `research.create_task`)
- **Parameter Binding**: Always use `$1`, `$2`, etc. - never string interpolation
- **Consistent Naming**: Table names are plural (`messages`, `research_tasks`)
- **Performance**: Include appropriate indexes for common queries

### Storage Service Implementation

**Required Constructor Pattern:**
```python
class [Entity]Storage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = logging.getLogger(__name__)
```

**Database Operation Pattern:**
```python
async def create_entity(self, entity: EntityModel) -> int:
    async with self.typed_pool.acquire() as conn:
        row = await conn.fetchrow(
            self.get_query("entity.create_entity"),
            entity.field1,
            entity.field2
        )
        return row["id"] if row and "id" in row else None
```

### Database Initialization Integration

**Add to `init_db.py`:**
All new storage services must be integrated into the initialization system:

```python
# Add to initialization_steps list:
(
    "Creating [entity] tables",
    [
        ("[entity].create_[entity]_table", []),
        ("[entity].create_[entity]_indexes", []),
        ("[entity].create_[entity]_hypertable", ["timescaledb"]),
        ("[entity].create_[entity]_embedding_index", ["vector"]),
    ],
)
```

**Service Registration in `db/__init__.py`:**
```python
# Import storage service
from .[entity]_storage import [Entity]Storage

# Add to Storage class __init__:
self.[entity] = None

# Add to Storage class initialize method:
self.[entity] = [Entity]Storage(self.pool, get_query)

# Add to Storage class close method:
self.[entity] = None
```

### Extension Integration

**TimescaleDB Support:**
- Use hypertables for time-series data (partitioned by `created_at`)
- Enable compression policies for historical data
- Set retention policies for data lifecycle management
- Check for extension availability before creating time-series features

**pgvector Support:**
- Use `VECTOR(768)` columns for embeddings
- Create `ivfflat` indexes with `vector_cosine_ops`
- Implement similarity search with `<=>` operator
- Always check vector extension availability before vector operations

### Error Handling Patterns

**Connection Management:**
```python
try:
    async with self.typed_pool.acquire() as conn:
        # Database operations
except Exception as e:
    logger.error(f"Database error: {e}")
    raise
```

**Query Execution:**
- Always use parameterized queries to prevent SQL injection
- Handle connection timeouts and retries
- Log errors with sufficient context for debugging
- Use appropriate transaction boundaries for multi-step operations

### Performance Considerations

**Indexing Strategy:**
- Primary keys: Always `SERIAL PRIMARY KEY`
- Foreign keys: Always indexed
- Query columns: Index based on WHERE clauses
- Time-series: Use TimescaleDB hypertables for time-based partitioning
- Text search: Use appropriate indexing for search patterns

**Query Optimization:**
- Use `EXPLAIN ANALYZE` to validate query performance
- Limit result sets with pagination
- Use appropriate JOIN strategies
- Consider query caching for frequent operations 

## Web Scraping
Web scraping is handled by Scrapy in `inference/server/services/web_extraction_service.py`.
these are the settings available: https://docs.scrapy.org/en/latest/topics/settings.html
and main docs: https://docs.scrapy.org/en/latest/

## Environment Variable Troubleshooting

### Common Configuration Issues

**Service Won't Start:**
1. Check required environment variables are set
2. Validate boolean values are "true"/"false" (lowercase)
3. Ensure numeric values are within schema constraints
4. Verify virtual environment paths exist

**Configuration Not Loading:**
```bash
# Debug configuration loading in pod
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python -c "from composer.config import config; print('Host:', config.service.host, 'Port:', config.service.port)"

# Check environment variable parsing
k exec -it -n ollama $POD_NAME -- env | grep COMPOSER | head -10
```

**Schema Validation Errors:**
1. Run validation script: `debug/test_k8s_env_vars.py`
2. Check YAML schema constraints in `schemas/`
3. Verify environment variable naming conventions
4. Ensure proper type conversion (string → bool/int/float)

**User Config Override Issues:**
```bash
# Test user preference resolution
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python -c "
from composer.config import config
from models.workflow_config import WorkflowConfig
user_config = WorkflowConfig(enable_streaming=False)
resolved = config.get_workflow_config(user_config)
print('User override working:', not resolved.enable_streaming)
"
```

### Environment Variable Development Workflow

**When Adding New Environment Variables:**
1. **Schema First**: Update `schemas/[service]_config.yaml`
2. **Generate Models**: Run `./regenerate_models.sh`
3. **Add to K8s**: Include in `k8s/deployment.yaml`
4. **Test**: Use `debug/test_k8s_env_vars.py`
5. **Document**: Update `docs/k8s_environment_variables.md`

**Validation Commands:**
```bash
# Full environment validation
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python debug/test_k8s_env_vars.py

# Quick config check
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python -c "from composer.config import config; print('✅ Config loaded')"

# Environment variable debugging
k exec -it -n ollama $POD_NAME -- env | grep -E "(COMPOSER|DB_|REDIS_)" | sort
```

**Documentation Standards:**
- **Current State ONLY**: Document what the system currently does, never what it was changed from
- **No Change Documentation**: Never document fixes, cleanups, refactors, migrations, or "before/after" states
- **No Transformation History**: Avoid any mention of previous implementations or what was replaced
- **Implementation Focus**: Describe current functionality, integration points, and usage patterns only
- **Architecture Description**: Document current system architecture, capabilities, and design decisions
- **Feature Documentation**: Only document completed, production-ready features and their usage

**Code Replacement Policy:**
- **Remove and Replace**: When fixing architectural violations or major issues, remove the problematic file completely and replace with clean implementation
- **No Deprecation Warnings**: Don't add deprecation notices or warnings - simply remove and replace
- **No Alias Files**: Don't create compatibility alias files that reference what was replaced
- **Clean Slate**: New implementations should not reference or mention what they replaced
- **Direct Replacement**: Use the same filename when replacing functionality to minimize import disruption
- **Update Imports**: Update all imports to use the new clean implementation directly

## Testing Strategy

### Unit Tests (`inference/test/`)
Use for **automated testing** of interfaces, components, and business logic:
- **When**: Testing public APIs, service interfaces, data models, utility functions, schema validation
- **Characteristics**: Fast, isolated, mockable dependencies, no external services, automated execution
- **Examples**: Pydantic model creation, configuration parsing, data validation, error handling, interface compatibility
- **Framework**: pytest with mocking for external dependencies
- **Execution**: `python -m pytest test/` or individual test files
- **Key Indicator**: Can run without database, GPU, or network - pure Python logic testing

### Manual Verification Tests (`inference/debug/`)
Use for **manual validation** and **integration testing** requiring real services:
- **When**: End-to-end workflows, GPU operations, database connections, model execution, system integration
- **Characteristics**: Requires real infrastructure, longer execution time, manual inspection, external dependencies
- **Examples**: Model pipeline testing, database queries, GPU memory validation, service integration, LLM execution
- **Framework**: Standalone Python scripts with detailed output
- **Execution**: Direct script execution in pods or local environment with real services
- **Key Indicator**: Needs actual database, GPU, network services, or manual human verification

**Rule**: If testing **pure Python logic** (models, interfaces, parsing, validation), write **unit tests**. If testing **system integration** or requiring **real infrastructure**, use **debug scripts**.


## IMPORTANT NOTES

DO NOT USE LONG OR COMPLEX COMMANDS. USE SCRIPTS INSTEAD.
ALWAYS SYNC CODE WITH `inference/sync-code.sh` INSTEAD OF MANUAL RSYNC/CP COMMANDS.
EVERY CHANGE SHOULD HAVE A GIT COMMIT.

## IMPORT PATH RULES

NEVER USE HARDCODED PATH IMPORTS:
```python
# ❌ NEVER DO THIS
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')
```

USE PROPER PYTHON MODULE IMPORTS AND PYTHONPATH CONFIGURATION INSTEAD.
RELY ON VIRTUAL ENVIRONMENT ACTIVATION AND PROJECT STRUCTURE FOR IMPORTS.

## DATABASE LAYER RULES

ALL SQL FILES MUST BE IDEMPOTENT - THEY MUST BE SAFE TO RUN MULTIPLE TIMES.
FOLLOW THE ESTABLISHED STORAGE SERVICE PATTERNS - USE `typed_pool`, `get_query`, AND PROPER ERROR HANDLING.
ADD ALL NEW STORAGE SERVICES TO `init_db.py` INITIALIZATION AND `db/__init__.py` REGISTRATION.
SQL FILES GO IN `db/sql/[entity]/` DIRECTORIES WITH CLEAR NAMING CONVENTIONS.
ALWAYS USE PARAMETER BINDING ($1, $2) NEVER STRING FORMATTING IN SQL.
TEST DATABASE INITIALIZATION AND STORAGE SERVICES AFTER CHANGES.

NESTED COMMANDS SUCH AS
```bash
POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}') && kubectl exec -it -n ollama $POD_NAME -- /app/v.sh server python test_real_end_to_end_pipeline.py qwen3-30b-a3b-q4-k-m
```
ARE TOO COMPLEX FOR AUTO-APPROVAL. USE SOMETHING LIKE:
```bash
kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}'
# remember the pod name printed
kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh server python -m debug.test_real_end_to_end_pipeline qwen3-30b-a3b-q4-k-m
```

DO NOT ADD DOCUMENTATION FOR FIXES, CHANGES, CLEANUPS, OR TRANSFORMATIONS. ONLY DOCUMENT FULLY IMPLEMENTED FEATURES, AND ALWAYS IN THE `docs/` FOLDER. ALWAYS LINK TO THE DOCS FROM THE README IF IT'S IMPORTANT.

ADD IMPORTS ONLY AT THE TOP OF THE FILE UNLESS THERE IS A TECHNICAL REASON NOT TO. AVOID NESTED IMPORTS UNLESS ABSOLUTELY NECESSARY. IF NECESSARY, ADD A COMMENT ON THE SAME LINE `# pylint: disable=import-outside-toplevel`, AND A COMMENT ABOVE EXPLAINING WHY IT'S NEEDED. A COMMON IMPORT WHERE THIS IS NEEDED IS `from db import storage` TO AVOID CIRCULAR DEPENDENCIES.

IF THE POD IS IN AN ERROR STATE, FIX IT. DO NOT IGNORE ERRORS OR LEAVE THE POD IN A BAD STATE.