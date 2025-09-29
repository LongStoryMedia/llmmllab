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
- `schemas/composer_service_config.yaml` - System service settings
- `schemas/workflow_config.yaml` - User workflow preferences (caching, streaming, timeouts)  
- `schemas/tool_config.yaml` - User tool preferences (thresholds, generation, search)
- `schemas/user_config.yaml` - Complete user configuration schema
- `inference/composer/config.py` - Configuration loading and environment variables

**Usage Pattern:**
```python
# Access system settings
config.service.host, config.service.port

# Get user preferences with fallbacks
workflow_config = config.get_workflow_config(user_config.workflow)
tool_config = config.get_tool_config(user_config.tool)
```

## Environment Variable Management

The platform uses environment variables for infrastructure configuration with a **hierarchical override system**:

### Environment Variable Hierarchy
1. **Kubernetes Deployment** (`k8s/deployment.yaml`) - Production defaults
2. **Local Development** (`.env` files) - Development overrides  
3. **Runtime Configuration** - Dynamic user preferences via API

### Composer Service Environment Variables

**System Configuration (Infrastructure):**
```bash
# Service binding
COMPOSER_HOST=0.0.0.0
COMPOSER_PORT=8001
COMPOSER_DEBUG=false
COMPOSER_LOG_LEVEL=INFO

# Performance & Security
COMPOSER_ENABLE_CORS=true
COMPOSER_RATE_LIMIT_RPM=60
COMPOSER_HEALTH_CHECK_INTERVAL=30

# Virtual environment
COMPOSER_VENV=/opt/venv/composer
```

**User Configuration Defaults (UI-Customizable):**
```bash
# Workflow behavior defaults
COMPOSER_ENABLE_STREAMING=true
COMPOSER_MAX_PARALLEL_TOOLS=5
COMPOSER_DEFAULT_TIMEOUT=60.0
COMPOSER_CACHE_TTL=3600

# Tool behavior defaults
COMPOSER_TOOL_SIMILARITY_THRESHOLD=0.9
COMPOSER_ENABLE_TOOL_GENERATION=true
COMPOSER_TOOL_TIMEOUT=30.0
COMPOSER_SEARCH_TOP_K=10
```

### Environment Variable Best Practices

**When Adding New Environment Variables:**
1. **Add to Schema First**: Update relevant YAML schema in `schemas/`
2. **Update Config Loading**: Modify `composer/config.py` with proper parsing
3. **Add to Kubernetes**: Include in `k8s/deployment.yaml` with production defaults
4. **Document**: Add to `docs/k8s_environment_variables.md`
5. **Validate**: Test with `debug/test_k8s_env_vars.py`

**Environment Variable Naming Conventions:**
- **System Settings**: `{SERVICE}_{SETTING}` (e.g., `COMPOSER_HOST`)
- **User Defaults**: `{SERVICE}_{CATEGORY}_{SETTING}` (e.g., `COMPOSER_TOOL_TIMEOUT`)
- **Boolean Values**: Use `"true"/"false"` strings (lowercase)
- **Numeric Values**: Use string representations with proper validation

**Testing Environment Variables:**
```bash
# Validate Kubernetes deployment configuration
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python debug/test_k8s_env_vars.py

# Test configuration loading locally
COMPOSER_DEBUG=true COMPOSER_PORT=8002 python -c "from composer.config import config; print(config.service.debug)"

# Check all composer env vars in pod
k exec -it -n ollama $POD_NAME -- env | grep COMPOSER
```

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
the ui is fully local and connects to remote inference services.


### Code Generation
```bash
# Generate Python and TypeScript models from YAML schemas
./regenerate_models.sh

# Language-specific generation
./regenerate_models.sh python     # Generate only Python models
./regenerate_models.sh typescript # Generate only TypeScript models
```

## Critical Patterns

### Multi-Environment Architecture
The inference service uses **four isolated Python environments**:
- `evaluation/`: Benchmarking and fine-tuning (separate deps from serving)
- `server/`: FastAPI REST services 
- `runner/`: Model execution pipelines
- `composer/`: LangGraph-based workflow orchestration and agentic system runtime

Always use `/app/v.sh {service}` when executing commands in Kubernetes pods.

### Schema-Driven Development
YAML schemas in `schemas/` define the data contracts. When modifying APIs:
1. Update relevant YAML schema first
2. Run `./regenerate_models.sh` to generate Python models and TypeScript types
3. Generated files: `inference/models/*.py`, `ui/src/types/*.ts`

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

## Database Access
The platform uses PostgreSQL for persistent storage. 
Access the database from within the psql Kubernetes pod:

```bash
k exec -it psql-0 -n psql -- psql -h localhost -U lsm -d llmmll -v "ON_ERROR_STOP=1" -c "<SQL_COMMAND>"
```  

## SQL files
SQL schema and migration files are in `inference/server/db/sql/`.
The code interfaces are in `inference/server/db/`. 

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
3. **Update Config**: Modify `composer/config.py` parsing
4. **Add to K8s**: Include in `k8s/deployment.yaml`
5. **Test**: Use `debug/test_k8s_env_vars.py`
6. **Document**: Update `docs/k8s_environment_variables.md`

**Validation Commands:**
```bash
# Full environment validation
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python debug/test_k8s_env_vars.py

# Quick config check
k exec -it -n ollama $POD_NAME -- /app/v.sh composer python -c "from composer.config import config; print('✅ Config loaded')"

# Environment variable debugging
k exec -it -n ollama $POD_NAME -- env | grep -E "(COMPOSER|DB_|REDIS_)" | sort
```

---
DO NOT USE LONG OR COMPLEX COMMANDS. USE SCRIPTS INSTEAD.
ALWAYS SYNC CODE WITH `inference/sync-code.sh` INSTEAD OF MANUAL RSYNC/CP COMMANDS.
EVERY CHANGE SHOULD HAVE A GIT COMMIT.

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

DO NOT ADD DOCUMENTATION FOR FIXES. ONLY DOCUMENT FULLY IMPLEMENTED FEATURES, AND ALWAYS IN THE `docs/` FOLDER. ALWAYS LINK TO THE DOCS FROM THE README IF IT'S IMPORTANT.

## Testing Strategy

### Unit Tests (`inference/test/`)
Use for **automated testing** of interfaces, components, and business logic:
- **When**: Testing public APIs, service interfaces, data models, utility functions
- **Characteristics**: Fast, isolated, mockable dependencies, no external services
- **Examples**: Functional interfaces, configuration parsing, data validation, error handling
- **Framework**: pytest with mocking for external dependencies
- **Execution**: `python -m pytest test/` or individual test files

### Manual Verification Tests (`inference/debug/`)
Use for **manual validation** and **integration testing** requiring real services:
- **When**: End-to-end workflows, GPU operations, database connections, model execution
- **Characteristics**: Requires real infrastructure, longer execution time, manual inspection
- **Examples**: Model pipeline testing, database queries, GPU memory validation, service integration
- **Framework**: Standalone Python scripts with detailed output
- **Execution**: Direct script execution in pods or local environment with real services

**Rule**: If testing an **interface** (API, service boundary, public functions), write **unit tests**. If testing **integration** or requiring **real infrastructure**, use **debug scripts**.

