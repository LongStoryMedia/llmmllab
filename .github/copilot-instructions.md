# LLM ML Lab - AI Agent Instructions

NO GUESSWORK!
ALWAYS be certain and check the codebase!
ALWAYS create a git commit with a descriptive message after making changes.
ALWAYS sync code after making changes.
NEVER leave the kubernetes pod in a crash loop or bad state.
NEVER use complex nested commands; use simple commands or scripts instead.
NEVER leave code that didn't help solve the problem, or create the feature. (cleanup)

## Architecture Overview

**Core Services:**
- **inference/**: Python services running in unified container environment
  - **server/**: FastAPI REST services  
  - **runner/**: Pure LLM interface (simplified from orchestration component)
  - **composer/**: LangGraph workflow orchestration and agentic system runtime
  - **evaluation/**: Model benchmarking tools
- **ui/**: React TypeScript frontend
- **schemas/**: YAML schema definitions → auto-generated Python/TypeScript models

**Key Architectural Pattern:** Composer is the authoritative runtime for LangGraph execution, removing orchestration from runner.

## Development Workflow

### Code Generation
```bash
./regenerate_models.sh  # Generate Python + TypeScript from YAML schemas
```

### Environment Setup
```bash
# Container commands (no separate virtual environments)
kubectl exec -it -n ollama $(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}') -- /app/v.sh python [command]

# Code sync (use this, not manual rsync/cp)
inference/sync-code.sh
```

**NOTE:** if sync-code.sh fails, try again. it always works on the second attempt.
**Never use long commands.** Use simple commands instead. if necessary, use a script.

### Database Development
1. **Schema First**: Create YAML schema in `schemas/[entity].yaml`
2. **Generate Models**: Run `./regenerate_models.sh`
3. **SQL Files**: Create idempotent SQL in `db/sql/[entity]/`
4. **Storage Service**: Implement `[entity]_storage.py` with `typed_pool`, `get_query` patterns
5. **Register**: Add to `init_db.py` and `db/__init__.py`

### Model Creation (NEW)
**For creating new standalone Pydantic models:**
1. **Create Schema**: Define YAML schema in `schemas/[model_name].yaml` 
2. **Use schema2code**: Generate single model with `schema2code --language python --output inference/models/[model_name].py schemas/[model_name].yaml`
3. **NOT regenerate_models.sh**: The `regenerate_models.sh` script only processes existing schemas, use `schema2code` for new models

## Critical Rules

### Documentation Guidelines
- ✅ **Document current state only** - describe what exists, not what changed
- ✅ **Update existing docs** to reflect current architecture and implementation
- ❌ **Don't document changes** - no "before/after", "migration", or "update" documentation
- ✅ **Create documentation** if none exists for current features
- ✅ **Focus on usage and architecture** - how things work now, not how they evolved

### Import Organization
- ✅ **Top-level imports preferred** - place at file top unless technical reason prevents it
- ❌ **Avoid nested imports** - only use for circular dependencies or performance reasons
- ❌ **Never hardcode paths**: `sys.path.append('/Users/...')` 

### Database Layer
- ✅ **All SQL must be idempotent** - safe to run multiple times
- ✅ **Use parameter binding** ($1, $2) never string formatting
- ✅ **Follow patterns**: `typed_pool()`, `get_query()`, proper error handling

### Component Architecture
**Composer (NEW):** All LangGraph orchestration, workflow construction, agent nodes
**Runner (SIMPLIFIED):** Pure LLM interface, grammar constraints, streaming - NO orchestration
**Shared Code:** `models/`, `utils/`, `db/` - shared across all services

### Configuration
- **System Config**: `schemas/config.yaml` - service settings
- **User Config**: `schemas/user_config.yaml` - user preferences  
- **Environment Variables**: Add to schema first, then k8s deployment, then validate

## Quick Commands

```bash
# Test syntax
kubectl exec -n ollama $POD_NAME -- python3 -c "import ast; ast.parse(open('file.py').read())"

# Debug database
kubectl exec -it psql-0 -n psql -- psql -h localhost -U lsm -d llmmll -c "SELECT 1"

# Validate environment
kubectl exec -n ollama $POD_NAME -- /app/v.sh python -c "from composer.config import config; print('✅ Config loaded')"
```

## Testing Strategy
- **Unit Tests** (`inference/test/`): Pure Python logic, fast, mockable
- **Manual Verification** (`inference/debug/`): Integration tests requiring real services

## Critical Notes
- **Always sync with** `inference/sync-code.sh` - never use manual rsync/cp
- **Always commit changes** with descriptive messages
- **Avoid complex nested commands** - use scripts instead
- **Fix pod errors immediately** - don't ignore crash loops or leave pods in bad state
- **Document only completed features** in `docs/` folder
- **Always source the virtual environment** before running Python commands

## Import Path Rules

**NEVER use hardcoded path imports:**
```python
# ❌ NEVER DO THIS
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')
```

**Use proper Python module imports and PYTHONPATH configuration instead.**
**Rely on virtual environment activation and project structure for imports.**

## Database Layer Rules

- **ALL SQL files must be idempotent** - safe to run multiple times
- **Follow established storage service patterns** - use `typed_pool`, `get_query`, proper error handling
- **Add all new storage services** to `init_db.py` and `db/__init__.py` registration
- **SQL files go in `db/sql/[entity]/`** directories with clear naming conventions
- **Always use parameter binding** ($1, $2) never string formatting in SQL
- **Test database initialization** and storage services after changes

## Command Complexity Rules

**Avoid nested commands like:**
```bash
POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}') && kubectl exec -it -n ollama $POD_NAME -- /app/v.sh server python test.py
```

**Use simple commands instead:**
```bash
kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}'
# Remember the pod name, then:
kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh server python -m debug.test
```

## Python Module Execution Rules

**ALWAYS run Python files as modules using -m flag:**
```bash
# ✅ CORRECT - Run as module with proper import resolution
kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh runner python -m debug.test_qwen3_pipeline_creation

# ❌ WRONG - Direct file execution causes import issues
kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh runner python debug/test_qwen3_pipeline_creation.py
```

**Module execution benefits:**
- Proper PYTHONPATH resolution
- Correct relative imports
- Avoids "No module named" errors
- Follows Python best practices

## Schema-Driven Development
1. Update YAML schema in `schemas/`
2. Run `./regenerate_models.sh` 
3. Generated files: `inference/models/*.py`, `ui/src/types/*.ts`
4. Use generated models for type safety across services

# A Task Is Not Complete If:
- There are any linting errors
- There are any import path issues
- There are any hardcoded paths
- the kubernetes pod is not running correctly
- the pattern does not follow the documented architecture or patterns in existing code

# REMEMBER:

NO GUESSWORK!
ALWAYS be certain and check the codebase!
ALWAYS create a git commit with a descriptive message after making changes.
ALWAYS sync code after making changes.
NEVER leave the kubernetes pod in a crash loop or bad state.
NEVER use complex nested commands; use simple commands or scripts instead.
NEVER leave code that didn't help solve the problem, or create the feature. (cleanup)