# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development

```bash
make start          # Start inference (dev mode) and UI in parallel
make inference-dev  # Start inference service only (syncs code to k8s, tails logs)
make start-ui       # Start UI dev server only (cd ui && npm run dev)
```

### Testing

```bash
make test                           # Run all tests (inference + UI)
cd inference && pytest test/        # Run Python tests only
cd inference && pytest test/unit/test_foo.py  # Run a single test file
cd ui && npm run test               # Run UI tests (Vitest)
```

### Validation & Linting

```bash
make validate       # TypeScript tsc --noEmit + Python compileall + Pyright type check
cd ui && npm run lint   # ESLint on UI code
```

### Code Generation

```bash
./regenerate_models.sh              # Regenerate all models from YAML schemas
./regenerate_models.sh python       # Python models only → inference/models/
./regenerate_models.sh typescript   # TypeScript types only → ui/src/types/
```

### Deployment

```bash
make deploy         # Deploy all services (inference, maistro, ui) to k8s
make clean          # Remove build artifacts (debug/out/, ui/build/, inference/models/)
make e2e-<test>     # Run end-to-end test inside k8s pod
make clear-debug    # Remove debug output files and sync code
```

## Architecture

### Structure

```
inference/      Python FastAPI inference service (deployed to k8s)
  server/       API layer: routers, middleware, app.py entry point
  runner/       Model execution: pipeline_factory, pipeline_cache, pipelines/
  composer/     LangGraph agent orchestration, tool generation
  db/           Multi-tier storage: PostgreSQL, Redis, in-memory
  evaluation/   Model benchmarking
  models/       Generated from YAML schemas (do not edit directly)
  debug/        Debug utilities and output files
ui/             React 19 + Vite frontend
schemas/        YAML schema definitions (source of truth for models)
docs/           Architecture documentation
```

### Key Architectural Patterns

**Schema-Driven Development**: All data contracts are defined as YAML schemas in `schemas/`. The `schema2code` tool generates `inference/models/*.py` and `ui/src/types/*.ts` from these. **Never edit generated model files directly** — edit the YAML schema and regenerate.

**Pipeline System**: The runner uses a pluggable pipeline pattern. `pipeline_factory.py` creates appropriate pipelines (text, image, embeddings, multimodal). `pipeline_cache.py` manages instances. All pipeline implementations live in `runner/pipelines/`.

**Multi-Tier Caching**: User config flows in-memory → Redis → PostgreSQL. See `docs/multi_tier_user_config_caching.md`.

**Provider Compatibility**: The server implements both OpenAI-compatible endpoints (`routers/openai/`) and Anthropic-compatible endpoints (`routers/anthropic/`). These share the underlying runner/pipeline infrastructure.

**Streaming**: Chat and image responses stream token-by-token. The streaming architecture is documented in `docs/RUNNER_ARCHITECTURE_OVERHAUL.md`.

**Deployment**: The inference service runs in Kubernetes (`inference/k8s/`). `make inference-dev` syncs local code to the cluster via `inference/sync-code.sh` and tails logs.

### Composer Isolation

**ServerInterface Pattern**: Composer is isolated from server through the `composer/server/interface.py` protocol. All data access flows through server services accessed via the `ServerInterface` protocol:

- `user_config`: UserConfigService for configuration retrieval
- `conversation`: ConversationService for conversation management
- `message`: MessageService for message storage/retrieval
- `memory`: MemoryService for vector search and storage
- `summary`: SummaryService for conversation summaries
- `model_profile`: ModelProfileService for model configuration
- `dynamic_tool`: DynamicToolService for tool management

**Dependency Injection**: Workflow builders (`GraphBuilder` subclasses) receive server services through constructor injection rather than importing the singleton server. This enables:
- Clean architectural separation between composer and server
- Easier testing with mock services
- Independent deployment of composer component

**Workflow Factory Pattern**: The `composer/graph/workflows/factory.py` provides `get_builder()` to create appropriate workflow builders (IDE or Dialog) with proper dependency injection.

**ComposerService**: The `ComposerService` in `composer/core/service.py` orchestrates graph construction and execution. It receives a `GraphBuilder` and optional `ServerInterface` for per-request data access.

### Key Entry Points

| Component | Entry Point |
|-----------|-------------|
| FastAPI app | `inference/server/app.py` |
| OpenAI chat endpoint | `inference/server/routers/openai/chat.py` |
| Anthropic messages endpoint | `inference/server/routers/anthropic/messages.py` |
| Pipeline creation | `inference/runner/pipeline_factory.py` |
| Composer/agents | `inference/composer/__init__.py` |
| Graph Builder | `inference/composer/graph/workflows/factory.py` |
| React app | `ui/src/main.tsx` |
| Routes | `ui/src/Router.tsx` |

### Configuration

- Python type checking: `pyrightconfig.json` (Python 3.12, basic mode, covers server/composer/runner)
- TypeScript: `ui/tsconfig.json` (ESNext, paths aliased `@/*` → `./src/*`, strict: false)
- Pytest: `inference/pytest.ini` (asyncio_mode: auto, testpaths: `test/unit tests`)
- ESLint: `ui/.eslintrc.cjs` (xo + TypeScript + React hooks, 2-space indent)

### Architectural Patterns

**Protocol-Based Decoupling**: Composer uses Python Protocols (PEP 544) to define interfaces for server services. The `ServerInterface` and related service protocols are defined in `composer/server/interface.py`. This enables:
- Type-safe service access without runtime dependencies
- Easy mocking for testing
- Clean separation between composer and server implementations

**Adapter Pattern**: The `ServerAdapter` in `composer/server/interface.py` wraps the singleton server instance and implements `ServerInterface`, allowing composer to access server services through the protocol interface.

**Workflow Caching**: Composer implements workflow caching in `composer/graph/cache.py` with TTL and LRU eviction. The cache key is computed from user_id, workflow_type, tools, and configuration to ensure safe reuse.

**Model Profile Pattern**: Model configurations are managed through `ModelProfile` objects and retrieved via `get_model_profile_for_task()` in `composer/utils/model_profile.py`. The `PROFILE_TYPE_TO_CONFIG_FIELD` mapping connects task types to configuration fields.
