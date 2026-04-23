# LLM ML Lab – AI Coding Agent Guide
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

### Deployment

```bash
make deploy         # Deploy all services (inference, maistro, ui) to k8s
make clean          # Remove build artifacts (ui/build/)
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
  models/       Pydantic data models (edit directly)
  debug/        Debug utilities and output files
ui/             React 19 + Vite frontend
```

### Key Architectural Patterns

**Models**: Python models live in `inference/models/*.py` (Pydantic) and are edited directly. TypeScript types in `ui/src/types/*.ts` can be generated from the FastAPI OpenAPI schema at `/openapi.json` using `openapi-typescript`.

**Pipeline System**: The runner uses a pluggable pipeline pattern. `pipeline_factory.py` creates appropriate pipelines (text, image, embeddings, multimodal). `pipeline_cache.py` manages instances. All pipeline implementations live in `runner/pipelines/`.

**Multi-Tier Caching**: User config flows in-memory → Redis → PostgreSQL. See `docs/multi_tier_user_config_caching.md`.

**Provider Compatibility**: The server implements both OpenAI-compatible endpoints (`routers/openai/`) and Anthropic-compatible endpoints (`routers/anthropic/`). These share the underlying runner/pipeline infrastructure.

**Streaming**: Chat and image responses stream token-by-token. The streaming architecture is documented in `docs/RUNNER_ARCHITECTURE_OVERHAUL.md`.

**Deployment**: The inference service runs in Kubernetes (`inference/k8s/`). `make inference-dev` syncs local code to the cluster via `inference/sync-code.sh` and tails logs.

### Key Entry Points

| Component | Entry Point |
|-----------|-------------|
| FastAPI app | `inference/server/app.py` |
| OpenAI chat endpoint | `inference/server/routers/openai/chat.py` |
| Anthropic messages endpoint | `inference/server/routers/anthropic/messages.py` |
| Pipeline creation | `inference/runner/pipeline_factory.py` |
| Composer/agents | `inference/composer/__init__.py` |
| React app | `ui/src/main.tsx` |
| Routes | `ui/src/Router.tsx` |

### Configuration

- Python type checking: `pyrightconfig.json` (Python 3.12, basic mode, covers server/composer/runner)
- TypeScript: `ui/tsconfig.json` (ESNext, paths aliased `@/*` → `./src/*`, strict: false)
- Pytest: `inference/pytest.ini` (asyncio_mode: auto, testpaths: `test/unit tests`)
- ESLint: `ui/.eslintrc.cjs` (xo + TypeScript + React hooks, 2-space indent)
