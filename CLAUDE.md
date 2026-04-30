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

### Deployment

```bash
make deploy         # Deploy all services (inference, maistro, ui) to k8s
make clean          # Remove build artifacts (ui/build/)
```

## Architecture

### Structure

```
inference/      Python FastAPI inference service (deployed to k8s)
  app.py        FastAPI entry point
  routers/      API layer: openai/, anthropic/, common/ routers + middleware
  middleware/   Authentication, database init, message validation
  services/     Business logic: completion, token, tool services
  runner/       Model execution: pipeline_factory, pipeline_cache, pipelines/
  composer_init.py  Composer public API (workflow orchestration)
  agents/       Agent implementations (chat, embed)
  core/         Core composer components (service, errors)
  graph/        LangGraph workflow builder, executor, state, nodes
  tools/        Tool registry and static tools
  db/           PostgreSQL + Redis storage
  models/       Pydantic data models (edit directly)
  utils/        Shared helpers
  k8s/          Kubernetes manifests
ui/             React 19 + Vite frontend
```

### Key Architectural Patterns

**Models**: Python models live in `inference/models/*.py` (Pydantic) and are edited directly. TypeScript types in `ui/src/types/*.ts` can be generated from the FastAPI OpenAPI schema at `/openapi.json` using `openapi-typescript`.

**Pipeline System**: The runner uses a pluggable pipeline pattern. `pipeline_factory.py` creates appropriate pipelines (text, image, embeddings, multimodal). `pipeline_cache.py` manages instances. All pipeline implementations live in `runner/pipelines/`.

**Provider Compatibility**: The server implements both OpenAI-compatible endpoints (`routers/openai/`) and Anthropic-compatible endpoints (`routers/anthropic/`). These share the underlying runner/pipeline infrastructure.

**Streaming**: Chat and image responses stream token-by-token.

**Deployment**: The inference service runs in Kubernetes (`inference/k8s/`). `make inference-dev` syncs local code to the cluster via `inference/sync-code.sh` and tails logs.

### Key Entry Points

| Component | Entry Point |
|-----------|-------------|
| FastAPI app | `inference/app.py` |
| OpenAI chat endpoint | `inference/routers/openai/chat.py` |
| Anthropic messages endpoint | `inference/routers/anthropic/messages.py` |
| Pipeline creation | `inference/runner/pipeline_factory.py` |
| Composer/agents | `inference/composer_init.py` |
| React app | `ui/src/main.tsx` |
| Routes | `ui/src/Router.tsx` |

### Configuration

- Python type checking: `pyrightconfig.json` (Python 3.12, basic mode, covers routers/composer/runner)
- TypeScript: `ui/tsconfig.json` (ESNext, paths aliased `@/*` → `./src/*`, strict: false)
- Pytest: `inference/pytest.ini` (asyncio_mode: auto, testpaths: `test/unit tests`)
- ESLint: `ui/.eslintrc.cjs` (xo + TypeScript + React hooks, 2-space indent)
