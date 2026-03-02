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
    routers/
      openai/   Auto-generated OpenAI-compatible endpoints (23 routers)
      anthropic/ Anthropic-compatible endpoints (messages, completions)
      common/   Shared endpoints (models, files)
    middleware/ Authentication, database init, message validation
  runner/       Model execution: pipeline_factory, pipeline_cache, pipelines/
    pipelines/
      llamacpp/ Llama.cpp chat, embedding pipelines
      txt2img/    Text-to-image pipelines (Flux, Stable Diffusion)
      img2img/    Image-to-image pipelines
      external/   External service integrations
  composer/     LangGraph agent orchestration, tool generation
    graph/
      workflows/ IDE (default), Dialog, and custom workflow types
      nodes/      LangGraph nodes for intent analysis, tool calling
      cache.py    Per-user workflow caching
      executor.py Streaming workflow execution
    agents/       BaseAgent with specialized agents (chat, embed, engineering)
    tools/        Dynamic tool generation for model execution
  db/           Multi-tier storage: PostgreSQL, Redis, in-memory
    multi_tier_cache.py  User config caching (memory → Redis → DB)
  models/       Generated from YAML schemas (do not edit directly)
  debug/        Debug utilities and output files
ui/             React 19 + Vite frontend
schemas/        YAML schema definitions (source of truth for models)
```

### Key Architectural Patterns

**Schema-Driven Development**: All data contracts are defined as YAML schemas in `schemas/`. The `schema2code` tool generates `inference/models/*.py` and `ui/src/types/*.ts` from these. **Never edit generated model files directly** — edit the YAML schema and regenerate.

**Pipeline System**: The runner uses a pluggable pipeline pattern. `pipeline_factory.py` creates appropriate pipelines (text, image, embeddings, multimodal). `pipeline_cache.py` manages instances with intelligent memory-based eviction. All pipeline implementations live in `runner/pipelines/`.

**Multi-Tier Caching**: User config flows in-memory → Redis → PostgreSQL. See `docs/multi_tier_user_config_caching.md`. The cache has 5-minute TTL for memory, 30 minutes for Redis, and is permanent in PostgreSQL.

**Provider Compatibility**: The server implements both OpenAI-compatible endpoints (`routers/openai/`) and Anthropic-compatible endpoints (`routers/anthropic/`). These share the underlying runner/pipeline infrastructure.

**Streaming**: Chat and image responses stream token-by-token. The streaming architecture is documented in `docs/PIPELINE_DOCUMENTATION_OVERVIEW.md`.

**Composer/LangGraph**: The `composer/` component implements LangGraph-based workflows with:
- Workflow caching (per-user, 1-hour TTL by default)
- Tool calling with dynamic tool generation
- Intent analysis for intelligent routing
- Multi-agent orchestration with `BaseAgent` base class
- Streaming execution via `execute_workflow()`

**Deployment**: The inference service runs in Kubernetes (`inference/k8s/`). `make inference-dev` syncs local code to the cluster via `inference/sync-code.sh` and tails logs.

### Key Entry Points

| Component | Entry Point |
|-----------|-------------|
| FastAPI app | `inference/server/app.py` |
| OpenAI chat endpoint | `inference/server/routers/openai/chat.py` |
| Anthropic messages endpoint | `inference/server/routers/anthropic/messages.py` |
| Pipeline creation | `inference/runner/pipeline_factory.py` |
| Pipeline caching | `inference/runner/pipeline_cache.py` |
| Composer service | `inference/composer/__init__.py` |
| Composer workflow builder | `inference/composer/graph/workflows/ide/builder.py` |
| React app | `ui/src/main.tsx` |
| Routes | `ui/src/Router.tsx` |

### Configuration

- Python type checking: `pyrightconfig.json` (Python 3.12, basic mode, covers server/composer/runner)
- TypeScript: `ui/tsconfig.json` (ESNext, paths aliased `@/*` → `./src/*`, strict: false)
- ESLint: `ui/.eslintrc.cjs` (xo + TypeScript + React hooks, 2-space indent)

### Environment Variables (Server)

Key environment variables for the inference server:

- `DB_CONNECTION_STRING`: PostgreSQL connection string
- `REDIS_HOST`, `REDIS_PORT`: Redis configuration
- `AUTH_JWKS_URI`: JWT authentication JWKS endpoint
- `HF_HOME`: Hugging Face cache directory
- `LOG_LEVEL`: Logging level (trace, debug, info, warning, error)

### Pipeline Types

| Task Type | Pipeline | Description |
|-----------|----------|-------------|
| TextToText | ChatLlamaCppPipeline | Standard LLM chat with llama.cpp |
| VisionTextToText | ChatLlamaCppPipeline | Multimodal models (Qwen2.5-VL, GLM-4V) |
| TextToEmbeddings | EmbedLlamaCppPipeline | Text embedding generation |
| TextToImage | FluxPipe | Text-to-image generation (Flux, Stable Diffusion) |
| ImageToImage | FluxKontextPipe | Image-to-image transformation |

### Workflow Types

- `WorkFlowType.IDE`: Interactive Development Environment (default) - full tool calling, intent analysis
- `WorkFlowType.Dialog`: Simplified dialog workflow for conversational interactions