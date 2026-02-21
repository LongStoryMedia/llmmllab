# LLM ML Lab – AI Coding Agent Guide

Focus: Execute precisely against current architecture. No speculation.

## Core Principles
1. Verify code before modifying – read surrounding files first.
2. Prefer removal/simplification over added complexity.
3. Never leave unused experimental code; clean as you go.
4. Keep Kubernetes pod healthy – fix crash loops immediately.
5. Use short, single-purpose commands (avoid long chained subshells).
6. Strong typing over reflection (`getattr`/`hasattr` avoided).
7. Always commit + sync after meaningful changes.

## High-Level Architecture
- `inference/` container houses: `composer/` (LangGraph orchestration), `runner/` (pure LLM interface & streaming), `server/` (FastAPI + gRPC), `evaluation/` (benchmarking), `db/` provides storage services; register new storage in `init_db.py` & `db/__init__.py`, `utils/` shared utilities, `test/` unit and integration tests.
- `inference/composer/graph/` builds workflow state machines (see `subgraphs/tools_agent.py`, `summarization_middleware.py`). Composer owns all orchestration; runner must stay stateless regarding workflows.
- `schemas/` YAML → generated models in `inference/models/` and TS types in `ui/src/types/` via `./regenerate_models.sh` (existing schemas) or `schema2code` for single new model.
- `ui/` React TS (MUI) consumes OpenAI-compatible and custom endpoints from server.

## Workflow & Agents Pattern
- Agents inherit `BaseAgent` for metadata injection, logging, error handling (see `docs/base_agent_architecture.md`).
- Subgraphs: build `StateGraph(WorkflowState)` with nodes; tool routing uses conditional edges (example: `should_continue_tool_calls` in `tools_agent.py`).
- Middleware (e.g. `SummarizationMiddleware`) modifies message lists before model invocation using token thresholds; respect its patterns (ID assignment, safe cutoff logic, tool pair preservation).
- Context assembly occurs via `assemble_context_messages()` (see `docs/context_assembly_usage.md`) combining summaries, memories, search results, then recent messages.

## Database & Schema Rules
- All SQL idempotent; use parameter placeholders `$1`, `$2` etc. Never string format SQL.
- New entity flow: schema → `./regenerate_models.sh` → SQL in `db/sql/<entity>/` → storage service (`<entity>_storage.py`) using `typed_pool`, `get_query` → register.
- New standalone model (no full regen): `schema2code --language python --output inference/models/<name>.py schemas/<name>.yaml`.

## Execution & Environment
- **CRITICAL**: Debug files (`debug/`) MUST be run in Kubernetes pod, never locally
- Pod execution: `kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.<test_name>`
- Local env (non-debug only): `source inference/.venv/bin/activate`
- Always run modules with `python -m <module>` (avoid direct file paths) for import correctness.
- Sync code: `inference/sync-code.sh` (retry once if fails).

## Testing & Validation
- **CRITICAL**: All debug files must run in pod: `kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.<test_name>`
- Unit: `cd inference && pytest test/` for pure logic changes (local OK).
- Full E2E: `kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.e2e` (composer + runner + db).
- Tools agent focus: `kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.tools_agent`.
- Memory E2E: `kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.memory_e2e`.
- A change is incomplete if: lint/import errors, hardcoded paths, failing pod, or architectural pattern violations.

## UI Conventions
- Lint & typecheck: `cd ui && npm run lint && npm run typecheck`.
- Use generated types in `ui/src/types/` for API models; avoid manual duplication.

## Configuration
- System settings: `schemas/config.yaml`. User prefs: `schemas/user_config.yaml`.
- Add new env var: update schema → regenerate models → k8s deployment → runtime validation.

## Safe Change Checklist
1. Read target file + related doc in `docs/`.
2. Confirm pattern match (agent inheritance, context assembly, storage registration, etc.).
3. Implement minimal diff; avoid unrelated refactors.
4. Add/adjust tests only for changed logic.
5. Run appropriate test command.
6. Commit descriptive message; run `inference/sync-code.sh`.

## Anti-Patterns (Avoid)
- Long chained kubectl commands; break into two steps.
- Direct file execution (`python path/file.py`).
- Hardcoded absolute import path modifications.
- Leaving experimental commented blocks or dead code.
- SQL with string interpolation.

## Quick Command Examples
```bash
# Pod name lookup
kubectl get pods -n llmmll -o jsonpath='{.items[0].metadata.name}'

# Validate config load
kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -c "from composer.config import config; print('CONFIG_OK')"

# Run summarization middleware test (example)
kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.test_composer_real_e2e

# Run memory E2E test
kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.memory_e2e
```

## When Adding Middleware or Subgraphs
- Preserve message ID handling (`uuid` assignment if missing).
- Maintain separation of AI/Tool message pairs; use similar safe cutoff logic if trimming.
- Return updated state, not raw model outputs; append `response.message` only if present.

## Final Reminder
No guesswork. Every modification must align with existing documented patterns and minimal diff philosophy.

---
Feedback welcome: highlight unclear sections or missing workflows.
