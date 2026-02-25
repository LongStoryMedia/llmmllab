## Inference Python App – Architecture Overview

> **Goal:** Provide a clear, self‑contained explanation of the modules, dependencies, and workflow that power the `inference` Python application.

> **File:** `/home/lsm/Nextcloud/llmmllab/inference/inference-architecture.md`

---

### 1. Project Scope

| Area | Purpose | Key Components |
|------|---------|----------------|
| **Composer** | Orchestrates workflow state machines and tool routing | `composer/graph/` |
| **Runner** | Provides stateless LLM interface and streaming | `runner/` |
| **Server** | FastAPI + gRPC endpoint | `server/` |
| **DB** | Persistence layer | `db/` |
| **Utils** | Shared utilities | `utils/` |
| **Test** | Unit & integration tests | `test/` |
| **Docs** | Architecture documentation | `docs/` |
| **Schemas** | Data model definitions | `schemas/` |
| **UI** | React/TS frontend | `ui/` |

### 2. Directory Structure

```
/home/lsm/Nextcloud/llmmllab/inference/
├── activate_env.sh
├── copilot-out.json
├── Dockerfile
├── download_models.sh
├── generate_routers.py
├── gguf_dump.py
├── ollama-compat.md
├── pytest.ini
├── README.md
├── requirements.txt
├── routing_test_output.txt
├── run_tests.sh
├── run_with_env.sh
├── run.sh
├── setup_cuda_runtime.sh
├── setup_environments.sh
├── setup_memory_optimization.sh
├── startup.sh
├── sync-code.sh
└── v.sh

├── benchmark_data/
│   ├── composer/
│   ├── db/
│   ├── debug/
│   ├── docs/
│   ├── evaluation/
│   ├── k8s/
│   ├── models/
│   ├── oom_recovery_data/
│   ├── runner/
│   ├── server/
│   ├── test/
│   └── utils/

├── schemas/
│   ├── analysis_depth.yaml
│   ├── api_key_request.yaml
│   ├── api_key_response.yaml
│   ├── api_key.yaml
│   ├── auth_config.yaml
│   ├── capability_profile_mapping.yaml
│   ├── chat_req.yaml
│   ├── chat_response.yaml
│   ├── circuit_breaker_config.yaml
│   └── ...

└── ui/
```

### 3. Core Modules

| Module | Description | Entry Point |
|--------|-------------|-------------|
| **Composer** | Orchestrates LangGraph workflow states | `composer.graph.GraphComposer` |
| **Runner** | Consumes LLM output and streams to UI | `runner.llm_runner` |
| **Server** | REST/gRPC endpoints | `server.main` |
| **DB** | ORM & query helpers | `db.models` |
| **Utils** | Common helpers (logging, config, metrics) | `utils.helpers` |
| **Test** | pytest suite | `test.integration` |
| **Docs** | Architecture docs | `docs/` |
| **UI** | Frontend (React/TS) | `ui/src/App.tsx` |

### 4. Workflow Diagram (Textual)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Composer   │───▶│  Runner     │───▶│  Server     │───▶│  DB         │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

- **LangGraph orchestration** (`composer.graph.GraphComposer`) creates and manages workflow states.
- **LLM interface** (`runner.llm_runner`) uses `langchain` to generate and stream responses.
- **API layer** (`server.main`) exposes endpoints to external services and UI.
- **Database** (`db.models`) stores results and maintains persistence.
- **Shared utilities** (`utils.helpers`) provide common functions (logging, config, metrics).
- **Documentation** (`docs/`) keeps the architecture diagrams and reference docs.
- **UI** (`ui/src/App.tsx`) consumes the API and renders UI.

### 5. Dependencies

- **Python 3.11** (managed via `venv` or `conda`)
- **FastAPI** for REST API
- **gRPC** for streaming
- **LangGraph** for orchestration
- **SQLAlchemy** for ORM
- **Pytest** for testing
- **Jupyter** for notebook notebooks
- **VS Code** for development

> **Installation**
> 1. Create virtual environment: `python -m venv .venv`
> 2. Activate: `source .venv/bin/activate`
> 3. Install dependencies: `pip install -r requirements.txt`
> 4. Run the app: `python -m server.main`

### 6. Example Usage

```bash
# Run the server
python -m server.main

# Run the test suite
pytest test/

# Build the UI
npm run build
```

### 7. Testing & Validation

| Test | Description | Command |
|------|-------------|---------|
| **Unit** | Test individual modules | `pytest test/unit` |
| **Integration** | Test end‑to‑end flow | `pytest test/integration` |
| **E2E** | Full composer‑runner‑server‑db flow | `pytest test/e2e` |

> **Note:** All tests must run in the Kubernetes pod: `kubectl exec -it -n llmmll <POD_NAME> -- /app/v.sh python -m debug.<test_name>`

### 8. Key Architectural Patterns

| Pattern | Implementation | Why it matters |
|---------|------------------|----------------|
| **Stateful orchestration** | LangGraph state machines | Enables dynamic workflow control |
| **Stateless runner** | LLM interface with streaming | Simplifies scaling and deployment |
| **API gateway** | FastAPI + gRPC | Unified interface for services |
| **Idempotent DB** | SQLAlchemy + migrations | Reliable persistence and rollback |
| **Type‑first** | YAML schemas → models | Ensures consistent data contracts |
| **CI/CD** | GitHub Actions + Kubernetes | Continuous integration & deployment |

---

> **Author:** GitHub Copilot
> **Date:** 2026-02-25