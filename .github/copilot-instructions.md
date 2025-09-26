# LLM ML Lab - AI Agent Instructions

## Project Architecture

LLM ML Lab is a multi-modal language model platform with microservice architecture:

- **inference/**: Python services (evaluation, server, runner) with isolated virtual environments
     - **evaluation/**: Model benchmarking and fine-tuning tools
     - **server/**: FastAPI REST services for model interaction (calls runner for execution)
     - **runner/**: Model execution pipelines with dynamic tool integration
- **ui/**: React TypeScript frontend with Material UI Joy
- **schemas/**: YAML schema definitions for type safety across services (generates code via `./regenerate_models.sh`)

## Key Development Workflows

### Environment Setup



```bash
# Kubernetes pod commands (production/staging)
POD_NAME=$(k get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
k exec -it -n ollama $POD_NAME -- /app/v.sh server python -m uvicorn app:app --port 8000
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
The inference service uses **three isolated Python environments**:
- `evaluation/`: Benchmarking and fine-tuning (separate deps from serving)
- `server/`: FastAPI REST services 
- `runner/`: Model execution pipelines

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
kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh server python test_real_end_to_end_pipeline.py qwen3-30b-a3b-q4-k-m
```

DO NOT ADD DOCUMENTATION FOR FIXES. ONLY DOCUMENT FULLY IMPLEMENTED FEATURES, AND ALWAYS IN THE `docs/` FOLDER. ALWAYS LINK TO THE DOCS FROM THE README IF IT'S IMPORTANT.