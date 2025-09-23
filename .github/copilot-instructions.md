# LLM ML Lab - AI Agent Instructions

## Project Architecture

LLM ML Lab is a multi-modal language model platform with microservice architecture:

- **inference/**: Python services (evaluation, server, runner) with isolated virtual environments
     - **evaluation/**: Model benchmarking and fine-tuning tools
     - **server/**: FastAPI REST + gRPC services for model interaction (calls runner for execution)
     - **runner/**: Model execution pipelines with dynamic tool integration
- **ui/**: React TypeScript frontend with Material UI Joy
- **proto/**: Protocol buffer definitions for gRPC APIs
- **schemas/**: YAML schema definitions for type safety across services

## Key Development Workflows

### Environment Setup



```bash
# Kubernetes pod commands (production/staging)
POD_NAME=$(k get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
k exec -it -n ollama $POD_NAME -- /app/v.sh server python -m uvicorn app:app --port 8000
k exec -it -n ollama $POD_NAME -- /app/v.sh runner python -c "import torch; print(torch.cuda.is_available())"
```

inference does not generally run locally due to hardware needs. use `inferece/sync-code.sh` to sync code to remote cluster.
the ui is fully local and connects to remote inference services.


### Code Generation
```bash
# Generate Python and Typescript models from YAML schemas
./regenerate_models.sh
```

## Critical Patterns

### Multi-Environment Architecture
The inference service uses **three isolated Python environments**:
- `evaluation/`: Benchmarking and fine-tuning (separate deps from serving)
- `server/`: FastAPI REST + gRPC services 
- `runner/`: Model execution pipelines

Always use `/app/v.sh {service}` when executing commands in Kubernetes pods.

### Schema-Driven Development
YAML schemas in `schemas/` define the data contracts. When modifying APIs:
1. Update relevant YAML schema first
2. Run `./regenerate_models.sh` to generate Python models

### Memory Management
The platform implements sophisticated memory optimization:
- Models loaded on-demand and unloaded after use
- GPU memory tracking and automatic resource management
- Multiple memory optimization strategies based on available VRAM

## Service Communication

```
UI (React) ←→ REST API (FastAPI) ←→ gRPC (Internal) ←→ Model Runner
     ↓                ↓                    ↓
WebSocket        RabbitMQ           GPU Resources
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

## File Conventions

- **API endpoints**: REST in `server/`, gRPC services use protobuf contracts
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