# LLM ML Lab

A comprehensive platform for language model inference, evaluation, and deployment with multi-modal capabilities.

## Overview

LLM ML Lab is a full-featured platform for deploying, serving, and evaluating large language models. The platform consists of multiple components that work together to provide a complete solution for language model infrastructure:

1. **Server** - Python-based gRPC/HTTP service for orchestration and API routing
2. **Composer** - Python-based LangGraph agent orchestration and tool generation
3. **Runner** - Python-based model execution service (GPU-enabled, runs on lsnode-3)
4. **UI** - React-based user interface for interacting with the services

## Project Structure

```text
/llmmllab
├── server/                   # Python-based orchestration service (gRPC/HTTP)
│   └── k8s/                  # Kubernetes manifests and build scripts
├── composer/                 # Python-based LangGraph agent orchestration
│   └── k8s/                  # Kubernetes manifests and build scripts
├── runner/                   # Python-based model execution (GPU-enabled)
│   └── k8s/                  # Kubernetes manifests and build scripts
├── inference/                # Legacy Python inference service (deprecated)
├── ui/                       # React-based frontend
│   ├── public/               # Static assets
│   └── src/                  # React components and application logic
├── proto/                    # Protocol buffer definitions
├── docs/                     # Documentation
├── schemas/                  # Common schema definitions
├── build.sh                  # Code generation script
├── regenerate_models.sh      # Model regeneration script
├── build-image.sh            # Multi-arch build helper
└── Makefile                  # Build and deployment commands
```

## Key Features

- **Multi-Modal Support**: Text generation, image generation, and embeddings
- **Multiple API Interfaces**: OpenAI-compatible REST endpoints, gRPC for internal communication
- **Model Management**: Add, configure, and switch between models
- **Memory Optimization**: Automatic memory management and resource allocation
- **Performance Monitoring**: Logging and metrics collection
- **Session Management**: User sessions and conversation context
- **Scalable Architecture**: Components can be deployed independently
- **WebSocket Support**: Real-time communication for chat and status updates
- **RabbitMQ Integration**: Message queuing for asynchronous processing
- **Context Extension**: Sophisticated system to extend LLM context windows
- **Schema Validation**: YAML schemas for type-safety and consistency

## Architecture

The system follows a microservice architecture where components communicate through well-defined APIs:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     UI      │────>│   Server    │────>│   Composer    │
└─────────────┘     └─────────────┘     └─────────────┘
                                            │
                                            ▼
                                    ┌─────────────┐
                                    │    Runner     │
                                    │   (GPU)       │
                                    └─────────────┘
```

- **Server** - Entry point for all API requests, routes to appropriate services
- **Composer** - LangGraph-based agent orchestration, tool calling, intent analysis
- **Runner** - GPU-accelerated model execution (llama.cpp, Flux, Qwen3-VL)
- **WebSockets** - Real-time communication for streaming responses
- **RabbitMQ** - Message queuing for async tasks
- **PostgreSQL** - Persistent storage for users, conversations, configurations
- **Redis** - Caching layer for improved performance

## Configuration Architecture

The platform uses a hierarchical configuration system that separates system administration from user preferences:

- **System Configuration**: Infrastructure settings (ports, databases, logging) managed by operators
- **User Configuration**: Workflow and tool preferences customizable per user via UI
- **Schema-Driven**: YAML schemas automatically generate Python models and TypeScript types

Key configuration areas:

- **Workflow Settings**: Caching, streaming, timeouts, multi-agent capabilities
- **Tool Management**: Selection thresholds, generation preferences, execution settings
- **Memory & Context**: Retrieval settings, circuit breakers, model profiles

See [Configuration Architecture](docs/configuration_architecture.md) for detailed documentation.

## Component Documentation

Each component has its own detailed README with specific instructions:

- [Server](server/README.md) - Orchestration service and API routing
- [Composer](composer/README.md) - LangGraph agent orchestration and tool generation
- [Runner](runner/README.md) - Model execution and pipeline management
- [UI Application](ui/README.md) - User interface for interacting with the services
- [YAML Schemas](schemas/README.md) - Data structure definitions
- [Context Extension Architecture](docs/context_extension.md) - LLM context window extension system
- [Dynamic Tool Generation](inference/server/tools/README.md) - Tool generation for model execution
- [Configuration Architecture](docs/configuration_architecture.md) - Hierarchical configuration system
- [Multi-Tier User Config Caching](docs/multi_tier_user_config_caching.md) - In-memory → Redis → Database caching system

## Pipeline Documentation

The inference runner module includes comprehensive pipeline support for all model types. For developers building custom pipelines or working with existing ones:

- [**Pipeline Documentation Overview**](docs/PIPELINE_DOCUMENTATION_OVERVIEW.md) - Complete guide to all available pipeline documentation
- [**Pipeline Implementation Guide**](docs/PIPELINE_IMPLEMENTATION_GUIDE.md) - Comprehensive step-by-step guide for implementing custom pipelines
- [**Pipeline API Reference**](docs/PIPELINE_API_REFERENCE.md) - Complete API documentation for all pipeline interfaces
- [**Runner Architecture Overhaul**](docs/RUNNER_ARCHITECTURE_OVERHAUL.md) - Recent improvements including streaming architecture and pipeline-specific processing

The pipeline system supports all major AI workflows including text generation, embeddings, image generation, and multimodal interactions with advanced features like circuit breakers, memory optimization, and real-time streaming.

## Getting Started

### Prerequisites

- Python 3.12+ (for server, composer, runner)
- Node.js 18+ (for UI)
- Docker and Docker Compose (for local development)
- Kubernetes cluster (for production deployment)
- CUDA-compatible GPU (for runner - lsnode-3 only)

### Quick Start (Local Development)

For local development without Kubernetes:

```bash
# Clone the repository
git clone https://github.com/LongStoryMedia/llmmllab.git
cd llmmllab

# Install UI dependencies
cd ui && npm install && cd ..

# Start all services
make start
```

### Development Mode

Run individual components locally:

```bash
# Start server locally (without k8s)
make dev-server

# Start composer locally
make dev-composer

# Start runner locally
make dev-runner

# Start UI
make start-ui
```

### Code Generation

The platform uses YAML schemas to define data contracts and automatically generate Python models and TypeScript types.

```bash
# Generate all models (Python + TypeScript)
make gen

# Generate only Python models
make gen-python

# Generate only TypeScript types
make gen-typescript
```

### Kubernetes Deployment

For production deployment on Kubernetes:

```bash
# Deploy all services
make deploy

# Deploy individual services
make deploy-server      # Server service (multi-arch)
make deploy-composer    # Composer service (multi-arch)
make deploy-runner      # Runner service (GPU, lsnode-3)

# Deploy legacy inference service
make inference

# Deploy UI
make ui
```

## Makefile Commands

### Development Servers
| Command | Description |
|---------|-------------|
| `make start` | Start all development servers (inference + UI) |
| `make start-inference` | Start inference service in dev mode (syncs to k8s) |
| `make start-ui` | Start UI development server |
| `make start-maistro` | Start maistro service |

### Code Generation
| Command | Description |
|---------|-------------|
| `make gen` | Generate all models (Python + TypeScript) |
| `make gen-python` | Generate Python models only |
| `make gen-typescript` | Generate TypeScript types only |

### Kubernetes Deployment
| Command | Description |
|---------|-------------|
| `make deploy` | Deploy all services (server, composer, runner) |
| `make deploy-server` | Deploy server service (multi-arch) |
| `make deploy-composer` | Deploy composer service (multi-arch) |
| `make deploy-runner` | Deploy runner service (GPU-enabled, lsnode-3) |

### Local Development
| Command | Description |
|---------|-------------|
| `make dev-server` | Run server locally (without k8s) |
| `make dev-composer` | Run composer locally (without k8s) |
| `make dev-runner` | Run runner locally (without k8s) |

### Validation & Testing
| Command | Description |
|---------|-------------|
| `make validate` | Run TypeScript and Python validation |
| `make test` | Run all tests (inference + UI) |
| `make test-inference` | Run inference tests only |
| `make test-ui` | Run UI tests only |

### Cleanup
| Command | Description |
|---------|-------------|
| `make clean` | Remove build artifacts |
| `make clean-py` | Remove Python cache files |

### Submodule Management
| Command | Description |
|---------|-------------|
| `make sync-submodules` | Sync all submodules and push changes |
| `make push-all` | Sync submodules and push all changes |

### Help
| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |

## Development Workflow

### Schema-Driven Development

The platform uses YAML schemas to define data contracts and automatically generate Python models and TypeScript types.

#### Creating New Schemas

1. Create new YAML schema in `schemas/[name].yaml`
2. Generate code: `make gen` or `make gen-all`
3. Use the generated models in your code

#### Schema Development Workflow

When modifying APIs or data structures:

1. Update the relevant YAML schema first
2. Run generation commands to update models
3. Test the changes with the generated types
4. Generated files: `inference/models/*.py`, `ui/src/types/*.ts`

#### Schema Design Rules

- **Avoid Duplication**: If an enum or structure is used in multiple schemas, extract it to a separate schema file
- **Use $ref**: Reference shared schemas using `$ref: "shared_schema.yaml"` instead of copying definitions
- **Single Source of Truth**: Each data structure should be defined exactly once

### Multi-Platform Build

The build system uses Docker Buildx for multi-arch images:

- **Server & Composer**: Build for both `linux/amd64` and `linux/arm64`
- **Runner**: Build directly on lsnode-3 (AMD64) for GPU compatibility

## Release Notes

Version history and release notes are maintained in [docs/releases/](docs/releases/). See the [CHANGELOG](docs/releases/CHANGELOG.md) for a detailed history of changes across versions.

## License

[MIT License](LICENSE)