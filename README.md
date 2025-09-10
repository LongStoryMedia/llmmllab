# LLM ML Lab

A comprehensive platform for language model inference, evaluation, and deployment with multi-modal capabilities.

## Overview

LLM ML Lab is a full-featured platform for deploying, serving, and evaluating large language models. The platform consists of multiple components that work together to provide a complete solution for language model infrastructure:

1. **Inference Service** - Python-based service for model execution and API endpoints
2. **UI** - React-based user interface for interacting with the services

## Project Structure

```text
/llmmllab
├── inference/                # Python-based inference services
│   ├── evaluation/           # Model benchmarking and evaluation tools
│   ├── server/               # REST and gRPC API services
│   └── runner/               # Model execution and pipeline management
├── ui/                       # React-based frontend
│   ├── public/               # Static assets
│   └── src/                  # React components and application logic
├── proto/                    # Protocol buffer definitions
├── docs/                     # Documentation
└── schemas/                  # Common schema definitions
```

## Key Features

- **Multi-Modal Support**: Text generation, image generation, and embeddings
- **Multiple API Interfaces**: REST and gRPC endpoints
- **Model Management**: Add, configure, and switch between models
- **Memory Optimization**: Automatic memory management and resource allocation
- **Performance Monitoring**: Logging and metrics collection
- **Session Management**: User sessions and conversation context
- **Scalable Architecture**: Components can be deployed independently
- **WebSocket Support**: Real-time communication for chat and status updates
- **RabbitMQ Integration**: Message queuing for asynchronous processing
- **Context Extension**: Sophisticated system to extend LLM context windows
- **Schema Validation**: YAML schemas for type-safety and consistency

## Component Documentation

Each component has its own detailed README with specific instructions:

- [Inference Services](inference/README.md) - API services and model execution
- [UI Application](ui/README.md) - User interface for interacting with the services
- [YAML Schemas](schemas/README.md) - Data structure definitions
- [Context Extension Architecture](docs/context_extension.md) - LLM context window extension system
- [Dynamic Tool Generation](inference/server/tools/README.md) - Tool generation for model execution

## Getting Started

### Prerequisites

- Python 3.12+ (for inference services)
- Node.js 18+ (for UI)
- Docker and Docker Compose (optional for containerized deployment)
- CUDA-compatible GPU (recommended for performance)

### Quick Start with Docker Compose

The simplest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/LongStoryMedia/llmmllab.git
cd llmmllab

# Start all services
docker-compose up -d
```

This will start all the necessary services and make them available on their respective ports.

### Manual Setup

For development or custom deployments, you can set up each component separately:

1. **Set up inference services**:

   ```bash
   cd inference
   ./setup_environments.sh
   ```

2. **Set up UI application**:

   ```bash
   cd ui
   npm install
   npm run dev
   ```

See the individual component READMEs for more detailed instructions.

## Architecture

The system follows a microservice architecture where components communicate through well-defined APIs:

- The **UI** makes direct requests to the **Inference Services**
- **WebSockets** provide real-time communication for chat, image generation, and status updates
- **RabbitMQ** handles asynchronous processing for computationally intensive tasks
- **PostgreSQL** provides persistent storage for user data, conversations, and configurations

### RabbitMQ Integration

The platform uses RabbitMQ as a message broker for:

- **Task Queuing**: Managing computationally intensive tasks like image generation
- **Load Balancing**: Distributing tasks across multiple worker instances
- **Priority Processing**: Handling high-priority requests ahead of others
- **Failure Recovery**: Ensuring tasks are not lost if a worker fails

Configuration is defined in `schemas/rabbitmq_config.yaml`.

### WebSocket Communication

Real-time communication is handled through WebSocket connections for:

- **Chat Streaming**: Streaming token-by-token responses for chat completions
- **Image Generation Status**: Real-time updates on image generation progress
- **System Status**: Updates on model loading, resource availability, and errors

WebSocket schemas are defined in `schemas/web_socket_connection.yaml` and related files.

### Context Extension System

The platform includes a sophisticated Context Extension System (documented in [context_extension.md](docs/context_extension.md)) that:

- **Extends LLM Context Windows**: Overcomes token limitations of models
- **Semantic Memory**: Retrieves relevant conversation history
- **External Search**: Incorporates real-time web knowledge
- **Hierarchical Summarization**: Compresses conversation context intelligently

## License

[MIT License](LICENSE)
