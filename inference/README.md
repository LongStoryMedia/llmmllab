# LLM ML Lab Server

This project provides API servers for language model inference, with both OpenAI-compatible REST API and gRPC interfaces.

## Overview

The LLM ML Lab Server project provides:

- OpenAI-compatible REST API endpoints
- gRPC services for efficient communication
- Service layer for handling model operations

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the REST API Server

```bash
# Start the FastAPI server
python -m app
```

### Starting the gRPC Server

```bash
# Start the gRPC server
python -m grpc_server
```

### Configuration

Create a `.env` file with the following settings:

```
MODEL_CONFIG_PATH=/path/to/models.json
PORT=8000
GRPC_PORT=50051
```

## Project Structure

- `app.py`: FastAPI application entry point
- `routers/`: FastAPI route handlers (openai/, anthropic/, common/)
- `middleware/`: Authentication, database init, message validation
- `services/`: Business logic (completion, token, tool)
- `runner/`: Model execution (pipelines, pipeline factory/cache)
- `composer_init.py`: Composer public API (workflow orchestration)
- `agents/`: Agent implementations (chat, embed)
- `core/`: Core composer components (service, errors)
- `graph/`: LangGraph workflow builder, executor, state, nodes
- `tools/`: Tool registry and static tools
- `db/`: Multi-tier storage (memory → Redis → Postgres)
- `models/`: Pydantic data models
- `utils/`: Shared helpers (logging, message conversion)
- `k8s/`: Kubernetes deployment manifests
- `test/`: Unit and integration tests

## API Documentation

When running the server, visit:
- REST API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## License

[License information]
