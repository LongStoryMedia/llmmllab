# llmmllab Proto

This repository contains the Protocol Buffer definitions for the llmmllab microservices architecture.

## Overview

The proto repository defines gRPC service interfaces for communication between:

- **Server**: FastAPI inference service with HTTP endpoints
- **Composer**: LangGraph agent orchestration system
- **Runner**: Model execution pipeline system

## Directory Structure

```
proto/
├── common/           # Common protobuf types (timestamp, etc.)
├── server/           # Server service definitions
├── composer/         # Composer service definitions
├── runner/           # Runner service definitions
└── models/           # Data model definitions (shared across services)
```

## Services

### ServerService
- HTTP endpoint orchestration
- Authentication and authorization
- Database operations
- Model management

### ComposerService
- Workflow composition
- Workflow execution with streaming
- Agent orchestration
- Tool calling

### RunnerService
- Pipeline creation and management
- Model execution
- Embedding generation
- Cache management

## Usage

### Generating Code

```bash
# Generate Python code
python -m grpc_tools.protoc -I. --python_out=../server/gen --grpc_python_out=../server/gen server/v1/server.proto

# Generate TypeScript code (with grpc_tools)
npx -y @grpc/grpc-js --proto_path=. --ts_out=../ui/src/grpc server/v1/server.proto
```

### Adding New Services

1. Create a new `.proto` file in the appropriate directory
2. Define your service and messages
3. Generate code for each consumer service
4. Update this README with the new service documentation

## Versioning

Protos use semantic versioning:
- `v1/` - Stable API
- `v1alpha/` - Alpha (experimental)
- `v1beta/` - Beta (stabilizing)

## License

MIT