# gRPC-Based Inference API

This project implements a gRPC-based architecture for communication between a Python inference service and a Go maistro service.

## Architecture Overview

The system consists of two main components:

1. **Python Inference Service**: Implements a gRPC server that provides procedures for:
   - Chat generation with streaming responses
   - Text generation with streaming responses
   - Embedding generation
   - Image generation and editing
   - VRAM memory management with bidirectional streaming

2. **Go Maistro Service**: Implements a gRPC client that communicates with the inference service.

## Getting Started

### Prerequisites

- Python 3.8+
- Go 1.18+
- Protocol Buffers compiler (protoc)
- Docker and Docker Compose (optional)

### Installing Dependencies

For Python:
```bash
pip install -r inference/requirements.txt
pip install grpcio grpcio-tools grpcio-reflection
```

For Go:
```bash
cd maistro
go mod download
```

### Generating Proto Code

Run the following command to generate both Python and Go code from the protobuf definitions:

```bash
./generate_proto.sh
```

This will generate:
- Python code in `inference/grpc_server/proto/`
- Go code in `maistro/grpc/inference/`

### Running the Services

#### Option 1: Running Directly

Run the Python gRPC server:
```bash
cd inference
python grpc_server/server.py
```

Run the Go service:
```bash
cd maistro
go run main.go
```

#### Option 2: Using Docker Compose

```bash
docker-compose -f docker-compose.grpc.yml up
```

## Configuration

### Python gRPC Server

The gRPC server can be configured using environment variables:

- `GRPC_PORT`: Port to listen on (default: 50051)
- `GRPC_MAX_WORKERS`: Maximum number of worker threads (default: 10)
- `GRPC_MAX_MESSAGE_SIZE`: Maximum message size in bytes (default: 100MB)
- `GRPC_MAX_CONCURRENT_RPCS`: Maximum concurrent RPCs (default: 100)
- `GRPC_ENABLE_REFLECTION`: Enable server reflection (default: true)
- `GRPC_REQUIRE_API_KEY`: Require API key for authentication (default: false)
- `GRPC_API_KEY`: API key for authentication (default: "")
- `GRPC_USE_TLS`: Use TLS/SSL (default: false)
- `GRPC_CERT_FILE`: Path to SSL certificate file (default: "/etc/inference/certs/server.crt")
- `GRPC_KEY_FILE`: Path to SSL key file (default: "/etc/inference/certs/server.key")

### Go Maistro Service

The gRPC client can be configured in `config.yaml`:

```yaml
inference:
  use_grpc: true
  grpc_address: "localhost:50051"
  use_secure_connection: false
  api_key: ""
```

## API Documentation

The gRPC API is defined in `proto/inference.protos`. It provides the following services:

- `ChatStream`: Streaming chat responses
- `GenerateStream`: Streaming text generation
- `GetEmbedding`: Generate embeddings from text
- `GenerateImage`: Request image generation
- `EditImage`: Request image editing
- `CheckImageStatus`: Check status of image generation/editing
- `ManageVRAM`: Bidirectional streaming for VRAM management
- `ClearMemory`: Clear VRAM and cache
- `GetMemoryStats`: Get memory allocation stats
