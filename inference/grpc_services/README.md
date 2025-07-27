# gRPC Inference API

This directory contains the implementation of the gRPC server for the inference service.

## Overview

The gRPC server provides the following services:
- Chat generation with streaming responses
- Text generation with streaming responses
- Embedding generation
- Image generation
- Image editing
- VRAM memory management with bidirectional streaming

## Getting Started

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Generate Proto Code

Before running the server, you need to generate the Python code from the protobuf definitions:

```bash
cd grpc_server/proto
python generate_proto.py
```

### 3. Run the Server

```bash
python grpc_server/server.py
```

By default, the server listens on port 50051. You can specify a different port:

```bash
python grpc_server/server.py --port 50052
```

### 4. Configuration

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

## Service Implementations

The server delegates requests to specialized service implementations:

- `ChatService`: Handles chat and text generation requests
- `ImageService`: Handles image generation and editing requests
- `EmbeddingService`: Handles embedding generation requests
- `VRAMService`: Handles VRAM memory management requests

## Client Usage

For Go client usage, see the maistro/grpc/inference package.

## Docker Support

To run the server in a Docker container, use the provided Dockerfile:

```bash
docker build -t inference-grpc -f Dockerfile.Inference.GRPC .
docker run -p 50051:50051 inference-grpc
```
