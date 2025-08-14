# LLM ML Lab - Inference

This directory contains three separate but interconnected projects for language model infrastructure:

1. [evaluation](#evaluation) - For benchmarking, training, and fine-tuning
2. [server](#server) - For API services (REST and gRPC)
3. [runner](#runner) - For model execution and management

Each project has its own isolated virtual environment to avoid dependency conflicts.

## Directory Structure

```
/inference
├── evaluation/            # Benchmarks, training, and fine-tuning code
├── server/                # REST API and gRPC server code
├── runner/                # Model runner and pipelines
├── Dockerfile             # Container definition with isolated environments
├── startup.sh             # Container startup script for service orchestration
├── run_with_env.sh        # Helper script to run commands in specific environments
└── setup_environments.sh  # Script to set up local development environments
```

## Overview

### evaluation

The evaluation project contains tools for:
- Evaluating model performance on academic and practical benchmarks
- Fine-tuning models for specific tasks
- Visualizing and analyzing model performance

### server

The server project provides:
- OpenAI-compatible REST API endpoints
- Efficient gRPC services
- Business logic and service layer
- Model definition and type safety

### runner

The model runner project provides:
- Pipeline implementations for text and image generation
- Model downloading, processing, and quantization utilities
- Configuration management

## Features

- **Image Generation**: Generate images from text prompts using Stable Diffusion models
- **Text Generation**: Generate text completions with streaming support
- **Chat Completions**: Generate conversational responses with streaming support
- **Embeddings**: Generate vector embeddings for text
- **gRPC API**: High-performance gRPC API for efficient inter-service communication
- **HTTP API**: RESTful API endpoints for direct client access
- **Multiple Model Support**: Add, remove, and switch between different models
- **LoRA Adapter Support**: Manage and apply LoRA adapters to customize model outputs
- **On-Demand Model Loading**: Models are loaded only when needed and unloaded after use to conserve memory
- **Hardware Resource Management**: Automatic detection and optimization based on available GPU resources
- **Adaptive Performance**: Automatically adjusts parameters based on available memory
- **Memory Optimization**: Multiple strategies to reduce VRAM usage
- **Multi-GPU Support**: Can utilize multiple GPUs if available

## Setup and Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for performance)

### Local Development

For local development, use the `setup_environments.sh` script to create virtual environments:

```bash
./setup_environments.sh
```

To run commands in a specific environment, use the `run_with_env.sh` script:

```bash
# Examples
./run_with_env.sh server python -m uvicorn app:app --port 8000
./run_with_env.sh evaluation python -m run_eval_direct
./run_with_env.sh runner python -c "import torch; print(torch.cuda.is_available())"
```

### Docker

#### Building the Docker Image

```bash
docker build -t llmmllab:latest -f inference/Dockerfile .
```

#### Running the Docker Container

```bash
docker run --gpus all -p 8000:8000 -p 50051:50051 llmmllab:latest
```

This will start:
1. REST API server 
2. gRPC server (if available)

#### Running Commands in Docker

To run commands in a specific environment in the Docker container:

```bash
docker exec -it <container_id> /app/run_with_env.sh server python -m your_command
```

### Logs

In Docker, service logs are available in:

- `/var/log/server_api.log` - REST API server logs
- `/var/log/grpc_server.log` - gRPC server logs

### Running the gRPC Server

```bash
# Generate proto code
cd server/protos
python ../proto_wrapper.py

# Run the server
cd ..
python grpc_server.py
```

For more details, see the [GRPC README](../GRPC_README.md) and [gRPC Architecture Documentation](../docs/grpc_architecture.md).

## License

[MIT License](../LICENSE)

For more details, see [gRPC Server README](grpc_server/README.md) and [gRPC Architecture Documentation](../docs/grpc_architecture.md).

## API Reference

The Inference service provides multiple API endpoints organized by resource type. Each endpoint is available in both non-versioned (`/resource`) and versioned (`/v1/resource`) paths.

### Chat API (`/chat`)

Handles conversation and chat completions with support for streaming responses.

- `POST /chat/completions` - Generate chat completions
  - Request: Messages, model parameters, streaming flag
  - Response: Chat completion or streaming response
  
- `GET /chat/admin` - Admin-only endpoint for chat management

### Config API (`/config`)

Manages user and system configurations.

- `GET /config/` - Get the current user's configuration
- `PUT /config/` - Update the user's configuration

### Images API (`/images`)

Handles image generation, editing, and retrieval.

- `POST /images/generate` - Generate images from text prompts
- `POST /images/edit` - Edit existing images with text prompts
- `GET /images/user/{user_id}` - Get images for a specific user
- `GET /images/download/{filename}` - Download a generated image
- `GET /images/check-image-status/{request_id}` - Check generation status
- `DELETE /images/{image_id}` - Delete a generated image
- `POST /images/store-image` - Upload and store an image

### Internal API (`/internal`)

Restricted access endpoints for internal services.

- `GET /internal/image/{user_id}/{filename}` - Get a user's image for internal services

### Models API (`/models`)

Manages model configurations and profiles.

- `GET /models/` - List all available models
- `GET /models/profiles` - Get model profiles for the current user
- `GET /models/profiles/{profile_id}` - Get a specific model profile
- `POST /models/profiles` - Create a new model profile
- `PUT /models/profiles/{profile_id}` - Update a model profile
- `DELETE /models/profiles/{profile_id}` - Delete a model profile

### OpenAI Compatible API (`/v1`)

OpenAI-compatible endpoints for drop-in compatibility.

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `POST /v1/completions` - OpenAI-compatible text completions
- `POST /v1/embeddings` - OpenAI-compatible text embeddings
- `GET /v1/models` - List available models in OpenAI format

### Resources API (`/resources`)

Manages hardware resources and GPU allocation.

- `GET /resources/malloc` - Get memory usage statistics
- `GET /resources/processes` - Get information about GPU processes
- `POST /resources/clear` - Clear memory cache for devices
- `GET /resources/health` - Check the health of GPU devices

### Static API (`/static`)

Serves static files like images.

- `GET /static/images/view/{filename}` - Serve an image for viewing
- `GET /static/images/download/{filename}` - Download an image

### Users API (`/users`)

Manages user information and conversations.

- `GET /users/` - Get all users (admin only)
- `GET /users/{user_id}/conversations` - Get conversations for a user

### WebSockets API (`/ws`)

Handles real-time WebSocket connections.

- `WebSocket /ws/chat/{conversation_id}` - Real-time chat communication
- `WebSocket /ws/image` - Real-time image generation status updates
- `WebSocket /ws/status` - Real-time system status updates
```

#### Add a new model

```
POST /models/
```

Request body:
```json
{
  "name": "SDXL",
  "source": "stabilityai/stable-diffusion-xl-base-1.0",
  "description": "Stable Diffusion XL model"
}
```

#### Set active model

```
PUT /models/active/{model_id}
```

#### Remove a model

```
DELETE /models/{model_id}
```

### LoRA Management

> **Note:** LoRA support requires the PEFT library. Install it with: `pip install -U peft`

#### List all LoRAs

```
GET /loras/
```

#### Add a new LoRA

```
POST /loras/
```

Request body:
```json
{
  "name": "Anime Style",
  "source": "anime-style-lora/anime",
  "description": "Anime style LoRA adapter",
  "weight": 0.75
}
```

#### Activate a LoRA

```
PUT /loras/{lora_id}/activate
```

#### Deactivate a LoRA

```
PUT /loras/{lora_id}/deactivate
```

#### Set LoRA weight

```
PUT /loras/{lora_id}/weight
```

Request body:
```json
{
  "weight": 0.8
}
```

## Memory Management Features

The API implements several strategies to optimize memory usage:

1. **On-Demand Model Loading**: Models are loaded only when needed for generation and immediately unloaded afterward
2. **Hardware-Aware Parameter Adjustment**: Generation parameters are automatically adjusted based on available VRAM
3. **Memory-Efficient Attention**: Optimized attention mechanisms to reduce memory usage
4. **VAE Slicing**: Reduces memory usage during the VAE decoding process
5. **CPU Offloading**: Options to offload parts of the model to CPU when memory is constrained
6. **Automatic Resolution Scaling**: Automatically reduces image dimensions when memory is low

These features allow the API to run on devices with limited VRAM while still producing high-quality images.

## License

[Specify your license here]
