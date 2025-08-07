# LLM ML Lab

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
- **RESTful API**: Clean API endpoints for all functionality

## Project Structure

```
inference/
├── app.py                  # Main application entry point
├── config.py               # Application configuration
├── models/                 # Pydantic models for request/response
│   ├── __init__.py
│   ├── models.py           # Model management data models
│   └── requests.py         # Request data models
├── routers/                # API routes
│   ├── __init__.py
│   ├── images.py           # Image generation routes
│   ├── models.py           # Model management routes
│   └── loras.py            # LoRA management routes
├── services/               # Business logic
│   ├── __init__.py
│   ├── cleanup_service.py  # Automatic cleanup service
│   ├── model_service.py    # Model management service
│   ├── lora_service.py     # LoRA management service
│   ├── image_generator.py  # Image generation service
│   └── hardware_manager.py # Hardware resource management
├── grpc_server/            # gRPC server implementation
│   ├── proto/              # Protocol buffers definitions
│   ├── server.py           # gRPC server entry point
│   └── README.md           # gRPC server documentation
└── Dockerfile              # Docker configuration
```

## Setup and Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for performance)
- Docker (optional, for containerized deployment)

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
docker run --gpus all -p 8000:8000 -p 50051:50051 -p 11434:11434 llmmllab:latest
```

This will start:
1. Ollama service
2. REST API server (if available)
3. gRPC server (if available)

#### Running Commands in Docker

To run commands in a specific environment in the Docker container:

```bash
docker exec -it <container_id> /app/run_with_env.sh server python -m your_command
```

### Logs

In Docker, service logs are available in:
- `/var/log/ollama.log` - Ollama service logs
- `/var/log/server_api.log` - REST API server logs
- `/var/log/grpc_server.log` - gRPC server logs

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd inference
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   # Make sure PEFT is installed for LoRA support
   pip install -U peft
   
   # Optional: Install xformers for memory optimization (CUDA only)
   pip install -U xformers
   ```

4. Run the application:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t stable-diffusion-api .
   ```

2. Run the container with GPU support:
   ```bash
   docker run --gpus all -p 8000:8000 stable-diffusion-api
   ```

### Running the gRPC Server

```bash
# Generate proto code
cd grpc_server/proto
python generate_proto.py

# Run the server
cd ../..
python grpc_server/server.py
```

For more details, see [gRPC Server README](grpc_server/README.md) and [gRPC Architecture Documentation](../docs/grpc_architecture.md).

## API Reference

### Image Generation

#### Generate an image from a prompt

```
POST /generate-image
```

Request body:
```json
{
  "prompt": "a photo of an astronaut riding a horse on mars",
  "width": 768,
  "height": 768,
  "inference_steps": 30,
  "guidance_scale": 7.5,
  "low_memory_mode": false
}
```

Response:
```json
{
  "image": "base64-encoded-image-data",
  "download": "/download/unique-identifier.png"
}
```

#### Download a generated image

```
GET /download/{filename}
```

Returns the image file.

### Model Management

#### List all models

```
GET /models/
```

Response:
```json
{
  "models": [
    {
      "id": "default",
      "name": "Default Stable Diffusion",
      "source": "runwayml/stable-diffusion-v1-5",
      "description": "Default Stable Diffusion model",
      "is_active": true
    }
  ],
  "active_model": "default"
}
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
