# Stable Diffusion API

A FastAPI-based service for generating images using Stable Diffusion models with advanced memory optimization and hardware management.

## Features

- **Image Generation**: Generate images from text prompts using Stable Diffusion models
- **Multiple Model Support**: Add, remove, and switch between different Stable Diffusion models
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
└── Dockerfile              # Docker configuration
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for performance)

### Dependencies

Key Python packages required:
- FastAPI
- diffusers
- torch
- accelerate
- transformers
- peft (for LoRA support)
- xformers (optional, for memory-efficient attention)

### Local Development

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
