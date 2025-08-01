"""
Stable Diffusion API with OpenAI Compatibility

This FastAPI application provides a comprehensive API for generating images using Stable Diffusion
and text generation with OpenAI-compatible endpoints. The server integrates multiple services:

- Image generation via Stable Diffusion
- Text generation via vLLM with OpenAI-compatible API
- Model management (loading, unloading, listing)
- LoRA adapter management
- Resource monitoring and management
- RabbitMQ integration for asynchronous processing
- gRPC service for additional communication

Environment Variables:
- HF_TOKEN: Hugging Face token for model access
- VLLM_MODEL: Model to use for vLLM service (default: "microsoft/DialoGPT-medium")
- RABBITMQ_PASSWORD: Password for RabbitMQ authentication
- PYTORCH_CUDA_ALLOC_CONF: Configured to "expandable_segments:True" to avoid memory fragmentation

Main Components:
- FastAPI application with various routers
- Lifespan context manager for service initialization and cleanup
- Hardware monitoring and memory management
- OpenAI-compatible endpoints (/v1/*)
- Health check endpoint for monitoring system status

Endpoints:
- /: Root endpoint with API information
- /health: Health check endpoint
- /images/*: Image generation endpoints
- /chat/*: Chat completion endpoints
- /models/*: Model management endpoints
- /loras/*: LoRA adapter management endpoints
- /resources/*: System resource endpoints
- /v1/*: OpenAI-compatible endpoints

The application handles initialization and cleanup of all services and provides
detailed logging throughout the startup and shutdown processes.
"""

import os
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login

from server.config import CONFIG_DIR, IMAGE_DIR
from server.routers import chat, images, models, resources

# from server.routers.openai_compatible import (
#     cleanup_vllm_service,
#     initialize_vllm_service,
# )
# from server.routers.openai_compatible import router as openai_router
from server.services.cleanup_service import cleanup_service
from server.services.hardware_manager import hardware_manager  # Import hardware manager
from server.services.rabbitmq_consumer import (
    rabbitmq_consumer,
)  # Import RabbitMQ consumer

# Create required directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("Logging into Hugging Face with token from environment variable")
    login(token=hf_token)
else:
    print(
        "Warning: No HF_TOKEN environment variable found. Some features may not work properly."
    )
    # Try login without token, will use cached credentials if available
    try:
        login(token=None)
    except Exception as e:
        print(f"Failed to log in to Hugging Face: {e}")
        print("Continuing without Hugging Face authentication")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize hardware monitoring and cleanup service
    print("Initializing services...")
    hardware_manager.clear_memory()
    cleanup_service.start()
    grpc_process = None

    # # Initialize vLLM service for OpenAI compatibility
    # print("Initializing vLLM service...")
    # try:
    #     # Get model name from environment or use default
    #     # vllm_model = os.environ.get("VLLM_MODEL", "microsoft/DialoGPT-medium")
    #     # await initialize_vllm_service(vllm_model)
    #     print(f"vLLM service initialized with model: {vllm_model}")
    # except Exception as e:
    #     print(f"Warning: Failed to initialize vLLM service: {e}")
    #     print("OpenAI-compatible endpoints may not be available")

    # Initialize and start RabbitMQ consumer
    print("Starting RabbitMQ consumer...")
    try:
        # Get password from environment variable
        rabbitmq_password = os.environ.get("RABBITMQ_PASSWORD", "")
        if not rabbitmq_password:
            print("Warning: RABBITMQ_PASSWORD environment variable not set")

        # Start the RabbitMQ consumer
        rabbitmq_consumer.password = rabbitmq_password
        rabbitmq_consumer.start()
        print("RabbitMQ consumer started successfully!")

        # Start the gRPC server in a subprocess for live reload support
        print("Starting gRPC server subprocess...")
        # grpc_process = subprocess.Popen(
        #     [
        #         sys.executable,
        #         os.path.join(os.path.dirname(__file__), "run_grpc_server.py"),
        #         "--port",
        #         "50051",
        #     ]
        # )
        # print(f"gRPC server started with PID {grpc_process.pid}")
    except Exception as e:
        print(f"Error starting RabbitMQ consumer: {e}")
        print("RabbitMQ integration will not be available")

    # Log hardware information
    # Add this near the beginning of your test to check CUDA capability
    import torch

    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability(i)}")

    try:
        yield  # This is where FastAPI serves requests
    finally:
        # Shutdown: clean up resources
        print("Shutting down services...")

        # Stop vLLM service
        try:
            # await cleanup_vllm_service()
            print("vLLM service shutdown completed")
        except Exception as e:
            print(f"Error stopping vLLM service: {e}")

        # Stop RabbitMQ consumer
        try:
            rabbitmq_consumer.stop()
        except Exception as e:
            print(f"Error stopping RabbitMQ consumer: {e}")

        # Terminate the gRPC server subprocess if running
        try:
            if grpc_process is not None and grpc_process.poll() is None:
                print(f"Terminating gRPC server subprocess (PID {grpc_process.pid})...")
                grpc_process.terminate()
                # grpc_process.wait(timeout=5)
                grpc_process.kill()
                print("gRPC server subprocess terminated.")
        except Exception as e:
            print(f"Error terminating gRPC server subprocess: {e}")

        cleanup_service.shutdown()
        hardware_manager.clear_memory(aggressive=True)


# Initialize the FastAPI application with the lifespan context manager
app = FastAPI(
    title="Stable Diffusion API with OpenAI Compatibility",
    description="API for generating images using Stable Diffusion and OpenAI-compatible text generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(images.router)
app.include_router(models.router)
app.include_router(resources.router)
app.include_router(chat.router)

# Include OpenAI-compatible router
# app.include_router(openai_router, tags=["openai-compatible"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Stable Diffusion API with OpenAI Compatibility",
        "endpoints": {
            "image_generation": "/docs#/images",
            "chat": "/docs#/chat",
            "openai_compatible": "/v1/",
            "models": "/docs#/models",
            "loras": "/docs#/loras",
            "resources": "/docs#/resources",
        },
        "openai_endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "health": "/v1/health",
        },
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    # from .routers.openai_compatible import vllm_service

    health_status = {
        "status": "healthy",
        "services": {
            "hardware_manager": "active",
            # "vllm_service": {
            #     "status": "ready" if vllm_service.is_ready() else "not_ready",
            #     "model": vllm_service.model_name or "not_loaded",
            # },
        },
        "cuda_available": False,
        "cuda_devices": 0,
    }

    # Check CUDA availability
    try:
        import torch

        health_status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            health_status["cuda_devices"] = torch.cuda.device_count()
    except ImportError:
        health_status["cuda_available"] = False

    return health_status
