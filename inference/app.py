from huggingface_hub import login
import os
from contextlib import asynccontextmanager
import subprocess
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import IMAGE_DIR, CONFIG_DIR
from routers import images, models, loras, resources, chat
from services.cleanup_service import cleanup_service
from services.hardware_manager import hardware_manager  # Import hardware manager
from services.image_generator import image_generator  # Import image generator
from services.model_service import model_service  # Import model service
from services.rabbitmq_consumer import rabbitmq_consumer  # Import RabbitMQ consumer

# Set PyTorch memory management environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Create required directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("Logging into Hugging Face with token from environment variable")
    login(token=hf_token)
else:
    print("Warning: No HF_TOKEN environment variable found. Some features may not work properly.")
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
        grpc_process = subprocess.Popen([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "run_grpc_server.py"),
            "--port", "50051"
        ])
        print(f"gRPC server started with PID {grpc_process.pid}")
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
    title="Stable Diffusion API",
    description="API for generating images using Stable Diffusion",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(images.router)
app.include_router(models.router)
app.include_router(loras.router)  # Add the LoRA router
app.include_router(resources.router)
app.include_router(chat.router)

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
        "message": "BUTTS."
    }
