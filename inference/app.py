import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import IMAGE_DIR, CONFIG_DIR
from routers import images, models, loras
from services.cleanup_service import cleanup_service
from services.hardware_manager import hardware_manager  # Import hardware manager

# Set PyTorch memory management environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Create required directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize hardware monitoring and cleanup service
    print("Initializing services...")
    cleanup_service.start()

    # Log hardware information
    print(f"Hardware status: {hardware_manager.get_memory_status_str()}")

    yield  # This is where FastAPI serves requests

    # Shutdown: clean up resources
    cleanup_service.shutdown()
    hardware_manager.clear_memory()

    # Clear CUDA cache on shutdown to free memory
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        "message": "Welcome to the Stable Diffusion API. Use POST /generate-image to create images."
    }
