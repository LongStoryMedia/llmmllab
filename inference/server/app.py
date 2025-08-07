"""

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
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import login

from inference.server.routers import chat as chat
from server.config import CONFIG_DIR, IMAGE_DIR
from server.routers import (
    images,
    models,
    resources,
    config,
    static,
    websockets,
    users,
)
from server.auth import AuthMiddleware
from server.config import API_VERSION

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
    except (ValueError, ConnectionError, TimeoutError) as e:
        print(f"Failed to log in to Hugging Face: {e}")
        print("Continuing without Hugging Face authentication")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: initialize hardware monitoring and cleanup service
    print("Initializing services...")
    hardware_manager.clear_memory()
    cleanup_service.start()
    grpc_process = None

    # Initialize auth middleware
    from server.config import AUTH_JWKS_URI

    print(f"Initializing auth middleware with JWKS URI: {AUTH_JWKS_URI}")
    # Store in app.state so we can access it in middleware without global variables
    _.state.auth_middleware = AuthMiddleware(AUTH_JWKS_URI)

    # Initialize database connection
    from server.db import storage
    from server.config import DB_CONNECTION_STRING

    if DB_CONNECTION_STRING:
        print("Initializing database connection...")
        try:
            await storage.initialize(DB_CONNECTION_STRING)
            print("Database connection initialized successfully")
        except Exception as e:
            print(f"Error initializing database connection: {e}")
            print("Some features that depend on the database may not work properly")

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
    from server.config import (
        RABBITMQ_ENABLED,
        RABBITMQ_PASSWORD,
        INFERENCE_SERVICES_PORT,
    )

    if not RABBITMQ_ENABLED:
        print("RabbitMQ integration disabled via configuration")
    else:
        print("Starting RabbitMQ consumer...")
        try:
            if not RABBITMQ_PASSWORD:
                print("Warning: RABBITMQ_PASSWORD not set")

            # Start the RabbitMQ consumer
            rabbitmq_consumer.password = RABBITMQ_PASSWORD
            rabbitmq_consumer.start()
            print("RabbitMQ consumer started successfully!")
        except (ConnectionError, OSError, ValueError) as e:
            print(f"Error starting RabbitMQ consumer: {e}")
            print("RabbitMQ integration will not be available")

    # Start the gRPC server in a subprocess for live reload support
    print("Starting gRPC server subprocess...")
    # grpc_process = subprocess.Popen(
    #     [
    #         sys.executable,
    #         os.path.join(os.path.dirname(__file__), "run_grpc_server.py"),
    #         "--port",
    #         str(INFERENCE_SERVICES_PORT),
    #     ]
    # )
    # print(f"gRPC server started with PID {grpc_process.pid}")

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
        except (RuntimeError, ValueError) as e:
            print(f"Error stopping vLLM service: {e}")

        # Stop RabbitMQ consumer
        try:
            rabbitmq_consumer.stop()
        except (AttributeError, ConnectionError) as e:
            print(f"Error stopping RabbitMQ consumer: {e}")

        # Terminate the gRPC server subprocess if running
        try:
            if grpc_process is not None and grpc_process.poll() is None:
                print(f"Terminating gRPC server subprocess (PID {grpc_process.pid})...")
                grpc_process.terminate()
                # grpc_process.wait(timeout=5)
                grpc_process.kill()
                print("gRPC server subprocess terminated.")
        except (ProcessLookupError, OSError) as e:
            print(f"Error terminating gRPC server subprocess: {e}")

        cleanup_service.shutdown()
        hardware_manager.clear_memory(aggressive=True)


# Initialize the FastAPI application with the lifespan context manager
app = FastAPI(
    title="Inference API",
    description="FastAPI server for inference with API versioning (current version: v1)",
    version="0.1.0",
    redoc_url="/redoc",
    docs_url="/docs",
)


@app.middleware("http")
async def auth_middleware_handler(request: Request, call_next):
    """Authentication middleware to handle token validation and user identification"""
    # Skip auth for public endpoints
    public_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/"]
    if any(request.url.path.startswith(path) for path in public_paths):
        response = await call_next(request)
        return response

    # Skip auth if middleware is not initialized or disabled
    app_instance = request.app
    if (
        not hasattr(app_instance.state, "auth_middleware")
        or os.environ.get("DISABLE_AUTH", "").lower() == "true"
    ):
        response = await call_next(request)
        return response

    try:
        # Get the auth middleware from app state
        auth_middleware = app_instance.state.auth_middleware

        # Authenticate the request
        await auth_middleware.authenticate(request)

        # If authentication succeeds, proceed with the request
        response = await call_next(request)

        # Add any auth-related response headers
        if hasattr(request.state, "response_headers"):
            for key, value in request.state.response_headers.items():
                response.headers[key] = value

        return response
    except HTTPException as e:
        # Handle FastAPI HTTP exceptions with proper status code and detail
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except ValueError as e:
        # Handle validation errors
        return JSONResponse(
            status_code=400, content={"error": f"Validation error: {str(e)}"}
        )
    except (ConnectionError, TimeoutError) as e:
        # Handle connection errors
        return JSONResponse(
            status_code=503, content={"error": f"Service unavailable: {str(e)}"}
        )
    except RuntimeError as e:
        # Handle runtime errors
        return JSONResponse(
            status_code=500, content={"error": f"Server error: {str(e)}"}
        )


# Include non-versioned routers (for backward compatibility)
app.include_router(images.router)
app.include_router(models.router)
app.include_router(resources.router)
app.include_router(chat.router)
app.include_router(config.router)
app.include_router(static.router)
app.include_router(websockets.router)
app.include_router(users.router)

# Import and include the internal router
from server.routers import internal

app.include_router(internal.router)

# Include versioned routers
version = API_VERSION
app.include_router(images.router, prefix=f"/{version}")
app.include_router(models.router, prefix=f"/{version}")
app.include_router(resources.router, prefix=f"/{version}")
app.include_router(chat.router, prefix=f"/{version}")
app.include_router(config.router, prefix=f"/{version}")
app.include_router(static.router, prefix=f"/{version}")
app.include_router(websockets.router, prefix=f"/{version}")
app.include_router(users.router, prefix=f"/{version}")
# Internal router is intentionally not versioned to maintain isolation
# app.include_router(internal.router, prefix=f"/{version}")

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
        "api_version": API_VERSION,
        "endpoints": {
            "image_generation": "/docs#/images",
            "chat": "/docs#/chat",
            "openai_compatible": "/v1/",
            "models": "/docs#/models",
            "loras": "/docs#/loras",
            "resources": "/docs#/resources",
        },
        "versioned_endpoints": {
            "image_generation": f"/{API_VERSION}/images",
            "chat": f"/{API_VERSION}/chat",
            "models": f"/{API_VERSION}/models",
            "config": f"/{API_VERSION}/config",
            "resources": f"/{API_VERSION}/resources",
            "websockets": f"/{API_VERSION}/ws",
            "users": f"/{API_VERSION}/users",
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
