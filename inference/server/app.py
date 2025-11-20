"""

This FastAPI application provides a comprehensive API for generating images using Stable Diffusion
and text generation with OpenAI-compatible endpoints. The server integrates multiple services:

- Image generation via Stable Diffusion
- Text generation via vLLM with OpenAI-compatible API
- Model management (loading, unloading, listing)
- LoRA adapter management
- Resource monitoring and management

Environment Variables:
- HF_TOKEN: Hugging Face token for model access
- VLLM_MODEL: Model to use for vLLM service (default: "microsoft/DialoGPT-medium")
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

from server.config import CONFIG_DIR, IMAGE_DIR
from server.routers import (
    images,
    config,
    static,
    websockets,
    users,
    todos,
    model,
    chat,
    conversation,
    internal,
    db_admin,
)
from server.middleware import (
    AuthMiddleware,
    db_init_middleware,
    MessageValidationMiddleware,
)
from server.config import API_VERSION, AUTH_JWKS_URI
from server.cleanup_service import cleanup_service
from db.maintenance import maintenance_service
from utils.logging import llmmllogger
from composer import shutdown_composer
from runner import local_pipeline_cache


logger = llmmllogger.bind(component="app")

# # # Enable auth bypass for testing
# os.environ["DISABLE_AUTH"] = "true"
# # # Set test user ID to match existing conversation owner
# os.environ["TEST_USER_ID"] = "CgNsc20SBGxkYXA"

# Create required directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    logger.info("Logging into Hugging Face with token from environment variable")
    login(token=hf_token)
else:
    logger.info(
        "Warning: No HF_TOKEN environment variable found. Some features may not work properly."
    )
    # Try login without token, will use cached credentials if available
    try:
        login(token=None)
    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.info(f"Failed to log in to Hugging Face: {e}")
        logger.info("Continuing without Hugging Face authentication")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Simplified lifespan: start services, optionally init composer, yield, then shutdown."""
    logger.info("Initializing services...")
    cleanup_service.start()

    # Initialize database connection and schema if configured
    try:
        from db import storage  # pylint: disable=import-outside-toplevel
        from server.config import (  # pylint: disable=import-outside-toplevel
            DB_CONNECTION_STRING,
        )

        assert DB_CONNECTION_STRING is not None, "DB_CONNECTION_STRING is not set"

        await storage.initialize(DB_CONNECTION_STRING)
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")

    try:
        yield  # Application runs here
    finally:
        # Shutdown: clean up resources
        logger.info("Shutting down services...")

        # Stop database maintenance service if running
        try:
            logger.info("Stopping database maintenance service...")
            await maintenance_service.stop_maintenance_schedule()
            logger.info("Database maintenance service stopped")
        except Exception as e:
            logger.info(f"Error stopping database maintenance service: {e}")

        # Stop composer service
        try:
            await shutdown_composer()
            logger.info("Composer service shutdown completed")
        except Exception as e:
            logger.info(f"Error stopping composer service: {e}")

        # Attempt to gracefully stop and cleanup pipeline cache
        try:
            if local_pipeline_cache is not None:
                logger.info("Stopping local pipeline cache and cleaning pipelines...")
                try:
                    local_pipeline_cache.stop()
                    logger.info("Local pipeline cache stopped and cleaned")
                except Exception as e:
                    logger.info(f"Error stopping local pipeline cache: {e}")
        except Exception:
            pass

        cleanup_service.shutdown()


logger.info(f"Pre-initializing auth middleware with JWKS URI: {AUTH_JWKS_URI}")
global_auth_middleware = AuthMiddleware(AUTH_JWKS_URI)

# Initialize the FastAPI application with the lifespan context manager
app = FastAPI(
    title="Inference API",
    description="FastAPI server for inference with API versioning (current version: v1)",
    version="0.1.0",
    redoc_url="/redoc",
    docs_url="/docs",
    lifespan=lifespan,
)

# Store auth middleware in app.state right away
app.state.auth_middleware = global_auth_middleware
# Add message validation middleware to ensure proper response structure
app.add_middleware(MessageValidationMiddleware)
app.middleware("http")(db_init_middleware)


@app.middleware("http")
async def auth_middleware_handler(request: Request, call_next):
    """Authentication middleware to handle token validation and user identification"""
    # Get logger for debugging
    logger.debug(f"Processing request for path: {request.url.path}")

    # Skip auth for public endpoints
    public_paths = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/static/images/view/",
    ]

    # Check for exact root path or if the path starts with any of the public paths
    if request.url.path == "/" or any(
        request.url.path.startswith(path) for path in public_paths
    ):
        logger.debug(f"Skipping auth for public path: {request.url.path}")
        response = await call_next(request)
        return response

    # Skip auth if middleware is not initialized or disabled
    app_instance = request.app
    if not hasattr(app_instance.state, "auth_middleware"):
        logger.error(
            "Auth middleware not initialized in app state - this should never happen now"
        )
        # Instead of skipping auth, we'll return an error
        return JSONResponse(
            status_code=500,
            content={"error": "Authentication middleware not initialized properly"},
        )

    if os.environ.get("DISABLE_AUTH", "").lower() == "true":
        logger.warning("Auth is disabled via environment variable")
        response = await call_next(request)
        return response

    try:
        # Get the auth middleware from app state
        auth_middleware = app_instance.state.auth_middleware
        logger.debug(f"Authenticating request for path: {request.url.path}")

        # Authenticate the request
        await auth_middleware.authenticate(request)
        logger.debug("Authentication successful")

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


# Add CORS middleware BEFORE including routers to ensure it's processed in the right order
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include non-versioned routers (for backward compatibility)
app.include_router(images.router)
app.include_router(model.router)
app.include_router(chat.router)
app.include_router(conversation.router)
app.include_router(config.router)
app.include_router(static.router)
app.include_router(websockets.router)
app.include_router(users.router)
app.include_router(todos.router)

# Import and include the internal router
app.include_router(internal.router)
app.include_router(db_admin.router)

# Include versioned routers
version = API_VERSION
app.include_router(images.router, prefix=f"/{version}")
app.include_router(model.router, prefix=f"/{version}")
app.include_router(chat.router, prefix=f"/{version}")
app.include_router(conversation.router, prefix=f"/{version}")
app.include_router(config.router, prefix=f"/{version}")
app.include_router(static.router, prefix=f"/{version}")
app.include_router(websockets.router, prefix=f"/{version}")
app.include_router(users.router, prefix=f"/{version}")
app.include_router(todos.router, prefix=f"/{version}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Stable Diffusion API with OpenAI Compatibility",
        "api_version": API_VERSION,
        "endpoints": {
            "image_generation": "/docs#/images",
            "chat": "/docs#/chat",
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
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""

    return "OK"
