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

from server.routers import model
from server.routers import chat
from server.routers import (
    conversation,
)  # Import conversation router to register endpoints
from server.config import CONFIG_DIR, IMAGE_DIR
from server.routers import (
    images,
    config,
    static,
    websockets,
    users,
)
from server.middleware.auth import AuthMiddleware
from server.config import API_VERSION
from server.cleanup_service import cleanup_service
from db.maintenance import maintenance_service

# Enable auth bypass for testing
os.environ["DISABLE_AUTH"] = "true"
# Set test user ID to match existing conversation owner
os.environ["TEST_USER_ID"] = "CgNsc20SBGxkYXA"

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
    cleanup_service.start()

    # Auth middleware is already initialized and stored in app.state
    print("Auth middleware already initialized and stored in app.state")

    # Initialize database connection
    from db import storage  # pylint: disable=import-outside-toplevel
    from server.config import (  # pylint: disable=import-outside-toplevel
        DB_CONNECTION_STRING,
        logger,
    )
    from db.init_db import (  # pylint: disable=import-outside-toplevel
        initialize_database,
    )

    # Build connection string from individual environment variables if not already set
    connection_string = DB_CONNECTION_STRING

    if connection_string:
        try:
            await storage.initialize(connection_string)
            logger.info("Database connection initialized successfully")

            # Ensure default model profiles exist
            logger.info("Initializing default model profiles...")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    else:
        logger.warning(
            "No database connection string provided, continuing without database access"
        )
        print(
            "DB_CONNECTION_STRING not found, attempting to build from individual variables..."
        )
        db_host = os.environ.get("DB_HOST")
        db_port = os.environ.get("DB_PORT")
        db_name = os.environ.get("DB_NAME")
        db_user = os.environ.get("DB_USER")
        db_password = os.environ.get("DB_PASSWORD")

        if db_host and db_port and db_name and db_user and db_password:
            connection_string = (
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            print(
                f"Built connection string from environment variables: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}"
            )
        else:
            print("Cannot build connection string, missing environment variables:")
            print(f"  DB_HOST: {'✓' if db_host else '✗'}")
            print(f"  DB_PORT: {'✓' if db_port else '✗'}")
            print(f"  DB_NAME: {'✓' if db_name else '✗'}")
            print(f"  DB_USER: {'✓' if db_user else '✗'}")
            print(f"  DB_PASSWORD: {'✓' if db_password else '✗'}")

    if connection_string:
        print("Initializing database connection...")
        try:
            # Initialize the connection pool
            await storage.initialize(connection_string)
            print("Database connection initialized successfully")

            # Initialize the database schema if needed
            if storage.pool:
                print("Initializing database schema...")
                schema_initialized = await initialize_database(storage.pool)
                if schema_initialized:
                    print("Database schema initialized successfully")

                    # Align sequences at startup to avoid ID drift after restores/migrations
                    try:
                        await maintenance_service.initialize(storage.pool)
                        await maintenance_service.align_sequences()
                        print("Database sequences aligned successfully at startup")
                    except Exception as e:
                        print(
                            f"Warning: failed to align database sequences at startup: {e}"
                        )

                    # Initialize and start the database maintenance service
                    # Set interval to 24 hours by default (can be configured via environment variable)
                    maintenance_interval = int(
                        os.environ.get("DB_MAINTENANCE_INTERVAL_HOURS", "24")
                    )
                    print(
                        f"Initializing database maintenance service with {maintenance_interval} hour interval"
                    )
                    await maintenance_service.initialize(
                        storage.pool, maintenance_interval
                    )
                    await maintenance_service.start_maintenance_schedule()
                    print("Database maintenance service started")

                    print("Creating/Updating default model profiles...")
                    try:
                        await storage.get_service(
                            storage.model_profile
                        ).upsert_default_model_profiles()
                        print("Default model profiles created/updated successfully.")
                    except Exception as e:
                        print(f"Error creating/updating default model profiles: {e}")
                else:
                    print("Failed to initialize database schema")
                    # If schema initialization failed, don't consider the storage initialized
                    storage.initialized = False
        except Exception as e:
            print(f"Error initializing database connection: {e}")
            print("Some features that depend on the database may not work properly")
            # We don't raise the exception here to allow the server to start
            # even if the database is not available. Routes will check for
            # initialization before accessing database components.
    else:
        print("\n" + "=" * 80)
        print("WARNING: NO DATABASE CONNECTION STRING AVAILABLE!")
        print("Database-dependent features will not work including:")
        print("  - User configuration")
        print("  - Conversation history")
        print("  - Memory features")
        print("  - Admin functions")
        print("=" * 80 + "\n")

        # Log critical information to help debug
        print("Database-related environment variables:")
        db_vars = [
            "DB_HOST",
            "DB_PORT",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
            "DB_CONNECTION_STRING",
        ]
        for var in db_vars:
            print(f"  {var}: {'✓' if os.environ.get(var) else '✗'}")
    # Initialize composer service
    try:
        from composer import initialize_composer

        await initialize_composer()
        logger.info("Composer service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize composer service: {e}")
        # Don't fail the entire service if composer fails
        print(f"Warning: Composer service initialization failed: {e}")

    print("Services initialization completed successfully!")
    yield
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

        # Stop database maintenance service if running
        try:
            print("Stopping database maintenance service...")
            await maintenance_service.stop_maintenance_schedule()
            print("Database maintenance service stopped")
        except Exception as e:
            print(f"Error stopping database maintenance service: {e}")

        # Stop composer service
        try:
            from composer import shutdown_composer

            await shutdown_composer()
            print("Composer service shutdown completed")
        except Exception as e:
            print(f"Error stopping composer service: {e}")

        # Stop vLLM service
        try:
            # await cleanup_vllm_service()
            print("vLLM service shutdown completed")
        except (RuntimeError, ValueError) as e:
            print(f"Error stopping vLLM service: {e}")

        cleanup_service.shutdown()


# Import logging for auth middleware debugging
from utils.logging import llmmllogger

# Initialize auth middleware first before creating the app
# This ensures middleware is ready before any routes are registered
from server.config import AUTH_JWKS_URI

print(f"Pre-initializing auth middleware with JWKS URI: {AUTH_JWKS_URI}")
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
from server.middleware.message_validation import MessageValidationMiddleware

app.add_middleware(MessageValidationMiddleware)


@app.middleware("http")
async def auth_middleware_handler(request: Request, call_next):
    """Authentication middleware to handle token validation and user identification"""
    # Get logger for debugging
    logger = llmmllogger.bind(component="auth-middleware")
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

# Add database initialization middleware
from server.middleware.db_init_middleware import db_init_middleware

app.middleware("http")(db_init_middleware)

# Include non-versioned routers (for backward compatibility)
app.include_router(images.router)
app.include_router(model.router)
app.include_router(chat.router)
app.include_router(conversation.router)
app.include_router(config.router)
app.include_router(static.router)
app.include_router(websockets.router)
app.include_router(users.router)

# Import and include the internal router
from server.routers import internal
from server.routers import db_admin

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
# Internal router is intentionally not versioned to maintain isolation
# app.include_router(internal.router, prefix=f"/{version}")

# Include OpenAI-compatible router
# app.include_router(openai_router, tags=["openai-compatible"])


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
