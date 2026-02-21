"""
Lightweight FastAPI server for local-only endpoints (Ollama, OpenAI-compatible).
Runs on port 11435 for IDE integration.

Port 11434 is reserved for the ollama binary.
This provides IDE integration endpoints without authentication.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routers import ollama, openai
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="local_api_server")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Startup and shutdown logic for the local API server."""
    logger.info("Local API server (port 11435) starting up...")
    # Initialize DB storage like main app so routers can use storage services
    try:
        from db import storage  # pylint: disable=import-outside-toplevel
        from server.config import DB_CONNECTION_STRING  # pylint: disable=import-outside-toplevel

        if DB_CONNECTION_STRING:
            await storage.initialize(DB_CONNECTION_STRING)
            logger.info("Local API server: database storage initialized")
        else:
            logger.warning("Local API server: DB_CONNECTION_STRING not set; continuing without DB initialization")
    except Exception as e:
        logger.error(f"Local API server failed to initialize database storage: {e}")
        # Continue; endpoints that require DB will fail with clear errors

    try:
        yield
    finally:
        # Attempt to gracefully close storage
        try:
            from db import storage as _storage  # pylint: disable=import-outside-toplevel

            if getattr(_storage, "initialized", False):
                await _storage.close()
                logger.info("Local API server: database storage closed")
        except Exception:
            pass
        logger.info("Local API server shutting down...")


# Create the FastAPI app
app = FastAPI(
    title="Local API Server",
    description="Internal-only API for IDE integration (Ollama, OpenAI-compatible)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins since this port isn't exposed externally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers - these endpoints skip authentication
# Note: routers already have their prefixes defined (/api for ollama, /v1 for openai)
app.include_router(ollama.router)
app.include_router(openai.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "local_api_server"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Local API Server",
        "description": "Internal-only endpoints for IDE integration",
        "endpoints": {
            "ollama": "/api/* (model listing, chat, completions)",
            "openai": "/v1/* (OpenAI-compatible endpoints)",
            "health": "/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LOCAL_API_PORT", 11435))
    logger.info(f"Starting local API server on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=None,  # Use our own logging
    )
