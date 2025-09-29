"""
FastAPI application entry point for composer service.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import time
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from composer.core.service import ComposerService
from composer.config import config
from composer.monitoring.logging import composer_logger
from models.conversation_ctx import ConversationCtx
from pydantic import BaseModel


# Request/Response models
class ComposerRequest(BaseModel):
    """Request to compose workflow."""
    conversation_ctx: ConversationCtx
    workflow_type: str
    config_overrides: Optional[Dict[str, Any]] = None


class ComposerResponse(BaseModel):
    """Response from workflow composition."""
    workflow_id: str
    workflow_type: str
    success: bool
    message: Optional[str] = None


# Global service instance and start time
composer_service: Optional[ComposerService] = None
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global composer_service
    
    # Startup
    composer_logger.logger.info("Starting composer service")
    composer_service = ComposerService()
    yield
    
    # Shutdown
    composer_logger.logger.info("Shutting down composer service")
    if composer_service:
        await composer_service.shutdown()


# Create FastAPI application
app = FastAPI(
    title="Composer Service",
    description="LangGraph-based composer service for agentic workflows",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "composer",
        "version": "0.1.0"
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "service": "composer",
        "version": app.version,
        "uptime": time.time() - start_time,
        "host": config.service.host,
        "port": config.service.port,
        "debug": config.service.debug,
        "caching_enabled": config.default_workflow.enable_workflow_caching,
        "streaming_enabled": config.default_workflow.enable_streaming,
        "multi_agent_enabled": config.default_workflow.enable_multi_agent,
        "tool_generation_enabled": config.default_tool.enable_tool_generation,
        "cache_ttl": config.default_workflow.workflow_cache_ttl,
        "max_context_length": config.default_workflow.max_context_length
    }


@app.post("/compose", response_model=ComposerResponse)
async def compose_workflow(request: ComposerRequest):
    """
    Compose a workflow for the given conversation context.
    
    This is the main API endpoint for workflow composition.
    """
    if not composer_service:
        raise HTTPException(status_code=500, detail="Composer service not initialized")
    
    try:
        workflow = await composer_service.compose_workflow(
            request.conversation_ctx,
            request.workflow_type,
            request.config_overrides
        )
        
        # Generate workflow ID for tracking
        import hashlib
        import json
        workflow_data = {
            "type": request.workflow_type,
            "user_id": request.conversation_ctx.user_config.user_id if request.conversation_ctx.user_config else "anonymous",
            "nodes": workflow.nodes if hasattr(workflow, 'nodes') else []
        }
        workflow_id = hashlib.md5(json.dumps(workflow_data, sort_keys=True).encode()).hexdigest()[:12]
        
        return ComposerResponse(
            workflow_id=workflow_id,
            workflow_type=request.workflow_type,
            success=True,
            message="Workflow composed successfully"
        )
        
    except Exception as e:
        composer_logger.log_error(e, {"context": "compose_workflow_api"})
        raise HTTPException(status_code=500, detail=f"Failed to compose workflow: {str(e)}")


@app.post("/execute/{workflow_id}")
async def execute_workflow(workflow_id: str, request: ComposerRequest):
    """
    Execute a workflow with streaming support.
    
    This endpoint would be used for actual workflow execution.
    For Phase 1, this is a placeholder.
    """
    if not composer_service:
        raise HTTPException(status_code=500, detail="Composer service not initialized")
    
    try:
        # This is a placeholder for workflow execution
        # In Phase 3, this will implement actual streaming execution
        
        return {
            "workflow_id": workflow_id,
            "status": "execution_started",
            "message": "Workflow execution not yet implemented (Phase 3)"
        }
        
    except Exception as e:
        composer_logger.log_error(e, {"context": "execute_workflow_api"})
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    if not composer_service:
        raise HTTPException(status_code=500, detail="Composer service not initialized")
    
    try:
        # Get stats from various components
        stats = {
            "service": "composer",
            "version": "0.1.0",
            "cache_stats": {},
            "tool_stats": {}
        }
        
        # Get cache stats if available
        if composer_service.workflow_cache:
            cache_stats = await composer_service.workflow_cache.get_stats()
            stats["cache_stats"] = cache_stats
        
        # Get tool registry stats
        tool_stats = await composer_service.tool_registry.get_tool_stats()
        stats["tool_stats"] = tool_stats
        
        return stats
        
    except Exception as e:
        composer_logger.log_error(e, {"context": "get_stats_api"})
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.service.host,
        port=config.service.port,
        reload=config.service.reload,
        log_level="info"
    )