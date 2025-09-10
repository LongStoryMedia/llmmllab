"""
Pipeline management API for monitoring and controlling pipeline cache.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from runner.pipeline_factory import pipeline_factory, PipelinePriority
from server.auth import get_user_id, is_admin

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class PipelinePriorityUpdate(BaseModel):
    model_id: str
    priority: str  # Will be converted to PipelinePriority


@router.get("/stats")
async def get_pipeline_stats(request: Request) -> Dict[str, Any]:
    """Get detailed pipeline cache statistics."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Only admins can see detailed pipeline stats
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return pipeline_factory.get_cache_stats()


@router.get("/info")
async def get_pipeline_info(request: Request) -> Dict[str, Dict[str, Any]]:
    """Get information about all cached pipelines."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Only admins can see pipeline info
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return pipeline_factory.get_pipeline_info()


@router.post("/priority")
async def update_pipeline_priority(
    update: PipelinePriorityUpdate, request: Request
) -> Dict[str, str]:
    """Update the priority of a cached pipeline."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Only admins can update pipeline priorities
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        priority = PipelinePriority[update.priority.upper()]
    except KeyError:
        valid_priorities = [p.name for p in PipelinePriority]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid priority '{update.priority}'. Valid options: {valid_priorities}"
        )
    
    success = pipeline_factory.set_pipeline_priority(update.model_id, priority)
    
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Pipeline '{update.model_id}' not found in cache"
        )
    
    return {
        "status": "success",
        "message": f"Updated priority for {update.model_id} to {priority.name}"
    }


@router.post("/cleanup")
async def force_pipeline_cleanup(
    request: Request, target_free_gb: Optional[float] = None
) -> Dict[str, Any]:
    """Force cleanup of cached pipelines to free memory."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Only admins can force cleanup
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    target_bytes = None
    if target_free_gb is not None:
        target_bytes = target_free_gb * 1e9
    
    evicted_count = pipeline_factory.force_memory_cleanup(target_bytes)
    
    return {
        "status": "success",
        "evicted_pipelines": evicted_count,
        "message": f"Force cleanup evicted {evicted_count} pipelines"
    }


@router.delete("/cache/{model_id}")
async def evict_pipeline(model_id: str, request: Request) -> Dict[str, str]:
    """Evict a specific pipeline from cache."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Only admins can evict pipelines
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    pipeline_factory.clear_cache(model_id)
    
    return {
        "status": "success",
        "message": f"Evicted pipeline for model: {model_id}"
    }


@router.delete("/cache")
async def clear_all_cache(request: Request) -> Dict[str, str]:
    """Clear all cached pipelines."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Only admins can clear all cache
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    pipeline_factory.clear_cache()
    
    return {
        "status": "success",
        "message": "Cleared all cached pipelines"
    }