from typing import List

from fastapi import APIRouter, HTTPException, status

from models.model import Model
from models.requests import Malloc, ModelRequest, ModelsListResponse
from services.hardware_manager import hardware_manager
from services.lora_service import lora_service
import nvsmi

router = APIRouter(
    prefix="/resources",
    tags=["resources"]
)


@router.get("/malloc", response_model=Malloc)
async def get_malloc():
    """Get memory usage statistics for all devices."""
    try:
        # Update memory stats for all devices
        memory_stats = hardware_manager.update_all_memory_stats()
        for g in nvsmi.get_gpus():
            print(g)
        # Create response with device memory stats
        response = Malloc(devices=memory_stats)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving memory stats: {str(e)}"
        )


@router.post("/clear")
async def clear_memory():
    """Clear memory cache for all devices."""
    try:
        hardware_manager.clear_memory()
        return {"detail": "Memory cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing memory cache: {str(e)}"
        )
