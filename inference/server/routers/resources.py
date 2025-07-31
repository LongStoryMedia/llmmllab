from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from models.model import Model
from models.requests import Malloc, ModelRequest, ModelsListResponse
# Make sure this imports the enhanced version
from services.hardware_manager import hardware_manager
import nvsmi

router = APIRouter(
    prefix="/resources",
    tags=["resources"]
)


class ProcessInfo(BaseModel):
    pid: int
    name: str
    memory_mb: int
    device_idx: int


class GPUProcessResponse(BaseModel):
    device_processes: Dict[int, List[ProcessInfo]]


class ClearMemoryRequest(BaseModel):
    device_idx: Optional[int] = None
    aggressive: bool = False
    nuclear: bool = False
    kill_processes: bool = False


class ClearMemoryResponse(BaseModel):
    detail: str
    memory_before: Dict[str, Any]
    memory_after: Dict[str, Any]
    processes_killed: Optional[Dict[int, int]] = None


@router.get("/malloc", response_model=Malloc)
async def get_malloc():
    """Get memory usage statistics for all devices."""
    try:
        # Update memory stats for all devices
        memory_stats = hardware_manager.update_all_memory_stats()

        # Log detailed GPU info for debugging
        for g in nvsmi.get_gpus():
            print(f"GPU {g.id}: {g.name} - {g.mem_used}MB/{g.mem_total}MB used")

        # Create response with device memory stats
        response = Malloc(devices=memory_stats)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving memory stats: {str(e)}"
        )


@router.get("/processes", response_model=GPUProcessResponse)
async def get_gpu_processes():
    """Get information about all processes currently using GPUs."""
    try:
        process_info = hardware_manager.get_gpu_process_info()

        # Convert to response format
        device_processes = {}
        for device_idx, processes in process_info.items():
            device_processes[device_idx] = [
                ProcessInfo(
                    pid=proc['pid'],
                    name=proc['name'],
                    memory_mb=proc['memory_mb'],
                    device_idx=proc['device_idx']
                )
                for proc in processes
            ]

        return GPUProcessResponse(device_processes=device_processes)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving GPU processes: {str(e)}"
        )


@router.post("/clear", response_model=ClearMemoryResponse)
async def clear_memory(request: ClearMemoryRequest = ClearMemoryRequest()):
    """
    Clear memory cache for devices with various levels of aggression.

    - aggressive: More thorough memory clearing
    - nuclear: Nuclear option - affects other processes
    - kill_processes: Kill other processes using the GPU (requires nuclear=True)
    """
    try:
        # Get memory stats before clearing
        memory_before = hardware_manager.update_all_memory_stats()

        processes_killed = None

        if request.nuclear:
            # Use nuclear clearing with optional process killing
            if request.kill_processes:
                # Get process info before killing
                process_info = hardware_manager.get_gpu_process_info()
                processes_killed = {}

                # Count processes per device
                for device_idx, processes in process_info.items():
                    processes_killed[device_idx] = len([
                        p for p in processes if p['pid'] != hardware_manager.os.getpid()
                    ])

            hardware_manager.nuclear_clear_memory(
                device_idx=request.device_idx,
                kill_processes=request.kill_processes
            )
            detail = f"Nuclear memory clearing completed (kill_processes={request.kill_processes})"
        else:
            # Use standard clearing
            hardware_manager.clear_memory(
                device_idx=request.device_idx,
                aggressive=request.aggressive
            )
            detail = f"Memory cache cleared (aggressive={request.aggressive})"

        # Get memory stats after clearing
        memory_after = hardware_manager.update_all_memory_stats()

        return ClearMemoryResponse(
            detail=detail,
            memory_before={k: {
                "mem_used": v.mem_used,
                "mem_free": v.mem_free,
                "mem_util": v.mem_util
            } for k, v in memory_before.items()},
            memory_after={k: {
                "mem_used": v.mem_used,
                "mem_free": v.mem_free,
                "mem_util": v.mem_util
            } for k, v in memory_after.items()},
            processes_killed=processes_killed
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing memory cache: {str(e)}"
        )


@router.post("/clear/nuclear")
async def nuclear_clear_memory(device_idx: Optional[int] = None, kill_processes: bool = True):
    """
    Nuclear option: Clear ALL memory from GPU(s), including other processes.

    WARNING: This will affect other processes using the GPU!
    """
    try:
        # Get memory and process info before clearing
        memory_before = hardware_manager.update_all_memory_stats()
        process_info_before = hardware_manager.get_gpu_process_info()

        # Perform nuclear clearing
        hardware_manager.nuclear_clear_memory(
            device_idx=device_idx,
            kill_processes=kill_processes
        )

        # Get stats after clearing
        memory_after = hardware_manager.update_all_memory_stats()
        process_info_after = hardware_manager.get_gpu_process_info()

        # Calculate processes killed
        processes_killed = {}
        for dev_idx in process_info_before.keys():
            before_count = len(process_info_before.get(dev_idx, []))
            after_count = len(process_info_after.get(dev_idx, []))
            processes_killed[dev_idx] = max(0, before_count - after_count)

        return {
            "detail": "Nuclear memory clearing completed",
            "device_idx": device_idx,
            "kill_processes": kill_processes,
            "memory_before": {k: f"{v.mem_used}MB/{v.mem_total}MB" for k, v in memory_before.items()},
            "memory_after": {k: f"{v.mem_used}MB/{v.mem_total}MB" for k, v in memory_after.items()},
            "processes_killed": processes_killed,
            "memory_freed_mb": {
                k: memory_before[k].mem_used - memory_after[k].mem_used
                for k in memory_before.keys() if k in memory_after
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during nuclear memory clearing: {str(e)}"
        )


@router.post("/reset/device/{device_idx}")
async def reset_device(device_idx: int):
    """Reset CUDA context for a specific device."""
    try:
        if device_idx < 0 or device_idx >= hardware_manager.gpu_count:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid device index. Available devices: 0-{hardware_manager.gpu_count-1}"
            )

        # Get memory before reset
        memory_before = hardware_manager.update_all_memory_stats()

        # Reset the device
        hardware_manager._reset_cuda_context(device_idx)

        # Get memory after reset
        memory_after = hardware_manager.update_all_memory_stats()

        return {
            "detail": f"Device {device_idx} reset completed",
            "memory_before": f"{memory_before[str(device_idx)].mem_used}MB/{memory_before[str(device_idx)].mem_total}MB",
            "memory_after": f"{memory_after[str(device_idx)].mem_used}MB/{memory_after[str(device_idx)].mem_total}MB"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error resetting device {device_idx}: {str(e)}"
        )


@router.get("/monitor/{duration_seconds}")
async def monitor_gpu_usage(duration_seconds: int = 60):
    """Monitor GPU usage for a specified duration."""
    try:
        if duration_seconds > 300:  # Limit to 5 minutes
            raise HTTPException(
                status_code=400,
                detail="Duration cannot exceed 300 seconds"
            )

        # This would ideally be run in a background task
        # For now, just return current stats
        current_stats = hardware_manager.update_all_memory_stats()

        return {
            "detail": f"Current GPU stats (monitoring for {duration_seconds}s would require background task)",
            "current_stats": {
                k: {
                    "mem_util": v.mem_util,
                    "gpu_util": v.gpu_util,
                    "temperature": v.temperature,
                    "mem_used": v.mem_used,
                    "mem_free": v.mem_free
                }
                for k, v in current_stats.items()
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error monitoring GPU usage: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Check the health of GPU devices and hardware manager."""
    try:
        stats = hardware_manager.update_all_memory_stats()
        process_info = hardware_manager.get_gpu_process_info()

        health_info = {
            "gpu_count": hardware_manager.gpu_count,
            "has_gpu": hardware_manager.has_gpu,
            "current_device": str(hardware_manager.current_device_idx),
            "devices": {},
            "total_processes": sum(len(procs) for procs in process_info.values())
        }

        for device_id, device_stats in stats.items():
            health_info["devices"][device_id] = {
                "name": device_stats.name,
                "temperature": device_stats.temperature,
                "memory_utilization": device_stats.mem_util,
                "gpu_utilization": device_stats.gpu_util,
                "processes": len(process_info.get(int(device_id), [])),
                "status": "healthy" if device_stats.temperature < 80 and device_stats.mem_util < 95 else "warning"
            }

        return health_info

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error checking GPU health: {str(e)}"
        )
