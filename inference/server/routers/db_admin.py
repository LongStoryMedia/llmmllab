"""
Internal API endpoints for database maintenance and administration.
"""

from fastapi import APIRouter, HTTPException, Request
from server.middleware.auth import is_admin
from db.maintenance import maintenance_service

router = APIRouter(
    prefix="/internal/db",
    tags=["database"],
    # Access control will be handled by the auth middleware
)


@router.get("/maintenance/status")
async def get_maintenance_status():
    """Get the current status of the database maintenance service"""
    if not maintenance_service:
        raise HTTPException(
            status_code=503, detail="Database maintenance service not available"
        )

    return {
        "status": maintenance_service.status,
    }


@router.post("/maintenance/run")
async def trigger_maintenance(request: Request):
    """Manually trigger a database maintenance run"""
    if not maintenance_service:
        raise HTTPException(
            status_code=503, detail="Database maintenance service not available"
        )

    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        success = await maintenance_service.perform_maintenance()
        if success:
            return {
                "status": "success",
                "message": "Database maintenance completed successfully",
            }
        return {
            "status": "warning",
            "message": "Database maintenance completed with errors, check logs for details",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to run database maintenance: {str(e)}"
        ) from e
