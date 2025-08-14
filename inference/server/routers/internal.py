"""
Internal router for internal API endpoints with restricted access.

Note: This router is included in app.py with non-versioned path only.
It is intentionally not versioned to maintain isolation of internal endpoints.
"""

import os
import logging
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import FileResponse
from server.config import IMAGE_DIR, INTERNAL_ALLOWED_IPS, INTERNAL_API_KEY, logger
import ipaddress

router = APIRouter(prefix="/internal", tags=["internal"])


def verify_internal_api_key(request: Request):
    """Verify that the request has a valid internal API key."""
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Also check if the request comes from an allowed IP
    client_host = request.client.host if request.client else "unknown"
    if not is_ip_allowed(client_host):
        logger.warning(f"Unauthorized internal API access attempt from {client_host}")
        raise HTTPException(status_code=403, detail="Access denied")

    return True


def is_ip_allowed(ip: str) -> bool:
    """Check if an IP is in the allowed CIDR ranges."""
    if not INTERNAL_ALLOWED_IPS:
        return False

    allowed_ranges = INTERNAL_ALLOWED_IPS.split(",")

    try:
        client_ip = ipaddress.ip_address(ip)
        for allowed_range in allowed_ranges:
            network = ipaddress.ip_network(allowed_range.strip(), strict=False)
            if client_ip in network:
                return True
        return False
    except ValueError:
        # If IP parsing fails, deny access
        return False


@router.get("/image/{user_id}/{filename}")
async def internal_get_user_image(
    user_id: str,
    filename: str,
    request: Request,
    _: bool = Depends(verify_internal_api_key),
):
    """
    Get a user's image for internal services without requiring user auth.
    This is used by other services to access images securely.
    """
    # Validate user ID and filename to prevent directory traversal attacks
    if user_id == "" or filename == "" or "/" in user_id or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid user ID or filename")

    # Path to the locally stored image
    file_path = os.path.join(IMAGE_DIR, "originals", user_id, filename)

    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Add security headers to prevent browser caching and embedding
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "X-Content-Type-Options": "nosniff",
    }

    # Log the access
    logger.info(f"Internal API accessed image: user={user_id}, file={filename}")

    # Determine content type based on file extension
    content_type = "image/png"
    if filename.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"

    # Serve the file directly with appropriate content type
    return FileResponse(file_path, media_type=content_type, headers=headers)
