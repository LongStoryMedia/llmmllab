"""
Config router for handling user and system configuration.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /config/...
- Versioned: /v1/config/...
"""

from fastapi import APIRouter, HTTPException, Request
from server.middleware.auth import get_user_id
from server.config import logger

# Import storage layer
from db import storage

# Import models - use the same imports as storage layer
from models.user_config import UserConfig
from models.default_configs import create_default_user_config

router = APIRouter(prefix="/config", tags=["config"])


def create_default_config(user_id: str) -> UserConfig:
    """Create a default user configuration with sensible defaults for all settings"""
    return create_default_user_config(user_id)


@router.get("/")
async def get_user_config(request: Request) -> UserConfig:
    """Get the user's configuration"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.user_config:
        raise HTTPException(
            status_code=503, detail="User config storage not initialized"
        )

    try:
        config = await storage.user_config.get_user_config(user_id)
        if not config:
            # Create new config with sensible defaults
            config = create_default_config(user_id)
            await storage.user_config.update_user_config(user_id, config)
        return config
    except Exception as e:
        logger.error(f"Error getting user config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.put("/")
async def update_config(config: UserConfig, request: Request) -> UserConfig:
    """Update the user's configuration"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.user_config:
        raise HTTPException(
            status_code=503, detail="User config storage not initialized"
        )

    if config.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot modify config for other users"
        )

    try:
        await storage.user_config.update_user_config(user_id, config)
        return config
    except Exception as e:
        logger.error(f"Error updating user config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
