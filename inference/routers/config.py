"""
Config router for handling user and system configuration.

"""

from fastapi import APIRouter, HTTPException, Request
from middleware.auth import get_user_id
from config import logger

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
    if not storage.initialized or not storage.user_config:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        # This will now automatically create the user with default config if they don't exist
        config = await storage.user_config.get_user_config(user_id)
        return config
    except Exception as e:
        logger.error(f"Error getting user config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.put("/")
async def update_config(config: UserConfig, request: Request) -> UserConfig:
    """Update the user's configuration"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.user_config:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if config.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="Cannot modify config for other users"
        )

    try:
        await storage.user_config.update_user_config(user_id, config)
        return config
    except Exception as e:
        logger.error(f"Error updating user config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
