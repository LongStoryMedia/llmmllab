"""
Config router for handling user and system configuration.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /config/...
- Versioned: /v1/config/...
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import time
import uuid
import json

from server.auth import get_user_id, is_admin
from server.config import logger

# Import the storage layer
from server.db import storage

router = APIRouter(prefix="/config", tags=["config"])


class UserConfig(BaseModel):
    id: str
    user_id: str
    theme: Optional[str] = "light"
    language: Optional[str] = "en"
    message_display_mode: Optional[str] = "compact"
    notifications_enabled: Optional[bool] = True
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class UserConfigRequest(BaseModel):
    theme: Optional[str] = None
    language: Optional[str] = None
    message_display_mode: Optional[str] = None
    notifications_enabled: Optional[bool] = None


@router.get("/", response_model=UserConfig)
async def get_user_config(request: Request):
    """Get the user's configuration"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        logger.warning("Database not initialized, using mock user config")
        # Fallback to mock if database is not initialized
        return UserConfig(
            id=str(uuid.uuid4()),
            user_id=user_id,
            theme="light",
            language="en",
            message_display_mode="compact",
            notifications_enabled=True,
            created_at=time.time(),
            updated_at=time.time(),
        )

    try:
        # Get from database
        config_dict = await storage.user_config.get_user_config(user_id)

        if not config_dict:
            # User doesn't exist in database yet, create default config
            default_config = {
                "theme": "light",
                "language": "en",
                "message_display_mode": "compact",
                "notifications_enabled": True,
            }

            await storage.user_config.update_user_config(user_id, default_config)

            # Return default config with generated ID
            return UserConfig(
                id=str(uuid.uuid4()),
                user_id=user_id,
                **default_config,
                created_at=time.time(),
                updated_at=time.time(),
            )

        # Convert DB config format to our UserConfig format
        config_data = {}
        if "config" in config_dict and config_dict["config"]:
            if isinstance(config_dict["config"], str):
                config_data = json.loads(config_dict["config"])
            else:
                config_data = config_dict["config"]

        # Create UserConfig from the data
        return UserConfig(
            id=config_dict.get("id", str(uuid.uuid4())),
            user_id=user_id,
            theme=config_data.get("theme", "light"),
            language=config_data.get("language", "en"),
            message_display_mode=config_data.get("message_display_mode", "compact"),
            notifications_enabled=config_data.get("notifications_enabled", True),
            created_at=config_dict.get("created_at", time.time()),
            updated_at=config_dict.get("updated_at", time.time()),
        )

    except Exception as e:
        logger.error(f"Error getting user config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.put("/", response_model=UserConfig)
async def update_user_config(config_req: UserConfigRequest, request: Request):
    """Update the user's configuration"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        logger.warning("Database not initialized, using mock user config update")
        # Fallback to mock if database is not initialized
        current_config = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "theme": "light",
            "language": "en",
            "message_display_mode": "compact",
            "notifications_enabled": True,
            "created_at": time.time() - 3600,  # Created an hour ago
            "updated_at": time.time(),
        }

        # Update with requested changes (if provided)
        update_data = config_req.dict(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                current_config[key] = value

        return UserConfig(**current_config)

    try:
        # First, get current config to use as base
        current_config_dict = await storage.user_config.get_user_config(user_id)

        # Extract config data from the database format
        if not current_config_dict or not current_config_dict.get("config"):
            # No existing config, start with default
            config_data = {
                "theme": "light",
                "language": "en",
                "message_display_mode": "compact",
                "notifications_enabled": True,
            }
        else:
            # Parse existing config
            if isinstance(current_config_dict.get("config"), str):
                config_data = json.loads(current_config_dict["config"])
            else:
                config_data = current_config_dict.get("config", {})

        # Update with requested changes (if provided)
        update_data = config_req.dict(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                config_data[key] = value

        # Update in database
        await storage.user_config.update_user_config(user_id, config_data)

        # Return updated config
        return UserConfig(
            id=(
                current_config_dict.get("id", str(uuid.uuid4()))
                if current_config_dict
                else str(uuid.uuid4())
            ),
            user_id=user_id,
            **config_data,
            created_at=(
                current_config_dict.get("created_at", time.time())
                if current_config_dict
                else time.time()
            ),
            updated_at=time.time(),
        )

    except Exception as e:
        logger.error(f"Error updating user config: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
