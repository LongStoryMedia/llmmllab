"""
Models router for handling model management and configuration.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /models/...
- Versioned: /v1/models/...
"""

from typing import List, Optional, Any
from datetime import datetime
import uuid
import time
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import server.config as config
from server.auth import get_user_id, is_admin
from server.config import logger
from server.db import storage

from models.model_profile import ModelProfile, ModelParameters
from models.model import Model
from models.model_task import ModelTask
from models.model_details import ModelDetails


router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=List[Model])
async def list_models(request: Request):
    """List all available models."""
    # We're not currently using the user_id for filtering, but we may in the future
    _ = get_user_id(request)

    try:
        # Load models from JSON file
        with open(config.MODELS_CONFIG_PATH, "r") as f:
            models_data = json.load(f)

        # Convert to Model objects
        models = []
        for model_data in models_data:
            models.append(Model(**model_data))

        return models
    except Exception as e:
        logger.error(f"Error loading models from JSON: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error loading models: {str(e)}"
        ) from e


# Model profiles endpoints
@router.get("/profiles", response_model=List[ModelProfile])
async def list_model_profiles(request: Request):
    """List all model profiles for the authenticated user."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Get from database
        db_profiles = await storage.get_service(
            storage.model_profile
        ).list_model_profiles_by_user(user_id)

        return db_profiles or []

    except Exception as e:
        logger.error(f"Error listing model profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.get("/profiles/{profile_id}", response_model=ModelProfile)
async def get_model_profile_by_id(profile_id: str, request: Request):
    """Get a specific model profile by ID."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Convert the profile ID string to UUID
        profile_uuid = uuid.UUID(profile_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid profile ID format") from e

    try:
        # Get the profile from storage
        profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(profile_uuid, user_id)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile.user_id != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.post("/profiles", response_model=ModelProfile)
async def create_model_profile(profile_req: ModelProfile, request: Request):
    """Create a new model profile."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:

        # Save to database
        return await storage.get_service(storage.model_profile).create_model_profile(
            profile_req
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.put("/profiles/{profile_id}", response_model=ModelProfile)
async def update_model_profile(
    profile_id: str, profile_req: ModelProfile, request: Request
):
    """Update an existing model profile."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Convert the profile ID string to UUID
        profile_uuid = uuid.UUID(profile_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid profile ID format") from e

    try:
        # First check if profile exists and belongs to user
        profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(profile_uuid, user_id)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile.user_id != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")
        # Update in database
        return await storage.get_service(storage.model_profile).update_model_profile(
            profile_req
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.delete("/profiles/{profile_id}")
async def delete_model_profile(profile_id: str, request: Request):
    """Delete a model profile."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.model_profile:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        # Convert the profile ID string to UUID
        try:
            profile_uuid = uuid.UUID(profile_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid profile ID format")

        # First check if profile exists and belongs to user
        profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(profile_uuid, user_id)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile.user_id != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        await storage.get_service(storage.model_profile).delete_model_profile(
            profile_uuid, user_id
        )

        return {"status": "success", "message": "Profile deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
