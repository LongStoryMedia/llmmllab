"""
Models router for handling model management and configuration.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /models/...
- Versioned: /v1/models/...
"""

from typing import List, Optional, Any, Union
from datetime import datetime
import uuid
import time
import json

from fastapi import APIRouter, HTTPException, Request

import server.config as config
from server.middleware.auth import get_user_id, is_admin
from server.config import logger
from db import storage

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
        # Import here to avoid circular imports
        from models.default_model_profiles import DEFAULT_PROFILES

        # Handle UUID generation and validation
        if profile_req.id is None:
            # Generate a new UUID if none provided
            profile_req.id = uuid.uuid4()
            logger.info(f"Generated new UUID {profile_req.id} for new user profile")
        else:
            # Check if the user is trying to create a profile with a system default UUID
            is_default_uuid = any(
                profile_req.id == default_profile.id
                for default_profile in DEFAULT_PROFILES.values()
            )

            if is_default_uuid:
                # Generate a new UUID for user-created profile based on default
                profile_req.id = uuid.uuid4()
                logger.info(
                    f"Generated new UUID {profile_req.id} for user profile based on default"
                )

        # Ensure the profile is associated with the current user
        profile_req.user_id = user_id

        # Set timestamps
        profile_req.created_at = datetime.now()
        profile_req.updated_at = datetime.now()

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
        existing_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(profile_uuid, user_id)

        if not existing_profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if existing_profile.user_id != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        # Convert the request to a ModelProfile object
        profile_data = profile_req.dict(exclude_unset=True)

        # Maintain the original ID and user_id
        profile_data["id"] = profile_uuid
        profile_data["user_id"] = existing_profile.user_id
        profile_data["created_at"] = existing_profile.created_at
        profile_data["updated_at"] = datetime.now()

        # Create the ModelProfile object
        profile = ModelProfile(**profile_data)

        # Update in database
        return await storage.get_service(storage.model_profile).update_model_profile(
            profile
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
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
