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


def db_to_api_model_profile(db_profile: Any, user_id: str) -> ModelProfile:
    """Convert a database model profile to an API model profile."""
    # Handle dict or ModelProfile object
    if isinstance(db_profile, ModelProfile):
        return db_profile

    # Convert to dict if needed
    profile_dict = db_profile.dict() if hasattr(db_profile, "dict") else db_profile
    if not isinstance(profile_dict, dict):
        profile_dict = {}

    # Convert string UUID to UUID object if needed
    profile_id = profile_dict.get("id", uuid.uuid4())
    if isinstance(profile_id, str):
        profile_id = uuid.UUID(profile_id)

    # Extract parameters from JSON if needed
    params = {}
    if "parameters" in profile_dict:
        try:
            if isinstance(profile_dict["parameters"], str):
                params = json.loads(profile_dict["parameters"])
            elif isinstance(profile_dict["parameters"], dict):
                params = profile_dict["parameters"]
        except Exception as e:
            logger.warning(f"Error parsing parameters: {e}")
            params = {}
    elif "temperature" in profile_dict:
        # Handle old format where parameters were flat
        params = {
            "temperature": profile_dict.get("temperature", 0.7),
            "top_p": profile_dict.get("top_p", 1.0),
            "num_predict": profile_dict.get("max_tokens", 1024),
            "frequency_penalty": profile_dict.get("frequency_penalty", 0.0),
            "presence_penalty": profile_dict.get("presence_penalty", 0.0),
        }

    current_time = datetime.now()

    # Create ModelProfile object with appropriate defaults
    return ModelProfile(
        id=profile_id,
        user_id=profile_dict.get("user_id", user_id),
        name=profile_dict.get("name", ""),
        description=profile_dict.get("description", None),
        model_name=profile_dict.get(
            "model_name", profile_dict.get("model_id", "")
        ),  # Handle old format
        parameters=(
            ModelParameters(**params)
            if params
            else ModelParameters(temperature=0.7, top_p=1.0, num_predict=1024)
        ),
        system_prompt=profile_dict.get("system_prompt", ""),
        created_at=profile_dict.get("created_at", current_time),
        updated_at=profile_dict.get("updated_at", current_time),
        model_version=profile_dict.get("model_version", None),
        type=profile_dict.get("type", 0),  # Default to PRIMARY type
    )


class ModelProfileRequest(BaseModel):
    name: str
    model_id: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    is_default: Optional[bool] = False


class ModelProfileList(BaseModel):
    profiles: List[ModelProfile]


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

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profiles")
        profiles = [
            ModelProfile(
                id=uuid.uuid4(),
                user_id=user_id,
                name="Default Profile",
                model_name="gpt-4",
                parameters=ModelParameters(
                    temperature=0.7, top_p=1.0, num_predict=1024, top_k=40
                ),
                system_prompt="You are a helpful assistant.",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                type=1,  # Default type
            )
        ]
        return profiles

    try:
        # Get from database
        db_profiles = await storage.model_profile.list_model_profiles_by_user(user_id)

        if not db_profiles:
            # No profiles found, return empty list
            return []

        profiles = []
        for db_profile in db_profiles:
            profile = db_to_api_model_profile(db_profile, user_id)
            profiles.append(profile)

        return profiles

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
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile ID format")

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile")
        current_time = datetime.now()
        return ModelProfile(
            id=profile_uuid,
            user_id=user_id,
            name="Default Profile",
            model_name="gpt-4",  # TODO: Get from config
            parameters=ModelParameters(temperature=0.7, top_p=1.0, num_predict=1024),
            system_prompt="You are a helpful assistant.",
            created_at=current_time,
            updated_at=current_time,
            type=0,  # PRIMARY type
        )

    try:
        # Get the profile from storage
        profile = await storage.model_profile.get_model_profile(profile_uuid)

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

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile")
        current_time = datetime.now()
        return ModelProfile(
            id=profile_uuid,  # Use the validated UUID
            user_id=user_id,
            name="Default Profile",
            model_name="gpt-4",  # TODO: Fix model_id vs model_name discrepancy
            system_prompt="You are a helpful assistant.",
            parameters=ModelParameters(temperature=0.7, top_p=1.0, num_predict=1024),
            created_at=current_time,
            updated_at=current_time,
            type=0,  # PRIMARY type
        )
        return profile

    try:
        # Get from database
        db_profile = await storage.model_profile.get_model_profile(
            uuid.UUID(profile_id)
        )

        if not db_profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if (
            db_profile.user_id != user_id
            and not str.startswith(str(db_profile.id), "00000000-0000-0000-0000")
            and not is_admin(request)
        ):
            raise HTTPException(status_code=403, detail="Access denied")

        # Convert DB model to API model
        profile = db_to_api_model_profile(db_profile, user_id)

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.post("/profiles", response_model=ModelProfile)
async def create_model_profile(profile_req: ModelProfileRequest, request: Request):
    """Create a new model profile."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile creation")
        profile_id = str(uuid.uuid4())
        current_time = time.time()

        now = datetime.now()
        profile = ModelProfile(
            id=uuid.UUID(profile_id),
            user_id=user_id,
            name=profile_req.name,
            description=None,
            model_name=profile_req.model_id,  # Map model_id to model_name
            parameters=ModelParameters(
                temperature=profile_req.temperature,
                top_p=profile_req.top_p,
                num_predict=profile_req.max_tokens,
            ),
            system_prompt=profile_req.system_prompt or "",
            created_at=now,
            updated_at=now,
            model_version=None,
            type=0,  # PRIMARY type
        )

        return profile

    try:
        # Create parameter dictionary
        parameters = ModelParameters(
            temperature=profile_req.temperature,
            top_p=profile_req.top_p,
            num_predict=profile_req.max_tokens,
        )

        # Save to database
        profile_id = await storage.model_profile.create_model_profile(
            user_id=user_id,
            name=profile_req.name,
            description="",  # No description field in our model
            model_name=profile_req.model_id,
            parameters=parameters,
            system_prompt=(
                profile_req.system_prompt if profile_req.system_prompt else ""
            ),
            model_version="",  # No version field in our model
            profile_type="default",  # No type field in our model
        )

        if not profile_id:
            raise HTTPException(status_code=500, detail="Failed to create profile")

        current_time = time.time()

        # Get the created profile from database
        try:
            profile_uuid = uuid.UUID(profile_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid profile ID format")

        db_profile = await storage.model_profile.get_model_profile(profile_uuid)

        if not db_profile:
            # If we can't get it from DB, this is an error
            raise HTTPException(status_code=500, detail="Failed to get created profile")

        return db_profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.put("/profiles/{profile_id}", response_model=ModelProfile)
async def update_model_profile(
    profile_id: str, profile_req: ModelProfileRequest, request: Request
):
    """Update an existing model profile."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Convert the profile ID string to UUID
        profile_uuid = uuid.UUID(profile_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile ID format")

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile update")
        current_time = time.time()

        profile = ModelProfile(
            id=uuid.UUID(str(profile_id)),
            user_id=user_id,
            name=profile_req.name,
            model_name=profile_req.model_id,  # TODO: Fix model_id vs model_name discrepancy
            system_prompt=profile_req.system_prompt
            or "",  # Handle optional system prompt
            parameters=ModelParameters(
                temperature=profile_req.temperature,
                top_p=profile_req.top_p,
                num_predict=profile_req.max_tokens,
            ),
            created_at=datetime.fromtimestamp(
                time.time() - 3600
            ),  # Mock - created an hour ago
            updated_at=datetime.fromtimestamp(current_time),
            type=0,  # TODO: Get proper type from request
        )

        return profile

    try:
        # First check if profile exists and belongs to user
        profile = await storage.model_profile.get_model_profile(profile_uuid)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile.user_id != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        # Update profile fields
        update_profile = ModelProfile(
            id=profile_uuid,
            user_id=user_id,
            name=profile_req.name,
            model_name=profile_req.model_id,  # TODO: Fix model_id vs model_name discrepancy
            parameters=ModelParameters(
                temperature=profile_req.temperature,
                top_p=profile_req.top_p,
                num_predict=profile_req.max_tokens,
            ),
            system_prompt=profile_req.system_prompt or "",
            created_at=profile.created_at,
            updated_at=datetime.now(),
            type=profile.type,  # Keep the existing type
        )

        # Update in database
        await storage.model_profile.update_model_profile(mp=update_profile)

        # Retrieve and return the updated profile
        updated_profile = await storage.model_profile.get_model_profile(profile_uuid)

        if not updated_profile:
            raise HTTPException(
                status_code=404, detail="Profile not found after update"
            )

        return updated_profile

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
        profile = await storage.model_profile.get_model_profile(profile_uuid)

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile.user_id != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        await storage.model_profile.delete_model_profile(profile_uuid)

        return {"status": "success", "message": "Profile deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
