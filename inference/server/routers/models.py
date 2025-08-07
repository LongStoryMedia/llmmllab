from typing import List, Optional, Any
import uuid
import time
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


# Define a simplified model structure for the API
class ModelAPI(BaseModel):
    id: str
    name: str
    model_type: str
    provider: str


from server.auth import get_user_id, is_admin
from server.config import logger
from server.db import storage


# Define custom model profile models for API
class ModelProfile(BaseModel):
    id: str
    user_id: str
    name: str
    model_id: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    is_default: Optional[bool] = False


def db_to_api_model_profile(db_profile: Any, user_id: str) -> ModelProfile:
    """Convert a database model profile to an API model profile."""
    # Handle dict or ModelProfile object
    profile_dict = db_profile
    if hasattr(db_profile, "dict"):
        profile_dict = db_profile.dict()

    # Extract parameters from JSON if needed
    params = {}
    if isinstance(profile_dict, dict) and "parameters" in profile_dict:
        if isinstance(profile_dict["parameters"], str):
            try:
                params = json.loads(profile_dict["parameters"])
            except Exception:
                params = {}
        else:
            params = profile_dict.get("parameters", {})

    # Create ModelProfile object with appropriate defaults
    if isinstance(profile_dict, dict):
        return ModelProfile(
            id=profile_dict.get("id", str(uuid.uuid4())),
            user_id=profile_dict.get("user_id", user_id),
            name=profile_dict.get("name", ""),
            model_id=profile_dict.get(
                "model_name", ""
            ),  # model_name in DB maps to model_id in API
            system_prompt=profile_dict.get("system_prompt", ""),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_tokens", 1024),
            frequency_penalty=params.get("frequency_penalty", 0.0),
            presence_penalty=params.get("presence_penalty", 0.0),
            created_at=profile_dict.get("created_at", time.time()),
            updated_at=profile_dict.get("updated_at", time.time()),
            is_default=profile_dict.get("is_default", False),
        )
    else:
        # Fallback for non-dict objects
        return ModelProfile(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name="",
            model_id="",
            system_prompt="",
            temperature=0.7,
            top_p=1.0,
            max_tokens=1024,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            created_at=time.time(),
            updated_at=time.time(),
            is_default=False,
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


@router.get("/", response_model=List[ModelAPI])
async def list_models(request: Request):
    """List all available models."""
    # We're not currently using the user_id for filtering, but we may in the future
    _ = get_user_id(request)

    try:
        # Simple mock implementation - in a real system this would come from a model registry
        # or be fetched from the database
        return [
            ModelAPI(id="gpt-4", name="GPT-4", model_type="llm", provider="openai"),
            ModelAPI(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                model_type="llm",
                provider="openai",
            ),
            ModelAPI(
                id="dalle-3", name="DALL-E 3", model_type="diffusion", provider="openai"
            ),
        ]

        # # Safely filter models by checking if specialization attribute exists
        # filtered_models = []
        # for model in models:
        #     filtered_models.append(model)

        # # Create response
        # response = ModelsListResponse(models=filtered_models, active_model="")
        # return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing models: {str(e)}"
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
                id=str(uuid.uuid4()),
                user_id=user_id,
                name="Default Profile",
                model_id="gpt-4",
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                top_p=1.0,
                max_tokens=1024,
                created_at=time.time(),
                updated_at=time.time(),
                is_default=True,
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
async def get_model_profile(profile_id: str, request: Request):
    """Get a specific model profile by ID."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile")
        profile = ModelProfile(
            id=profile_id,
            user_id=user_id,
            name="Default Profile",
            model_id="gpt-4",
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            top_p=1.0,
            max_tokens=1024,
            created_at=time.time(),
            updated_at=time.time(),
            is_default=True,
        )
        return profile

    try:
        # Get from database
        db_profile = await storage.model_profile.get_model_profile(profile_id)

        if not db_profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if db_profile.get("user_id") != user_id and not is_admin(request):
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

        profile = ModelProfile(
            id=profile_id,
            user_id=user_id,
            name=profile_req.name,
            model_id=profile_req.model_id,
            system_prompt=profile_req.system_prompt,
            temperature=profile_req.temperature,
            top_p=profile_req.top_p,
            max_tokens=profile_req.max_tokens,
            frequency_penalty=profile_req.frequency_penalty,
            presence_penalty=profile_req.presence_penalty,
            created_at=current_time,
            updated_at=current_time,
            is_default=profile_req.is_default,
        )

        return profile

    try:
        # Create parameter dictionary
        parameters = {
            "temperature": profile_req.temperature,
            "top_p": profile_req.top_p,
            "max_tokens": profile_req.max_tokens,
            "frequency_penalty": profile_req.frequency_penalty,
            "presence_penalty": profile_req.presence_penalty,
        }

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
        db_profile = await storage.model_profile.get_model_profile(profile_id)

        if not db_profile:
            # If we can't get it from DB, create a minimal profile
            profile = ModelProfile(
                id=profile_id,
                user_id=user_id,
                name=profile_req.name,
                model_id=profile_req.model_id,
                system_prompt=profile_req.system_prompt,
                temperature=profile_req.temperature,
                top_p=profile_req.top_p,
                max_tokens=profile_req.max_tokens,
                frequency_penalty=profile_req.frequency_penalty,
                presence_penalty=profile_req.presence_penalty,
                created_at=current_time,
                updated_at=current_time,
                is_default=profile_req.is_default,
            )
        else:
            profile = db_to_api_model_profile(db_profile, user_id)

        return profile

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

    if not storage.initialized or not storage.model_profile:
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile update")
        current_time = time.time()

        profile = ModelProfile(
            id=profile_id,
            user_id=user_id,
            name=profile_req.name,
            model_id=profile_req.model_id,
            system_prompt=profile_req.system_prompt,
            temperature=profile_req.temperature,
            top_p=profile_req.top_p,
            max_tokens=profile_req.max_tokens,
            frequency_penalty=profile_req.frequency_penalty,
            presence_penalty=profile_req.presence_penalty,
            created_at=time.time() - 3600,  # Mock - created an hour ago
            updated_at=current_time,
            is_default=profile_req.is_default,
        )

        return profile

    try:
        # First check if profile exists and belongs to user
        profile_dict = await storage.model_profile.get_model_profile(profile_id)

        if not profile_dict:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile_dict.get("user_id") != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        # Create parameter dictionary
        parameters = {
            "temperature": profile_req.temperature,
            "top_p": profile_req.top_p,
            "max_tokens": profile_req.max_tokens,
            "frequency_penalty": profile_req.frequency_penalty,
            "presence_penalty": profile_req.presence_penalty,
        }

        # Update in database
        await storage.model_profile.update_model_profile(
            profile_id=profile_id,
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

        current_time = time.time()

        # Get updated profile from database
        updated_db_profile = await storage.model_profile.get_model_profile(profile_id)

        if not updated_db_profile:
            # If we can't get it from DB, create a minimal profile with the request data
            profile = ModelProfile(
                id=profile_id,
                user_id=user_id,
                name=profile_req.name,
                model_id=profile_req.model_id,
                system_prompt=profile_req.system_prompt,
                temperature=profile_req.temperature,
                top_p=profile_req.top_p,
                max_tokens=profile_req.max_tokens,
                frequency_penalty=profile_req.frequency_penalty,
                presence_penalty=profile_req.presence_penalty,
                created_at=profile_dict.get("created_at", time.time() - 3600),
                updated_at=current_time,
                is_default=profile_req.is_default,
            )
        else:
            profile = db_to_api_model_profile(updated_db_profile, user_id)

        return profile

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
        # Fallback to mock if database is not initialized
        logger.warning("Database not initialized, using mock model profile deletion")
        return {"status": "success", "message": f"Profile {profile_id} deleted"}

    try:
        # First check if profile exists and belongs to user
        profile_dict = await storage.model_profile.get_model_profile(profile_id)

        if not profile_dict:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Check if profile belongs to user or user is admin
        if profile_dict.get("user_id") != user_id and not is_admin(request):
            raise HTTPException(status_code=403, detail="Access denied")

        # Delete from database
        await storage.model_profile.delete_model_profile(profile_id)

        return {"status": "success", "message": f"Profile {profile_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model profile: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e
