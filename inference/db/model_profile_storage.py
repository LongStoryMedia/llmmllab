"""
Storage implementation for ModelProfile objects.
"""

import uuid
from datetime import datetime
import json
import logging
from typing import List, Optional
import asyncpg
from models.model_profile import ModelProfile
from models.default_model_profiles import DEFAULT_PROFILES
from db.db_utils import typed_pool
from .serialization import serialize_to_json

logger = logging.getLogger(__name__)


class ModelProfileStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def get_model_profile_by_id(
        self, profile_id: uuid.UUID, user_id: str
    ) -> Optional[ModelProfile]:
        """Get a model profile by ID"""
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("modelprofile.get_profile_by_id"), profile_id, user_id
            )
            if not row:
                # If the profile is a system default, return it
                for profile in DEFAULT_PROFILES.values():
                    if profile.id == profile_id:
                        return profile
                return None

            try:
                profile_data = dict(row)

                # Parse parameters if stored as JSON string
                if isinstance(profile_data.get("parameters"), str):
                    try:
                        profile_data["parameters"] = json.loads(
                            profile_data["parameters"]
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse parameters JSON for profile {profile_id}: {e}"
                        )

                # Parse circuit_breaker if stored as JSON string
                if isinstance(profile_data.get("circuit_breaker"), str):
                    try:
                        circuit_breaker_data = json.loads(
                            profile_data["circuit_breaker"]
                        )
                        # Keep as dictionary - let Pydantic deserialize it
                        profile_data["circuit_breaker"] = circuit_breaker_data
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.error(
                            f"Failed to parse circuit_breaker JSON for profile {profile_id}: {e}"
                        )
                        profile_data["circuit_breaker"] = None

                # Parse gpu_config if stored as JSON string
                if isinstance(profile_data.get("gpu_config"), str):
                    try:
                        gpu_config_data = json.loads(profile_data["gpu_config"])
                        # Keep as dictionary - let Pydantic deserialize it
                        profile_data["gpu_config"] = gpu_config_data
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.error(
                            f"Failed to parse gpu_config JSON for profile {profile_id}: {e}"
                        )
                        profile_data["gpu_config"] = None

                return ModelProfile(**profile_data)
            except Exception as e:
                logger.error(f"Error creating ModelProfile from database: {e}")
                return None

    async def list_model_profiles_by_user(self, user_id: str) -> List[ModelProfile]:
        """List all model profiles for a user"""
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("modelprofile.list_profiles_by_user"), user_id
            )
            profiles = []

            # Add default system profiles
            for profile in DEFAULT_PROFILES.values():
                # Create a copy with the user's ID
                user_profile = ModelProfile(
                    id=profile.id,
                    user_id=user_id,
                    name=profile.name,
                    description=profile.description,
                    model_name=profile.model_name,
                    parameters=profile.parameters,
                    system_prompt=profile.system_prompt,
                    created_at=profile.created_at,
                    updated_at=profile.updated_at,
                    model_version=profile.model_version,
                    type=profile.type,
                )
                profiles.append(user_profile)

            # Add user's custom profiles
            for row in rows:
                try:
                    profile_data = dict(row)

                    # Parse parameters if stored as JSON string
                    if isinstance(profile_data.get("parameters"), str):
                        try:
                            profile_data["parameters"] = json.loads(
                                profile_data["parameters"]
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse parameters JSON for profile: {e}"
                            )
                            continue

                    # Parse circuit_breaker if stored as JSON string
                    if isinstance(profile_data.get("circuit_breaker"), str):
                        try:
                            circuit_breaker_data = json.loads(
                                profile_data["circuit_breaker"]
                            )
                            # Keep as dictionary - let Pydantic deserialize it
                            profile_data["circuit_breaker"] = circuit_breaker_data
                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.error(
                                f"Failed to parse circuit_breaker JSON for profile: {e}"
                            )
                            profile_data["circuit_breaker"] = None

                    # Parse gpu_config if stored as JSON string
                    if isinstance(profile_data.get("gpu_config"), str):
                        try:
                            gpu_config_data = json.loads(profile_data["gpu_config"])
                            # Keep as dictionary - let Pydantic deserialize it
                            profile_data["gpu_config"] = gpu_config_data
                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.error(
                                f"Failed to parse gpu_config JSON for profile: {e}"
                            )
                            profile_data["gpu_config"] = None

                    profiles.append(ModelProfile(**profile_data))
                except Exception as e:
                    logger.error(f"Error creating ModelProfile from database: {e}")

            return profiles

    async def create_model_profile(self, profile: ModelProfile) -> ModelProfile:
        """Create a new model profile"""
        # Don't allow overwriting system default profiles (profiles with system UUIDs and user_id="system")
        if profile.user_id == "system" and any(
            profile.id == default_profile.id
            for default_profile in DEFAULT_PROFILES.values()
        ):
            raise ValueError("Cannot modify system default profiles")

        # Generate a new ID if not provided
        if not profile.id:
            profile.id = uuid.uuid4()

        # Serialize parameters to JSON if needed with advanced object handling
        if profile.parameters:
            params_dict = profile.parameters.model_dump()
            params_json = serialize_to_json(params_dict)
        else:
            params_json = "{}"

        # Serialize circuit_breaker to JSON if provided
        if profile.circuit_breaker:
            circuit_breaker_dict = profile.circuit_breaker.model_dump()
            circuit_breaker_json = serialize_to_json(circuit_breaker_dict)
        else:
            circuit_breaker_json = None

        # Serialize GPU config
        if profile.gpu_config:
            gpu_config_dict = profile.gpu_config.model_dump()
            gpu_config_json = serialize_to_json(gpu_config_dict)
        else:
            gpu_config_json = None

        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("modelprofile.create_profile"),
                profile.id,
                profile.user_id,
                profile.name,
                profile.description,
                profile.model_name,
                params_json,
                profile.system_prompt,
                profile.model_version,
                profile.type,
                circuit_breaker_json,
                gpu_config_json,
            )

            # Update the profile with current timestamps (which are set by the database)
            profile.created_at = datetime.now()
            profile.updated_at = profile.created_at

            return profile

    async def update_model_profile(self, profile: ModelProfile) -> ModelProfile:
        """Update an existing model profile"""
        # Don't allow modifying system default profiles (profiles with system UUIDs and user_id="system")
        if profile.user_id == "system" and any(
            profile.id == default_profile.id
            for default_profile in DEFAULT_PROFILES.values()
        ):
            raise ValueError("Cannot modify system default profiles")

        # Serialize parameters to JSON if needed with advanced object handling
        if profile.parameters:
            params_dict = profile.parameters.model_dump()
            params_json = serialize_to_json(params_dict)
        else:
            params_json = "{}"

        # Serialize circuit_breaker to JSON if provided
        if profile.circuit_breaker:
            circuit_breaker_dict = profile.circuit_breaker.model_dump()
            circuit_breaker_json = serialize_to_json(circuit_breaker_dict)
        else:
            circuit_breaker_json = None

        # Serialize GPU config
        if profile.gpu_config:
            gpu_config_dict = profile.gpu_config.model_dump()
            gpu_config_json = serialize_to_json(gpu_config_dict)
        else:
            gpu_config_json = None

        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("modelprofile.update_profile"),
                profile.id,
                profile.name,
                profile.description,
                profile.model_name,
                params_json,
                profile.system_prompt,
                profile.model_version,
                profile.type,
                circuit_breaker_json,
                gpu_config_json,
                profile.user_id,
            )

            # Update the profile with current timestamp (which is set by the database)
            profile.updated_at = datetime.now()

            return profile

    async def delete_model_profile(self, profile_id: uuid.UUID, user_id: str) -> bool:
        """Delete a model profile"""
        # Don't allow deleting system default profiles
        if any(
            profile_id == default_profile.id
            for default_profile in DEFAULT_PROFILES.values()
        ):
            raise ValueError("Cannot delete system default profiles")

        async with self.typed_pool.acquire() as conn:
            result = await conn.execute(
                self.get_query("modelprofile.delete_profile"), profile_id, user_id
            )

            return "DELETE" in result

    async def upsert_default_model_profiles(self) -> None:
        """Create or update default model profiles"""

        async with self.typed_pool.acquire() as conn:
            for profile in DEFAULT_PROFILES.values():
                # Serialize parameters to JSON if needed with advanced object handling
                if profile.parameters:
                    params_dict = profile.parameters.model_dump()
                    params_json = serialize_to_json(params_dict)
                else:
                    params_json = "{}"
                await conn.execute(
                    self.get_query("modelprofile.upsert_default_profile"),
                    profile.id,
                    profile.user_id,
                    profile.name,
                    profile.description,
                    profile.model_name,
                    params_json,
                    profile.system_prompt,
                    profile.model_version,
                    profile.type,
                )
