"""
Direct port of Maistro's modelprofile.go storage logic to Python with cache integration.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import asyncpg
from inference.models.model import Model
from inference.models.model_parameters import ModelParameters
from inference.server.db.db_utils import typed_pool
import json
import logging
from datetime import datetime
from models.model_profile import ModelProfile
from inference.server.db.cache_storage import cache_storage

logger = logging.getLogger(__name__)


class ModelProfileStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def create_model_profile(
        self,
        user_id: str,
        name: str,
        description: str,
        model_name: str,
        parameters: ModelParameters,
        system_prompt: str,
        model_version: str,
        profile_type: str,
    ) -> Optional[str]:
        parameters_json = json.dumps(parameters)
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("modelprofile.create_profile"),
                user_id,
                name,
                description,
                model_name,
                parameters_json,
                system_prompt,
                model_version,
                profile_type,
            )
            return str(row["id"]) if row and "id" in row else None

    async def get_model_profile(self, profile_id: UUID) -> Optional[ModelProfile]:
        # First try to get from cache
        cached_profile = cache_storage.get_model_profile_from_cache(profile_id)
        if cached_profile:
            return cached_profile

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("modelprofile.get_profile_by_id"), profile_id
            )
            if not row:
                return None

            profile = ModelProfile(**dict(row))

            # Cache the result for future use
            try:
                cache_storage.cache_model_profile(profile)
            except Exception as e:
                logger.warning(f"Failed to cache model profile {profile_id}: {e}")

            return profile

    async def update_model_profile(self, mp: ModelProfile) -> None:
        parameters_json = json.dumps(mp.parameters)
        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("modelprofile.update_profile"),
                mp.id,
                mp.name,
                mp.description,
                mp.model_name,
                parameters_json,
                mp.system_prompt,
                mp.model_version,
                mp.type,
            )

        # Update the cache - first get the current profile
        profile = cache_storage.get_model_profile_from_cache(mp.id)
        if profile:
            # Update the profile with new values
            cache_storage.cache_model_profile(profile)
        else:
            # If not in cache, invalidate to ensure next get will fetch from DB
            cache_storage.invalidate_model_profile_cache(mp.id)

        # Invalidate the user's model profiles list cache
        # Since we don't have user_id here, we'll have to get it first
        if profile:
            user_id = profile.user_id
            if user_id:
                cache_storage.invalidate_model_profiles_list_cache(user_id)

    async def delete_model_profile(self, profile_id: UUID) -> None:
        # Get user ID before deleting for cache invalidation
        profile = cache_storage.get_model_profile_from_cache(profile_id)
        user_id = profile.user_id if profile else None

        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("modelprofile.delete_profile"), profile_id
            )

        # Invalidate the profile cache
        cache_storage.invalidate_model_profile_cache(profile_id)

        # Invalidate user's model profiles list cache if we have the user_id
        if user_id:
            cache_storage.invalidate_model_profiles_list_cache(user_id)

    async def list_model_profiles_by_user(self, user_id: str) -> List[ModelProfile]:
        # First try to get from cache
        cached_profiles = cache_storage.get_model_profiles_list_from_cache(user_id)
        if cached_profiles is not None:
            return cached_profiles

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("modelprofile.list_profiles_by_user"), user_id
            )
            return [ModelProfile(**dict(row)) for row in rows]
