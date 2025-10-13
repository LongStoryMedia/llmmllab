"""
Direct port of Maistro's userconfig.go storage logic to Python with cache integration.
"""

import json
import logging
import time
import threading
from typing import List, Optional, Dict
import asyncpg

from models.user_config import UserConfig
from models.default_model_profiles import DEFAULT_MODEL_PROFILE_CONFIG
from models.default_configs import (
    DEFAULT_PREFERENCES_CONFIG,
    DEFAULT_MEMORY_CONFIG,
    DEFAULT_SUMMARIZATION_CONFIG,
    DEFAULT_REFINEMENT_CONFIG,
    DEFAULT_WEB_SEARCH_CONFIG,
    DEFAULT_IMAGE_GENERATION_CONFIG,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    DEFAULT_GPU_CONFIG,
    DEFAULT_WORKFLOW_CONFIG,
    DEFAULT_TOOL_CONFIG,
    DEFAULT_CONTEXT_WINDOW_CONFIG,
    create_default_user_config,
)


from db.cache_storage import cache_storage
from db.db_utils import typed_pool
from .serialization import serialize_to_json

logger = logging.getLogger(__name__)


class UserConfigStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

        # Multi-tier cache will be initialized lazily to avoid circular dependency
        self.multi_tier_cache = None

        # Initialize in-memory cache
        self._memory_cache = None

        logger.info("UserConfigStorage initialized")

    def _create_memory_cache(self):
        """Create a simple in-memory cache for user configs."""
        return {
            "cache": {},  # user_id -> (config, expires_at, access_count)
            "max_size": 500,
            "default_ttl": 300,  # 5 minutes
            "hits": 0,
            "misses": 0,
            "lock": threading.RLock(),
        }

    def _memory_cache_get(self, user_id: str) -> Optional[UserConfig]:
        """Get config from memory cache with TTL and LRU."""
        if not hasattr(self, "_memory_cache") or not self._memory_cache:
            self._memory_cache = self._create_memory_cache()
            return None

        cache = self._memory_cache
        with cache["lock"]:
            if user_id not in cache["cache"]:
                cache["misses"] += 1
                return None

            config, expires_at, access_count = cache["cache"][user_id]

            # Check if expired
            if time.time() > expires_at:
                del cache["cache"][user_id]
                cache["misses"] += 1
                return None

            # Update access stats and move to end (LRU)
            cache["cache"][user_id] = (config, expires_at, access_count + 1)
            # Move to end by re-inserting (simple LRU simulation)
            cache["cache"][user_id] = cache["cache"].pop(user_id)
            cache["hits"] += 1
            return config

    def _memory_cache_set(
        self, user_id: str, config: UserConfig, ttl: Optional[int] = None
    ) -> None:
        """Set config in memory cache with TTL."""
        if not hasattr(self, "_memory_cache") or not self._memory_cache:
            return

        cache = self._memory_cache
        ttl = ttl or cache["default_ttl"]
        expires_at = time.time() + ttl

        with cache["lock"]:
            # Add new entry
            cache["cache"][user_id] = (config, expires_at, 1)

            # Evict oldest entries if over capacity (simple LRU)
            while len(cache["cache"]) > cache["max_size"]:
                # Remove first item (oldest)
                oldest_key = next(iter(cache["cache"]))
                del cache["cache"][oldest_key]

    def _memory_cache_invalidate(self, user_id: str) -> bool:
        """Remove config from memory cache."""
        if not hasattr(self, "_memory_cache") or not self._memory_cache:
            return False

        cache = self._memory_cache
        with cache["lock"]:
            if user_id in cache["cache"]:
                del cache["cache"][user_id]
                return True
            return False

    def _memory_cache_clear(self) -> None:
        """Clear all memory cache entries."""
        if not hasattr(self, "_memory_cache") or not self._memory_cache:
            return

        cache = self._memory_cache
        with cache["lock"]:
            cache["cache"].clear()
            cache["hits"] = 0
            cache["misses"] = 0

    def _memory_cache_stats(self) -> Dict:
        """Get memory cache statistics."""
        if not hasattr(self, "_memory_cache") or not self._memory_cache:
            return {"enabled": False}

        cache = self._memory_cache
        with cache["lock"]:
            total_requests = cache["hits"] + cache["misses"]
            hit_rate = cache["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                "enabled": True,
                "entries": len(cache["cache"]),
                "max_size": cache["max_size"],
                "hits": cache["hits"],
                "misses": cache["misses"],
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }

    async def get_user_config(self, user_id: str) -> UserConfig:
        """Get user config with multi-tier caching: memory → Redis → database."""
        # Initialize in-memory cache if not already done
        if not hasattr(self, "_memory_cache"):
            self._memory_cache = self._create_memory_cache()
            logger.debug("Initialized in-memory user config cache")

        # Tier 1: In-memory cache
        config = self._memory_cache_get(user_id)
        if config:
            logger.debug(f"User config retrieved from memory cache: {user_id}")
            return config

        # Tier 2: Redis cache
        config = cache_storage.get_user_config_from_cache(user_id)
        if config:
            # Cache in memory for faster future access
            self._memory_cache_set(user_id, config)
            logger.debug(f"User config retrieved from Redis cache: {user_id}")
            return config

        # Tier 3: Database
        config = await self._get_user_config_from_database(user_id)
        if config:
            # Cache in both tiers for future access
            self._memory_cache_set(user_id, config)
            try:
                cache_storage.cache_user_config(user_id, config)
            except Exception as e:
                logger.warning(
                    f"Failed to cache user config in Redis for {user_id}: {e}"
                )
            logger.debug(f"User config retrieved from database: {user_id}")

        return config

    async def _get_user_config_from_database(self, user_id: str) -> UserConfig:
        """Direct database access for user config (used by multi-tier cache and as fallback)."""
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(self.get_query("user.get_config"), user_id)
            if not row:
                logger.info(
                    f"No user config found in database for {user_id}, creating default"
                )
                return create_default_user_config(user_id)

            # Create config dictionary by merging row data with user_id
            config_data = dict(row)

            # Parse config if it's a JSON string
            if isinstance(config_data.get("config"), str):
                try:
                    parsed_config = json.loads(config_data["config"])
                    config_data = parsed_config
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse config JSON for {user_id}: {e}")

            # Add the user_id to the config data
            config_data["user_id"] = user_id

            # Ensure all required fields have valid defaults
            self._ensure_required_fields(config_data)

            try:
                config = UserConfig(**config_data)
                logger.debug(
                    f"Successfully loaded user config from database: {user_id}"
                )
                return config
            except Exception as e:
                # If validation fails, return a default config with the user's ID
                logger.error(
                    f"Error creating UserConfig from database for {user_id}: {e}"
                )
                return create_default_user_config(user_id)

    def _ensure_required_fields(self, config_data: dict) -> None:
        """Ensure all required fields have valid defaults"""
        # Ensure all model components have at least empty dictionaries
        for field in [
            "preferences",
            "memory",
            "summarization",
            "web_search",
            "refinement",
            "image_generation",
            "model_profiles",
            "circuit_breaker",
            "gpu_config",
            "workflow",
            "tool",
            "context_window",  # Added missing context_window field
        ]:
            if field not in config_data or not isinstance(config_data[field], dict):
                config_data[field] = {}

        # Apply defaults from predefined config objects
        self._apply_defaults(
            config_data["preferences"], DEFAULT_PREFERENCES_CONFIG.dict()
        )
        self._apply_defaults(config_data["memory"], DEFAULT_MEMORY_CONFIG.dict())
        self._apply_defaults(
            config_data["summarization"], DEFAULT_SUMMARIZATION_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["refinement"], DEFAULT_REFINEMENT_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["web_search"], DEFAULT_WEB_SEARCH_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["image_generation"], DEFAULT_IMAGE_GENERATION_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["model_profiles"], DEFAULT_MODEL_PROFILE_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["circuit_breaker"], DEFAULT_CIRCUIT_BREAKER_CONFIG.dict()
        )
        self._apply_defaults(config_data["gpu_config"], DEFAULT_GPU_CONFIG.dict())
        self._apply_defaults(config_data["workflow"], DEFAULT_WORKFLOW_CONFIG.dict())
        self._apply_defaults(config_data["tool"], DEFAULT_TOOL_CONFIG.dict())
        self._apply_defaults(
            config_data["context_window"], DEFAULT_CONTEXT_WINDOW_CONFIG.dict()
        )

    def _apply_defaults(self, target_dict: dict, defaults_dict: dict) -> None:
        """Apply default values from a defaults dictionary to a target dictionary"""
        for key, value in defaults_dict.items():
            if key not in target_dict:
                target_dict[key] = value

    async def update_user_config(self, user_id: str, cfg: UserConfig) -> None:
        """Update user config in database and invalidate all cache tiers."""
        try:
            # Update database first
            await self._update_user_config_in_database(user_id, cfg)

            # Invalidate all cache tiers
            self.invalidate_user_config_cache(user_id)

            # Cache the updated config in all tiers
            self._memory_cache_set(user_id, cfg)
            try:
                cache_storage.cache_user_config(user_id, cfg)
            except Exception as e:
                logger.warning(
                    f"Failed to cache updated user config in Redis for {user_id}: {e}"
                )

            logger.info(f"Updated user config across all cache tiers: {user_id}")

        except Exception as e:
            logger.error(f"Error updating user config for user {user_id}: {e}")
            raise

    async def _update_user_config_in_database(
        self, user_id: str, cfg: UserConfig
    ) -> None:
        """Direct database update for user config (used by multi-tier cache and as fallback)."""
        # Validate the config by ensuring it's a complete UserConfig instance
        try:
            # Convert to model dict for JSON serialization
            config_dict = cfg.model_dump()
        except Exception as e:
            logger.warning(
                f"Invalid config provided for user {user_id}, using defaults: {e}"
            )
            # Use the already imported create_default_user_config

            default_config = create_default_user_config(user_id)
            config_dict = default_config.model_dump()

        # We don't need to store user_id in the config field as it's already the primary key
        if "user_id" in config_dict:
            del config_dict["user_id"]

        # Additional check to ensure all required fields are present
        config_data = dict(config_dict)
        self._ensure_required_fields(config_data)

        # Serialize the complete config to JSON with proper object handling
        config_json = serialize_to_json(config_data)

        # Save to database
        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("user.update_config"), config_json, user_id
            )
            logger.debug(f"Successfully updated user config in database: {user_id}")

    async def get_all_users(self) -> List[dict]:
        # This is an admin operation and doesn't need caching
        async with self.typed_pool.acquire() as conn:
            try:
                rows = await conn.fetch(self.get_query("user.get_all_users"))
                users = []

                for row in rows:
                    user_dict = dict(row)
                    user_id = user_dict.get("id", "unknown")

                    # Process config if it exists
                    if "config" in user_dict and user_dict["config"]:
                        try:
                            config_dict = {}

                            # Handle string JSON configs
                            if isinstance(user_dict["config"], str):
                                try:
                                    config_dict = json.loads(user_dict["config"])
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Failed to parse config JSON for user {user_id}: {e}"
                                    )
                                    config_dict = {}
                            elif isinstance(user_dict["config"], dict):
                                config_dict = user_dict["config"]

                            # Ensure user_id is included in the config
                            config_dict["user_id"] = user_id

                            # Ensure all required fields have defaults
                            self._ensure_required_fields(config_dict)

                            # Create a proper UserConfig instance and convert back to dict
                            try:
                                # Make sure all needed fields have proper values before creating the UserConfig instance
                                self._ensure_required_fields(config_dict)
                                user_dict["config"] = UserConfig(
                                    **config_dict
                                ).model_dump()
                            except Exception as e:
                                logger.warning(
                                    f"Failed to create UserConfig for user {user_id}: {e}"
                                )
                                from server.routers.config import create_default_config

                                user_dict["config"] = create_default_config(
                                    user_id
                                ).model_dump()
                        except Exception as e:
                            logger.warning(
                                f"Failed to process config for user {user_id}: {e}"
                            )
                            # Use empty config as fallback
                            from server.routers.config import create_default_config

                            user_dict["config"] = create_default_config(
                                user_id
                            ).model_dump()
                    else:
                        # No config or empty config, use default
                        from server.routers.config import create_default_config

                        user_dict["config"] = create_default_config(
                            user_id
                        ).model_dump()

                    users.append(user_dict)

                return users
            except Exception as e:
                logger.error(f"Error fetching all users: {str(e)}")
                return []

    def get_cache_stats(self) -> dict:
        """Get comprehensive cache statistics for monitoring."""
        memory_stats = self._memory_cache_stats()

        redis_enabled = cache_storage.is_storage_cache_enabled()

        return {
            "memory_cache": memory_stats,
            "redis_cache": {"enabled": redis_enabled, "connected": redis_enabled},
            "database": {"enabled": True},
        }

    def invalidate_user_config_cache(self, user_id: str) -> None:
        """Invalidate user config from all cache tiers."""
        # Invalidate memory cache
        memory_invalidated = self._memory_cache_invalidate(user_id)

        # Invalidate Redis cache
        redis_invalidated = False
        try:
            cache_storage.invalidate_user_config_cache(user_id)
            redis_invalidated = True
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache for {user_id}: {e}")

        logger.info(
            f"Invalidated user config - Memory: {memory_invalidated}, "
            f"Redis: {redis_invalidated}, User: {user_id}"
        )

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache for testing/debugging."""
        self._memory_cache_clear()
        logger.info("Cleared in-memory user config cache")
