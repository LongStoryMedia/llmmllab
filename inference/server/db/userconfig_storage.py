"""
Direct port of Maistro's userconfig.go storage logic to Python with cache integration.
"""

from typing import List, Optional
import asyncpg
from server.db.db_utils import typed_pool
import json
import logging
from models.user_config import UserConfig
from models.default_model_profiles import DEFAULT_MODEL_PROFILE_CONFIG
from utils.serialization import serialize_to_json
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
    create_default_user_config,
)
from server.db.cache_storage import cache_storage

logger = logging.getLogger(__name__)


class UserConfigStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def get_user_config(self, user_id: str) -> Optional[UserConfig]:
        # First try to get from cache
        cached_config = cache_storage.get_user_config_from_cache(user_id)
        if cached_config:
            return cached_config

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(self.get_query("user.get_config"), user_id)
            if not row:
                return None

            # Create config dictionary by merging row data with user_id
            config_data = dict(row)

            # Parse config if it's a JSON string
            if isinstance(config_data.get("config"), str):
                try:
                    import json

                    parsed_config = json.loads(config_data["config"])
                    config_data = parsed_config
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse config JSON: {e}")

            # Add the user_id to the config data
            config_data["user_id"] = user_id

            # Ensure all required fields have valid defaults
            self._ensure_required_fields(config_data)

            try:
                config = UserConfig(**config_data)

                # Cache the result for future use
                try:
                    cache_storage.cache_user_config(user_id, config)
                except Exception as e:
                    logger.warning(f"Failed to cache user config for {user_id}: {e}")

                return config
            except Exception as e:
                # If validation fails, return a default config with the user's ID
                logger.error(f"Error creating UserConfig from database: {e}")
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
        self._apply_defaults(
            config_data["gpu_config"], DEFAULT_GPU_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["workflow"], DEFAULT_WORKFLOW_CONFIG.dict()
        )
        self._apply_defaults(
            config_data["tool"], DEFAULT_TOOL_CONFIG.dict()
        )

    def _apply_defaults(self, target_dict: dict, defaults_dict: dict) -> None:
        """Apply default values from a defaults dictionary to a target dictionary"""
        for key, value in defaults_dict.items():
            if key not in target_dict:
                target_dict[key] = value

    async def update_user_config(self, user_id: str, cfg: UserConfig) -> None:
        # Make sure we're saving a valid config
        try:
            # Validate the config by ensuring it's a complete UserConfig instance
            # If the provided config is incomplete, we'll merge it with default values
            try:
                # Convert to model dict for JSON serialization
                config_dict = cfg.model_dump()
            except Exception as e:
                logger.warning(
                    f"Invalid config provided for user {user_id}, using defaults: {e}"
                )
                from server.routers.config import create_default_config

                default_config = create_default_config(user_id)
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

                # Invalidate cache to ensure we get fresh data on next read
                cache_storage.invalidate_user_config_cache(user_id)
        except Exception as e:
            logger.error(f"Error updating user config for user {user_id}: {e}")
            raise

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
