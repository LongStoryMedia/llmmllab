"""
User Storage - Handles user creation and management with proper default configuration setup.
"""

import asyncpg
from typing import Optional
from utils.logging import llmmllogger
from models.user_config import UserConfig
from models.default_configs import create_default_user_config
from db.db_utils import typed_pool
from .serialization import serialize_to_json

logger = llmmllogger.bind(component="UserStorage")


class UserStorage:
    """
    User storage service that ensures users are created with proper default configurations.
    """

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        logger.info("UserStorage initialized")

    async def ensure_user_exists(self, user_id: str) -> UserConfig:
        """
        Ensure a user exists with proper default configuration.
        
        If the user doesn't exist, creates them with default config.
        If the user exists but has no config, sets default config.
        Returns the user's configuration.
        """
        try:
            # Create default configuration for this user
            default_config = create_default_user_config(user_id)
            config_json = serialize_to_json(default_config.dict())

            async with self.typed_pool.acquire() as conn:
                # Create user with default config, or update config if user exists but has no config
                await conn.execute(
                    self.get_query("user.create_user_with_config"),
                    user_id,
                    config_json
                )
                
                logger.info(f"Ensured user exists with default config: {user_id}")
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to ensure user exists: {user_id}, error: {e}")
            raise

    async def get_user_config_from_users_table(self, user_id: str) -> Optional[UserConfig]:
        """
        Get user configuration directly from the users table.
        Returns None if user doesn't exist or has no config.
        """
        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT config FROM users WHERE id = $1",
                    user_id
                )
                
                if not row or not row['config']:
                    return None
                    
                config_data = dict(row['config'])
                config_data['user_id'] = user_id  # Ensure user_id is set
                
                return UserConfig(**config_data)
                
        except Exception as e:
            logger.error(f"Failed to get user config from users table: {user_id}, error: {e}")
            return None

    async def update_user_config_in_users_table(self, user_id: str, config: UserConfig) -> None:
        """
        Update user configuration in the users table.
        """
        try:
            config_json = serialize_to_json(config.dict())
            
            async with self.typed_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE users SET config = $2 WHERE id = $1",
                    user_id,
                    config_json
                )
                
                logger.info(f"Updated user config in users table: {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to update user config in users table: {user_id}, error: {e}")
            raise