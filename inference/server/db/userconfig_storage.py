"""
Direct port of Maistro's userconfig.go storage logic to Python with cache integration.
"""

from typing import List, Optional, Dict, Any
import asyncpg
from server.db.db_utils import typed_pool
import json
import logging
from models.user_config import UserConfig
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

            config = UserConfig(**dict(row))

            # Cache the result for future use
            try:
                cache_storage.cache_user_config(user_id, config)
            except Exception as e:
                logger.warning(f"Failed to cache user config for {user_id}: {e}")

            return config

    async def update_user_config(self, user_id: str, cfg: UserConfig) -> None:
        config_json = json.dumps(cfg)
        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("user.update_config"), config_json, user_id
            )

            cache_storage.invalidate_user_config_cache(user_id)

    async def get_all_users(self) -> List[dict]:
        # This is an admin operation and doesn't need caching
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, created_at, username FROM users")
            return [dict(row) for row in rows]
