"""
Workflow caching with TTL.
Caches compiled LangGraph workflows to avoid repeated compilation.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, Callable, List

from models import Tool, WorkflowType
from models.config import Config as config
from utils.logging import llmmllogger


class CacheEntry:
    """Represents a cached workflow with TTL."""

    def __init__(self, workflow: Any, ttl_seconds: int):
        self.workflow = workflow
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.last_accessed = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def access(self) -> Any:
        """Access the cached workflow and update stats."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.workflow


class WorkflowCache:
    """
    Workflow caching with TTL and LRU eviction.
    Caches compiled LangGraph workflows by (user_config, workflow_type, tools) signature.
    """

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl or 300  # 5 minutes default TTL
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _get_user_config(self, user_id: str):
        """Get user configuration from shared data layer."""
        try:
            from db import storage  # pylint: disable=import-outside-toplevel

            # Initialize storage if not done
            if not storage.pool:
                llmmllogger.logger.warning("Database not initialized for WorkflowCache")
                return None

            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)
            if not user_config:
                llmmllogger.logger.warning(
                    f"No user config found for {user_id} in WorkflowCache"
                )
                return None
            return user_config
        except Exception as e:
            llmmllogger.logger.error(
                f"Failed to get user config for {user_id} in WorkflowCache: {e}"
            )
            return None

    async def get_cache_key(
        self, user_id: str, workflow_type: WorkflowType, tools: List[Tool]
    ) -> str:
        """
        Generate cache key from user_id, workflow type, and tools.
        Configuration is retrieved from shared data layer using user_id.

        The cache key uniquely identifies a workflow configuration to enable
        safe reuse across requests with identical parameters.
        """
        # Get user configuration from shared data layer
        user_config = await self._get_user_config(user_id)

        # Create deterministic representation
        key_data = {
            "user_id": user_id,
            "workflow_type": workflow_type.name,
            "model_profile": (
                user_config.model_profiles.primary_profile_id
                if user_config and user_config.model_profiles
                else "default"
            ),
            "tools": sorted([tool.name for tool in tools if tool.name]),
            "preferences": (
                user_config.preferences.dict()
                if user_config and user_config.preferences
                else {}
            ),
        }

        # Generate hash
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]

    async def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve workflow from cache if not expired."""
        async with self._lock:
            entry = self.cache.get(cache_key)

            if entry is None:
                llmmllogger.log_cache_operation("miss", cache_key)
                return None

            if entry.is_expired:
                # Remove expired entry
                del self.cache[cache_key]
                llmmllogger.log_cache_operation("evict", cache_key)
                return None

            llmmllogger.log_cache_operation("hit", cache_key)
            return entry.access()

    async def set(
        self, cache_key: str, workflow: Any, ttl_seconds: Optional[int] = None
    ) -> None:
        """Store workflow in cache with TTL."""
        async with self._lock:
            # Evict oldest entries if at capacity
            if len(self.cache) >= self.max_size:
                await self._evict_lru()

            ttl = ttl_seconds or self.default_ttl
            self.cache[cache_key] = CacheEntry(workflow, ttl)
            llmmllogger.log_cache_operation("set", cache_key)

    async def get_or_create(
        self,
        cache_key: str,
        factory_fn: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
    ) -> Any:
        """Get from cache or create using factory function."""
        # Try cache first
        workflow = await self.get(cache_key)
        if workflow is not None:
            return workflow

        # Create new workflow
        workflow = await factory_fn()
        await self.set(cache_key, workflow, ttl_seconds)
        return workflow

    async def invalidate(self, cache_key: str) -> bool:
        """Remove specific entry from cache."""
        async with self._lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                llmmllogger.log_cache_operation("evict", cache_key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached workflows."""
        async with self._lock:
            self.cache.clear()
            llmmllogger.log_cache_operation("clear", "all")

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return

        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)

        del self.cache[lru_key]
        llmmllogger.log_cache_operation("evict", lru_key)

    async def _periodic_cleanup(self) -> None:
        """Periodically remove expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes

                async with self._lock:
                    expired_keys = [
                        key for key, entry in self.cache.items() if entry.is_expired
                    ]

                    for key in expired_keys:
                        del self.cache[key]
                        llmmllogger.log_cache_operation("evict", key)

            except asyncio.CancelledError:
                break
            except Exception as e:
                llmmllogger.log_error(e, {"context": "cache_cleanup"})

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_entries = len(self.cache)
            total_accesses = sum(entry.access_count for entry in self.cache.values())

            return {
                "total_entries": total_entries,
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "hit_rate": (
                    0.0
                    if total_accesses == 0
                    else (total_accesses - total_entries) / total_accesses
                ),
            }

    async def close(self) -> None:
        """Clean up cache resources."""
        if hasattr(self, "_cleanup_task"):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.clear()
