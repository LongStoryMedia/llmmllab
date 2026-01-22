"""
Multi-tier cache system for user configuration with in-memory → Redis → database fallback.
Implements LRU in-memory cache with TTL, Redis cache, and database storage with proper invalidation.
"""

import logging
import time
import threading
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from collections import OrderedDict

from models.user_config import UserConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and expiration tracking."""

    value: UserConfig
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at

    def access(self) -> UserConfig:
        """Access the cached value and update statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class InMemoryUserConfigCache:
    """
    LRU in-memory cache for user configurations with TTL.

    Features:
    - LRU eviction when max capacity reached
    - TTL-based expiration
    - Thread-safe operations
    - Access statistics tracking
    - Periodic cleanup of expired entries
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries to cache
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

        # Start background cleanup
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True
        )
        self._cleanup_thread.start()

    def get(self, user_id: str) -> Optional[UserConfig]:
        """Get user config from in-memory cache."""
        with self._lock:
            entry = self._cache.get(user_id)

            if entry is None:
                self._misses += 1
                logger.debug(f"Memory cache miss for user {user_id}")
                return None

            if entry.is_expired():
                # Remove expired entry
                del self._cache[user_id]
                self._misses += 1
                logger.debug(f"Memory cache expired for user {user_id}")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(user_id)
            self._hits += 1
            logger.debug(f"Memory cache hit for user {user_id}")
            return entry.access()

    def set(self, user_id: str, config: UserConfig, ttl: Optional[int] = None) -> None:
        """Set user config in in-memory cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl

            # Create new entry
            entry = CacheEntry(value=config, expires_at=expires_at)

            # Add to cache
            self._cache[user_id] = entry

            # Move to end (most recently used)
            self._cache.move_to_end(user_id)

            # Evict oldest entries if at capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key}")

            logger.debug(f"Cached user config for {user_id} (TTL: {ttl}s)")

    def invalidate(self, user_id: str) -> bool:
        """Remove user config from in-memory cache."""
        with self._lock:
            if user_id in self._cache:
                del self._cache[user_id]
                logger.debug(f"Invalidated memory cache for user {user_id}")
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cleared in-memory user config cache")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }

    def _periodic_cleanup(self) -> None:
        """Background cleanup of expired entries."""
        while True:
            try:
                time.sleep(60)  # Cleanup every minute

                with self._lock:
                    expired_keys = [
                        user_id
                        for user_id, entry in self._cache.items()
                        if entry.is_expired()
                    ]

                    for user_id in expired_keys:
                        del self._cache[user_id]

                    if expired_keys:
                        logger.debug(
                            f"Cleaned up {len(expired_keys)} expired cache entries"
                        )

            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")


class MultiTierUserConfigCache:
    """
    Multi-tier cache system: In-Memory → Redis → Database

    Provides three-tier caching with automatic fallback and proper invalidation.
    """

    def __init__(
        self,
        redis_cache,
        database_storage,
        memory_ttl: int = 300,
        redis_ttl: int = 1800,
    ):
        """
        Initialize multi-tier cache.

        Args:
            redis_cache: Redis cache storage instance
            database_storage: Database storage instance
            memory_ttl: In-memory cache TTL (5 minutes)
            redis_ttl: Redis cache TTL (30 minutes)
        """
        self.memory_cache = InMemoryUserConfigCache(default_ttl=memory_ttl)
        self.redis_cache = redis_cache
        self.database_storage = database_storage
        self.redis_ttl = redis_ttl

    async def get_user_config(self, user_id: str) -> Optional[UserConfig]:
        """
        Get user config with three-tier fallback:
        1. Try in-memory cache
        2. Try Redis cache
        3. Try database
        """
        # Tier 1: In-memory cache
        config = self.memory_cache.get(user_id)
        if config:
            logger.debug(f"User config retrieved from memory cache: {user_id}")
            return config

        # Tier 2: Redis cache
        if self.redis_cache:
            config = self.redis_cache.get_user_config_from_cache(user_id)
            if config:
                # Cache in memory for faster future access
                self.memory_cache.set(user_id, config)
                logger.debug(f"User config retrieved from Redis cache: {user_id}")
                return config

        # Tier 3: Database
        if self.database_storage:
            try:
                config = await self.database_storage.get_user_config(user_id)
                if config:
                    # Cache in both tiers for future access
                    self.memory_cache.set(user_id, config)
                    if self.redis_cache:
                        self.redis_cache.cache_user_config(user_id, config)
                    logger.debug(f"User config retrieved from database: {user_id}")
                    return config
            except Exception as e:
                logger.error(
                    f"Database error retrieving user config for {user_id}: {e}"
                )

        logger.warning(f"User config not found in any cache tier: {user_id}")
        return None

    async def set_user_config(self, user_id: str, config: UserConfig) -> None:
        """
        Update user config and invalidate all cache tiers.
        """
        # Update database first
        if self.database_storage:
            try:
                await self.database_storage.update_user_config(user_id, config)
            except Exception as e:
                logger.error(f"Database error updating user config for {user_id}: {e}")
                raise

        # Update all cache tiers with new config
        self.memory_cache.set(user_id, config)
        if self.redis_cache:
            self.redis_cache.cache_user_config(user_id, config)

        logger.info(f"Updated user config across all tiers: {user_id}")

    def invalidate_user_config(self, user_id: str) -> None:
        """
        Invalidate user config from all cache tiers.
        """
        # Invalidate all tiers
        memory_invalidated = self.memory_cache.invalidate(user_id)

        redis_invalidated = False
        if self.redis_cache:
            self.redis_cache.invalidate_user_config_cache(user_id)
            redis_invalidated = True

        logger.info(
            f"Invalidated user config - Memory: {memory_invalidated}, "
            f"Redis: {redis_invalidated}, User: {user_id}"
        )

    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()

        redis_enabled = self.redis_cache and self.redis_cache.is_storage_cache_enabled()

        return {
            "memory_cache": memory_stats,
            "redis_cache": {"enabled": redis_enabled, "connected": redis_enabled},
            "database": {"enabled": self.database_storage is not None},
        }

    def clear_all_caches(self) -> None:
        """Clear all cache tiers (for testing/debugging)."""
        self.memory_cache.clear()

        # Note: We don't clear Redis here as it might contain other data
        # Use redis_cache.clear_all_caches() if needed

        logger.info("Cleared memory cache (Redis cache preserved)")


# Global instance will be initialized by the storage system
multi_tier_cache: Optional[MultiTierUserConfigCache] = None


def initialize_multi_tier_cache(
    redis_cache, database_storage
) -> MultiTierUserConfigCache:
    """Initialize the global multi-tier cache instance."""
    global multi_tier_cache
    multi_tier_cache = MultiTierUserConfigCache(redis_cache, database_storage)
    logger.info("Initialized multi-tier user config cache")
    return multi_tier_cache


def get_multi_tier_cache() -> Optional[MultiTierUserConfigCache]:
    """Get the global multi-tier cache instance."""
    return multi_tier_cache
