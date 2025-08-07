"""
cache_storage.py: Python port of maistro/storage/cache.go
Implements Redis-based caching for messages, summaries, conversations, model profiles, and user configs.

Class Usage:
    1. Use the singleton instance: from server.db.cache_storage import cache_storage
    2. Initialize Redis connection: cache_storage.init_storage_cache(redis_url)
    3. Use the methods to cache and retrieve objects

Note for Developers:
    All Redis operations have been made safe using the _safe_redis_call method.
    This pattern must be followed when adding new Redis operations:

    # Instead of:
    self.redis_client.some_method(arg1, arg2)

    # Use:
    self._safe_redis_call('some_method', arg1, arg2)

    This handles the case when Redis client is None and catches all exceptions.
"""

import logging
import time
import threading
from typing import Optional, List, Any, Type, Callable
from uuid import UUID

# Set up logger
logger = logging.getLogger(__name__)
import redis
from models.message import Message
from models.summary import Summary
from models.conversation import Conversation
from models.model_profile import ModelProfile
from models.user_config import UserConfig

__all__ = ["cache_storage", "CacheStorage"]


class CacheStorage:
    """
    Redis-based caching for storage operations.
    """

    # Cache key prefixes for different object types
    MESSAGE_KEY_PREFIX = "llmmll:message:"
    SUMMARY_KEY_PREFIX = "llmmll:summary:"
    CONVERSATION_KEY_PREFIX = "llmmll:conversation:"
    CONVERSATION_LIST_PREFIX = "llmmll:conversations:"
    MESSAGES_LIST_PREFIX = "llmmll:messages:"
    SUMMARIES_LIST_PREFIX = "llmmll:summaries:"
    USERCONFIG_KEY_PREFIX = "llmmll:userconfig:"
    MODELPROFILE_KEY_PREFIX = "llmmll:modelprofile:"
    MODELPROFILES_LIST_PREFIX = "llmmll:modelprofiles:"

    def __init__(self, redis_url: Optional[str] = None, timeout: int = 5):
        """Initialize the CacheStorage instance."""
        self.redis_client: Optional[redis.Redis] = None
        self._health_check_thread = None

        if redis_url:
            self.init_storage_cache(redis_url, timeout)

    def cache_key(self, prefix: str, key_id: Any) -> str:
        """Helper to construct cache keys."""
        return f"{prefix}{key_id}"

    def init_storage_cache(self, redis_url: str, timeout: int = 5):
        """Initialize the Redis client for storage caching."""
        self.redis_client = redis.Redis.from_url(redis_url, socket_timeout=timeout)
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise RuntimeError(f"Redis connection failed: {e}") from e
        # Optionally clear all keys for a fresh start
        # self.redis_client.flushdb()
        # Start a background thread to monitor Redis health
        self._health_check_thread = threading.Thread(
            target=self._start_redis_health_check, daemon=True
        )
        self._health_check_thread.start()

    def _start_redis_health_check(self):
        """Periodically check the Redis connection."""
        while True:
            try:
                if self.redis_client:
                    self.redis_client.ping()
            except redis.ConnectionError:
                pass
            time.sleep(30)

    def is_storage_cache_enabled(self) -> bool:
        """Check if the storage cache is enabled and connected."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except redis.RedisError as e:
            logger.warning(f"Redis connection error during ping: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Redis connection: {str(e)}")
            return False

    def _safe_redis_call(self, method_name: str, *args, **kwargs):
        """Safely call a Redis client method, handling the case when Redis client is None."""
        if not self.redis_client:
            return None
        try:
            method = getattr(self.redis_client, method_name)
            return method(*args, **kwargs)
        except redis.RedisError as e:
            logger.warning(f"Redis error in {method_name}: {str(e)}")
            return None
        except AttributeError as e:
            logger.error(f"Invalid Redis method {method_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Redis call {method_name}: {str(e)}")
            return None

    # Helper methods for common Redis operations with a generic approach
    def _get_from_cache(
        self, prefix: str, object_id: Any, model_class: Type
    ) -> Optional[Any]:
        """Generic method to get an object from cache."""
        if not self.is_storage_cache_enabled():
            return None
        key = self.cache_key(prefix, object_id)
        data = self._safe_redis_call("get", key)
        if not data:
            return None
        try:
            return model_class.parse_raw(
                data.decode() if isinstance(data, bytes) else str(data)
            )
        except ValueError as e:
            logger.warning(
                f"Error parsing data for {model_class.__name__} with ID {object_id}: {str(e)}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving {model_class.__name__} with ID {object_id}: {str(e)}"
            )
            return None

    def _cache_object(self, prefix: str, obj: Any, expire: int = 86400) -> None:
        """Generic method to cache an object."""
        if not self.is_storage_cache_enabled() or not obj:
            return
        key = self.cache_key(prefix, obj.id)
        self._safe_redis_call("set", key, obj.json(), ex=expire)

    def _invalidate_cache(self, prefix: str, object_id: Any) -> None:
        """Generic method to invalidate an object cache."""
        if not self.is_storage_cache_enabled():
            return
        key = self.cache_key(prefix, object_id)
        self._safe_redis_call("delete", key)

    def _get_list_from_cache(
        self,
        list_prefix: str,
        parent_id: Any,
        item_getter: Callable,
        item_id_processor: Callable = int,
    ) -> Optional[List[Any]]:
        """Generic method to get a list of objects from cache."""
        if not self.is_storage_cache_enabled():
            return None
        list_key = self.cache_key(list_prefix, parent_id)
        try:
            ids = self._safe_redis_call("lrange", list_key, 0, -1)
            if not ids:
                return None
            items = []
            for item_id in ids:
                id_val = (
                    item_id.decode() if isinstance(item_id, bytes) else str(item_id)
                )
                try:
                    # Process the ID and get the item
                    processed_id = item_id_processor(id_val)
                    item = item_getter(processed_id)
                    if item:
                        items.append(item)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing ID {id_val}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error retrieving item {id_val}: {str(e)}")
                    continue
            return items if items else None
        except Exception as e:
            logger.error(f"Error fetching list from cache: {str(e)}")
            return None

    def _cache_list(
        self, list_prefix: str, parent_id: Any, items: List[Any], item_cacher
    ) -> None:
        """Generic method to cache a list of objects."""
        if not self.is_storage_cache_enabled() or not items:
            return
        list_key = self.cache_key(list_prefix, parent_id)
        self._safe_redis_call("delete", list_key)
        for item in items:
            item_cacher(item)

    # ========== Message Cache Operations ==========
    def get_message_from_cache(self, message_id: int) -> Optional[Message]:
        """Get a message from cache by ID."""
        return self._get_from_cache(self.MESSAGE_KEY_PREFIX, message_id, Message)

    def cache_message(self, message: Message):
        """Cache a message."""
        if not message:
            return
        # Cache the message object
        self._cache_object(self.MESSAGE_KEY_PREFIX, message)
        # Also cache the message ID in the conversation's message list
        if message.conversation_id:
            list_key = self.cache_key(
                self.MESSAGES_LIST_PREFIX, message.conversation_id
            )
            self._safe_redis_call("rpush", list_key, str(message.id))

    def invalidate_message_cache(self, message_id: int):
        """Invalidate a message cache by ID."""
        self._invalidate_cache(self.MESSAGE_KEY_PREFIX, message_id)

    def invalidate_conversation_messages_cache(self, conversation_id: int):
        """Invalidate all messages cache for a conversation."""
        self._invalidate_cache(self.MESSAGES_LIST_PREFIX, conversation_id)
        # Optionally, delete all message keys for this conversation (inefficient)
        # keys = self._safe_redis_call('scan_iter', f"{self.MESSAGE_KEY_PREFIX}*")
        # if keys:
        #     for key in keys:
        #         self._safe_redis_call('delete', key)

    def get_messages_by_conversation_id_from_cache(
        self, conversation_id: int
    ) -> Optional[List[Message]]:
        """Get all messages for a conversation from cache."""
        return self._get_list_from_cache(
            self.MESSAGES_LIST_PREFIX, conversation_id, self.get_message_from_cache
        )

    def cache_messages_by_conversation_id(
        self, conversation_id: int, messages: List[Message]
    ):
        """Cache all messages for a conversation."""
        self._cache_list(
            self.MESSAGES_LIST_PREFIX, conversation_id, messages, self.cache_message
        )

    # ========== Summary Cache Operations ==========
    def get_summary_from_cache(self, summary_id: int) -> Optional[Summary]:
        """Get a summary from cache by ID."""
        return self._get_from_cache(self.SUMMARY_KEY_PREFIX, summary_id, Summary)

    def cache_summary(self, summary: Summary):
        """Cache a summary."""
        if not summary:
            return
        # Cache the summary object
        self._cache_object(self.SUMMARY_KEY_PREFIX, summary)
        # Also cache the summary ID in the conversation's summary list
        if summary.conversation_id:
            list_key = self.cache_key(
                self.SUMMARIES_LIST_PREFIX, summary.conversation_id
            )
            self._safe_redis_call("rpush", list_key, str(summary.id))

    def invalidate_summary_cache(self, summary_id: int):
        """Invalidate a summary cache by ID."""
        self._invalidate_cache(self.SUMMARY_KEY_PREFIX, summary_id)

    def get_summaries_by_conversation_id_from_cache(
        self, conversation_id: int
    ) -> Optional[List[Summary]]:
        """Get all summaries for a conversation from cache."""
        return self._get_list_from_cache(
            self.SUMMARIES_LIST_PREFIX, conversation_id, self.get_summary_from_cache
        )

    def cache_summaries_by_conversation_id(
        self, conversation_id: int, summaries: List[Summary]
    ):
        """Cache all summaries for a conversation."""
        self._cache_list(
            self.SUMMARIES_LIST_PREFIX, conversation_id, summaries, self.cache_summary
        )

    def invalidate_conversation_summaries_cache(self, conversation_id: int):
        """Invalidate all summaries cache for a conversation."""
        self._invalidate_cache(self.SUMMARIES_LIST_PREFIX, conversation_id)

    # ========== Conversation Cache Operations ==========
    def get_conversation_from_cache(
        self, conversation_id: int
    ) -> Optional[Conversation]:
        """Get a conversation from cache by ID."""
        return self._get_from_cache(
            self.CONVERSATION_KEY_PREFIX, conversation_id, Conversation
        )

    def cache_conversation(self, conversation: Conversation):
        """Cache a conversation."""
        if not conversation:
            return
        # Cache the conversation object
        self._cache_object(self.CONVERSATION_KEY_PREFIX, conversation)
        # Using the conversations list key with user ID
        if conversation.user_id:
            list_key = self.cache_key(
                self.CONVERSATION_LIST_PREFIX, conversation.user_id
            )
            self._safe_redis_call("rpush", list_key, str(conversation.id))

    def invalidate_conversation_cache(self, conversation_id: int):
        """Invalidate a conversation cache by ID."""
        self._invalidate_cache(self.CONVERSATION_KEY_PREFIX, conversation_id)

    def get_conversations_by_user_id_from_cache(
        self, user_id: str
    ) -> Optional[List[Conversation]]:
        """Get all conversations for a user from cache."""
        return self._get_list_from_cache(
            self.CONVERSATION_LIST_PREFIX, user_id, self.get_conversation_from_cache
        )

    def cache_conversations_by_user_id(
        self, user_id: str, conversations: List[Conversation]
    ):
        """Cache all conversations for a user."""
        self._cache_list(
            self.CONVERSATION_LIST_PREFIX,
            user_id,
            conversations,
            self.cache_conversation,
        )

    def invalidate_user_conversations_cache(self, user_id: str):
        """Invalidate all conversations cache for a user."""
        self._invalidate_cache(self.CONVERSATION_LIST_PREFIX, user_id)

    # ========== ModelProfile Cache Operations ==========
    def get_model_profile_from_cache(self, profile_id: UUID) -> Optional[ModelProfile]:
        """Get a model profile from cache by ID."""
        if not self.is_storage_cache_enabled():
            return None
        key = self.cache_key(self.MODELPROFILE_KEY_PREFIX, profile_id)
        data = self._safe_redis_call("get", key)
        if not data:
            return None
        try:
            return ModelProfile.parse_raw(
                data.decode() if isinstance(data, bytes) else str(data)
            )
        except Exception as e:
            logger.error(f"Error parsing model profile: {str(e)}")
            return None

    def cache_model_profile(self, profile: ModelProfile):
        """Cache a model profile."""
        if not self.is_storage_cache_enabled() or not profile:
            return
        key = self.cache_key(self.MODELPROFILE_KEY_PREFIX, profile.id)
        self._safe_redis_call("set", key, profile.json(), ex=86400)
        # Also cache the profile ID in the user's profile list
        if profile.user_id:
            list_key = self.cache_key(self.MODELPROFILES_LIST_PREFIX, profile.user_id)
            self._safe_redis_call("rpush", list_key, str(profile.id))

    def invalidate_model_profile_cache(self, profile_id: UUID):
        """Invalidate a model profile cache by ID."""
        if not self.is_storage_cache_enabled():
            return
        key = self.cache_key(self.MODELPROFILE_KEY_PREFIX, profile_id)
        self._safe_redis_call("delete", key)

    def get_model_profiles_list_from_cache(
        self, user_id: str
    ) -> Optional[List[ModelProfile]]:
        """Get all model profiles for a user from cache."""
        if not self.is_storage_cache_enabled():
            return None
        list_key = self.cache_key(self.MODELPROFILES_LIST_PREFIX, user_id)
        try:
            ids = self._safe_redis_call("lrange", list_key, 0, -1)
            if not ids:
                return None
            profiles = []
            for pid in ids:
                pid_val = UUID(pid.decode()) if isinstance(pid, bytes) else UUID(pid)
                profile = self.get_model_profile_from_cache(pid_val)
                if profile:
                    profiles.append(profile)
            return profiles if profiles else None
        except Exception as e:
            logger.error(f"Error getting model profiles from cache: {str(e)}")
            return None

    def cache_model_profiles_list(self, user_id: str, profiles: List[ModelProfile]):
        """Cache all model profiles for a user."""
        if not self.is_storage_cache_enabled() or not profiles:
            return
        list_key = self.cache_key(self.MODELPROFILES_LIST_PREFIX, user_id)
        self._safe_redis_call("delete", list_key)
        for profile in profiles:
            self.cache_model_profile(profile)

    def invalidate_model_profiles_list_cache(self, user_id: str):
        """Invalidate all model profiles cache for a user."""
        if not self.is_storage_cache_enabled():
            return
        list_key = self.cache_key(self.MODELPROFILES_LIST_PREFIX, user_id)
        self._safe_redis_call("delete", list_key)

    # ========== UserConfig Cache Operations ==========
    def get_user_config_from_cache(self, user_id: str) -> Optional[UserConfig]:
        """Get a user config from cache by user ID."""
        if not self.is_storage_cache_enabled():
            return None
        key = self.cache_key(self.USERCONFIG_KEY_PREFIX, user_id)
        data = self._safe_redis_call("get", key)
        if not data:
            return None
        try:
            return UserConfig.parse_raw(
                data.decode() if isinstance(data, bytes) else str(data)
            )
        except Exception as e:
            logger.error(f"Error parsing user config: {str(e)}")
            return None

    def cache_user_config(self, user_id: str, cfg: UserConfig):
        """Cache a user config."""
        if not self.is_storage_cache_enabled() or not cfg:
            return
        key = self.cache_key(self.USERCONFIG_KEY_PREFIX, user_id)
        self._safe_redis_call("set", key, cfg.json(), ex=86400)

    def invalidate_user_config_cache(self, user_id: str):
        """Invalidate a user config cache by user ID."""
        if not self.is_storage_cache_enabled():
            return
        key = self.cache_key(self.USERCONFIG_KEY_PREFIX, user_id)
        self._safe_redis_call("delete", key)

    # ========== Shutdown ==========
    def close_redis_cache(self):
        """Close the Redis client connection."""
        try:
            if self.redis_client:
                self.redis_client.close()
                self.redis_client = None
        except redis.RedisError as e:
            logger.warning(f"Error while closing Redis connection: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Redis shutdown: {str(e)}")


# Create a singleton instance of CacheStorage
cache_storage = CacheStorage()
