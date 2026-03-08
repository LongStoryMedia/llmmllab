"""
Unit tests for server/db/cache_storage.py.

Tests Redis-based caching for messages, summaries, conversations, model profiles, and user configs.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import UUID
import redis
from server.db.cache_storage import CacheStorage, cache_storage
from server.models.message import Message, MessageContent, MessageContentType
from server.models.summary import Summary
from server.models.conversation import Conversation
from server.models.model_profile import ModelProfile
from server.models.model_parameters import ModelParameters
from datetime import datetime
from server.models.user_config import UserConfig
from server.models.circuit_breaker_config import CircuitBreakerConfig
from server.models.gpu_config import GPUConfig
from server.models.parameter_optimization_config import ParameterOptimizationConfig
from server.models.workflow_config import WorkflowConfig
from server.models.crash_prevention import CrashPrevention
from server.models.performance_parameter import PerformanceParameter
from server.models.parameter_tuning_strategy import ParameterTuningStrategy


@pytest.fixture
def mock_redis_client(mocker):
    """Create a mock Redis client."""
    mock_client = MagicMock()
    mock_client.ping = MagicMock(return_value=True)
    mock_client.close = MagicMock()
    return mock_client


@pytest.fixture
def cache_storage_instance(mocker, mock_redis_client):
    """Create a CacheStorage instance with mocked Redis."""
    cache = CacheStorage()
    cache.redis_client = mock_redis_client
    mocker.patch.object(cache, 'is_storage_cache_enabled', return_value=True)
    return cache


class TestCacheStorageInitialization:
    """Tests for CacheStorage initialization."""

    def test_init_without_redis_url(self):
        """Test CacheStorage initialization without Redis URL."""
        cache = CacheStorage()

        assert cache.redis_client is None
        assert cache._health_check_thread is None

    def test_init_with_redis_url(self, mocker):
        """Test CacheStorage initialization with Redis URL."""
        mock_redis = MagicMock()
        mock_redis.ping = MagicMock(return_value=True)
        mocker.patch('server.db.cache_storage.redis.Redis.from_url', return_value=mock_redis)

        cache = CacheStorage(redis_url="redis://localhost:6379", timeout=10)

        assert cache.redis_client is not None
        assert cache._health_check_thread is not None

    def test_init_handles_redis_connection_error(self, mocker):
        """Test CacheStorage handles Redis connection error."""
        mock_redis = MagicMock()
        mock_redis.ping = MagicMock(side_effect=redis.ConnectionError("Connection failed"))

        mock_from_url = mocker.patch(
            'server.db.cache_storage.redis.Redis.from_url',
            return_value=mock_redis
        )

        with pytest.raises(RuntimeError, match="Redis connection failed"):
            CacheStorage(redis_url="redis://localhost:6379", timeout=10)


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_cache_key_message(self):
        """Test message cache key generation."""
        cache = CacheStorage()
        key = cache.cache_key(cache.MESSAGE_KEY_PREFIX, 123)
        assert key == "llmmll:message:123"

    def test_cache_key_summary(self):
        """Test summary cache key generation."""
        cache = CacheStorage()
        key = cache.cache_key(cache.SUMMARY_KEY_PREFIX, 456)
        assert key == "llmmll:summary:456"

    def test_cache_key_conversation(self):
        """Test conversation cache key generation."""
        cache = CacheStorage()
        key = cache.cache_key(cache.CONVERSATION_KEY_PREFIX, 789)
        assert key == "llmmll:conversation:789"

    def test_cache_key_userconfig(self):
        """Test user config cache key generation."""
        cache = CacheStorage()
        key = cache.cache_key(cache.USERCONFIG_KEY_PREFIX, "user-123")
        assert key == "llmmll:userconfig:user-123"

    def test_cache_key_modelprofile(self):
        """Test model profile cache key generation."""
        cache = CacheStorage()
        uuid = UUID("12345678-1234-5678-1234-567812345678")
        key = cache.cache_key(cache.MODELPROFILE_KEY_PREFIX, uuid)
        assert key == "llmmll:modelprofile:12345678-1234-5678-1234-567812345678"


class TestIsStorageCacheEnabled:
    """Tests for is_storage_cache_enabled method."""

    def test_enabled_with_redis_client(self, cache_storage_instance):
        """Test enabled returns True when Redis client exists and pings."""
        cache_storage_instance.redis_client.ping = MagicMock(return_value=True)
        assert cache_storage_instance.is_storage_cache_enabled() is True

    def test_disabled_without_redis_client(self):
        """Test disabled when Redis client is None."""
        cache = CacheStorage()
        assert cache.is_storage_cache_enabled() is False

    def test_disabled_on_redis_error(self, mocker):
        """Test disabled on Redis error."""
        cache = CacheStorage()
        mock_client = MagicMock()
        mock_client.ping = MagicMock(side_effect=Exception("Redis error"))
        cache.redis_client = mock_client

        result = cache.is_storage_cache_enabled()

        assert result is False


class TestSafeRedisCall:
    """Tests for _safe_redis_call method."""

    def test_safe_redis_call_success(self, cache_storage_instance):
        """Test successful Redis call."""
        cache_storage_instance.redis_client.get = MagicMock(return_value=b"value")

        result = cache_storage_instance._safe_redis_call("get", "key")

        assert result == b"value"

    def test_safe_redis_call_none_client(self):
        """Test safe Redis call with None client."""
        cache = CacheStorage()

        result = cache._safe_redis_call("get", "key")

        assert result is None

    def test_safe_redis_call_redis_error(self, mocker):
        """Test safe Redis call handles Redis error."""
        cache = CacheStorage()
        mock_client = MagicMock()
        mock_client.get = MagicMock(side_effect=Exception("Redis error"))
        cache.redis_client = mock_client

        result = cache._safe_redis_call("get", "key")

        assert result is None

    def test_safe_redis_call_attribute_error(self, mocker):
        """Test safe Redis call handles invalid method."""
        cache = CacheStorage()
        # Use spec to ensure AttributeError is raised for invalid methods
        mock_client = MagicMock(spec=redis.Redis)
        cache.redis_client = mock_client

        result = cache._safe_redis_call("nonexistent_method", "key")

        assert result is None


class TestMessageCacheOperations:
    """Tests for message cache operations."""

    def test_get_message_from_cache(self, cache_storage_instance, mocker):
        """Test getting a message from cache."""
        mock_message = Message(
            id=1,
            conversation_id=1,
            role="user",
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello")]
        )
        mocker.patch.object(cache_storage_instance, '_get_from_cache', return_value=mock_message)

        result = cache_storage_instance.get_message_from_cache(1)

        assert result.id == 1

    def test_cache_message(self, cache_storage_instance, mocker):
        """Test caching a message."""
        mock_message = Message(
            id=1,
            conversation_id=1,
            role="user",
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello")]
        )
        cache_storage_instance._cache_object = MagicMock()
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.cache_message(mock_message)

        cache_storage_instance._cache_object.assert_called_once()
        cache_storage_instance._safe_redis_call.assert_called_once()

    def test_invalidate_message_cache(self, cache_storage_instance):
        """Test invalidating message cache."""
        cache_storage_instance._invalidate_cache = MagicMock()

        cache_storage_instance.invalidate_message_cache(1)

        cache_storage_instance._invalidate_cache.assert_called_once()

    def test_get_messages_by_conversation_id_from_cache(self, cache_storage_instance, mocker):
        """Test getting messages by conversation ID from cache."""
        cache_storage_instance._get_list_from_cache = MagicMock(return_value=[])
        cache_storage_instance.get_message_from_cache = MagicMock(return_value=None)

        result = cache_storage_instance.get_messages_by_conversation_id_from_cache(1)

        assert result == []

    def test_cache_messages_by_conversation_id(self, cache_storage_instance, mocker):
        """Test caching messages by conversation ID."""
        cache_storage_instance._cache_list = MagicMock()
        mock_message = Message(
            id=1,
            conversation_id=1,
            role="user",
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello")]
        )

        cache_storage_instance.cache_messages_by_conversation_id(1, [mock_message])

        cache_storage_instance._cache_list.assert_called_once()

    def test_invalidate_conversation_messages_cache(self, cache_storage_instance):
        """Test invalidating conversation messages cache."""
        cache_storage_instance._invalidate_cache = MagicMock()

        cache_storage_instance.invalidate_conversation_messages_cache(1)

        cache_storage_instance._invalidate_cache.assert_called_once()


class TestSummaryCacheOperations:
    """Tests for summary cache operations."""

    def test_get_summary_from_cache(self, cache_storage_instance, mocker):
        """Test getting a summary from cache."""
        mock_summary = Summary(
            id=1,
            conversation_id=1,
            content="Summary content",
            level=1,
            source_ids=[1, 2],
            created_at=datetime.now()
        )
        mocker.patch.object(cache_storage_instance, '_get_from_cache', return_value=mock_summary)

        result = cache_storage_instance.get_summary_from_cache(1)

        assert result.id == 1

    def test_cache_summary(self, cache_storage_instance, mocker):
        """Test caching a summary."""
        mock_summary = Summary(
            id=1,
            conversation_id=1,
            content="Summary content",
            level=1,
            source_ids=[1, 2],
            created_at=datetime.now()
        )
        cache_storage_instance._cache_object = MagicMock()
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.cache_summary(mock_summary)

        cache_storage_instance._cache_object.assert_called_once()

    def test_get_summaries_by_conversation_id_from_cache(self, cache_storage_instance, mocker):
        """Test getting summaries by conversation ID from cache."""
        cache_storage_instance._get_list_from_cache = MagicMock(return_value=[])

        result = cache_storage_instance.get_summaries_by_conversation_id_from_cache(1)

        assert result == []

    def test_invalidate_conversation_summaries_cache(self, cache_storage_instance):
        """Test invalidating conversation summaries cache."""
        cache_storage_instance._invalidate_cache = MagicMock()

        cache_storage_instance.invalidate_conversation_summaries_cache(1)

        cache_storage_instance._invalidate_cache.assert_called_once()


class TestConversationCacheOperations:
    """Tests for conversation cache operations."""

    def test_get_conversation_from_cache(self, cache_storage_instance, mocker):
        """Test getting a conversation from cache."""
        mock_conversation = Conversation(
            id=1,
            user_id="user-1",
            title="Test Conversation",
            created_at="2024-01-01T00:00:00Z"
        )
        mocker.patch.object(cache_storage_instance, '_get_from_cache', return_value=mock_conversation)

        result = cache_storage_instance.get_conversation_from_cache(1)

        assert result.id == 1

    def test_cache_conversation(self, cache_storage_instance, mocker):
        """Test caching a conversation."""
        mock_conversation = Conversation(
            id=1,
            user_id="user-1",
            title="Test Conversation",
            created_at="2024-01-01T00:00:00Z"
        )
        cache_storage_instance._cache_object = MagicMock()
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.cache_conversation(mock_conversation)

        cache_storage_instance._cache_object.assert_called_once()

    def test_get_conversations_by_user_id_from_cache(self, cache_storage_instance, mocker):
        """Test getting conversations by user ID from cache."""
        cache_storage_instance._get_list_from_cache = MagicMock(return_value=[])

        result = cache_storage_instance.get_conversations_by_user_id_from_cache("user-1")

        assert result == []

    def test_invalidate_user_conversations_cache(self, cache_storage_instance):
        """Test invalidating user conversations cache."""
        cache_storage_instance._invalidate_cache = MagicMock()

        cache_storage_instance.invalidate_user_conversations_cache("user-1")

        cache_storage_instance._invalidate_cache.assert_called_once()


class TestModelProfileCacheOperations:
    """Tests for model profile cache operations."""

    def test_get_model_profile_from_cache(self, cache_storage_instance, mocker):
        """Test getting a model profile from cache."""
        uuid = UUID("12345678-1234-5678-1234-567812345678")
        mock_profile = ModelProfile(
            id=uuid,
            user_id="user-1",
            name="Test Profile",
            model_name="gpt-4",
            parameters=ModelParameters(temperature=0.7),
            system_prompt="Test prompt",
            type=1
        )
        # Patch _safe_redis_call to return valid JSON data
        import json
        profile_dict = mock_profile.model_dump(mode="json")
        mocker.patch.object(
            cache_storage_instance,
            '_safe_redis_call',
            return_value=json.dumps(profile_dict).encode()
        )

        result = cache_storage_instance.get_model_profile_from_cache(uuid)

        assert result.id == uuid

    def test_cache_model_profile(self, cache_storage_instance, mocker):
        """Test caching a model profile."""
        uuid = UUID("12345678-1234-5678-1234-567812345678")
        mock_profile = ModelProfile(
            id=uuid,
            user_id="user-1",
            name="Test Profile",
            model_name="gpt-4",
            parameters=ModelParameters(temperature=0.7),
            system_prompt="Test prompt",
            type=1
        )
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.cache_model_profile(mock_profile)

        assert cache_storage_instance._safe_redis_call.call_count == 2

    def test_get_model_profiles_list_from_cache(self, cache_storage_instance, mocker):
        """Test getting model profiles list from cache."""
        uuid = UUID("12345678-1234-5678-1234-567812345678")
        # Return profile ID list from Redis
        cache_storage_instance._safe_redis_call = MagicMock(return_value=[str(uuid).encode()])
        # Return a valid profile from get_model_profile_from_cache
        mock_profile = ModelProfile(
            id=uuid,
            user_id="user-1",
            name="Test Profile",
            model_name="gpt-4",
            parameters=ModelParameters(temperature=0.7),
            system_prompt="Test prompt",
            type=1
        )
        mocker.patch.object(cache_storage_instance, 'get_model_profile_from_cache', return_value=mock_profile)

        result = cache_storage_instance.get_model_profiles_list_from_cache("user-1")

        assert result[0].id == uuid

    def test_invalidate_model_profiles_list_cache(self, cache_storage_instance):
        """Test invalidating model profiles list cache."""
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.invalidate_model_profiles_list_cache("user-1")

        cache_storage_instance._safe_redis_call.assert_called_once()


class TestUserConfigCacheOperations:
    """Tests for user config cache operations."""

    def test_get_user_config_from_cache(self, cache_storage_instance, mocker):
        """Test getting a user config from cache."""
        mock_config = UserConfig(
            user_id="user-1",
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention()
            ),
            workflow=WorkflowConfig(),
        )
        # Patch _safe_redis_call to return valid JSON data
        import json
        mocker.patch.object(
            cache_storage_instance,
            '_safe_redis_call',
            return_value=mock_config.json().encode()
        )

        result = cache_storage_instance.get_user_config_from_cache("user-1")

        assert result.user_id == "user-1"

    def test_cache_user_config(self, cache_storage_instance, mocker):
        """Test caching a user config."""
        mock_config = UserConfig(
            user_id="user-1",
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention()
            ),
            workflow=WorkflowConfig(),
        )
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.cache_user_config("user-1", mock_config)

        cache_storage_instance._safe_redis_call.assert_called_once()

    def test_invalidate_user_config_cache(self, cache_storage_instance):
        """Test invalidating user config cache."""
        cache_storage_instance._safe_redis_call = MagicMock()

        cache_storage_instance.invalidate_user_config_cache("user-1")

        cache_storage_instance._safe_redis_call.assert_called_once()


class TestCloseRedisCache:
    """Tests for close_redis_cache method."""

    def test_close_redis_cache_success(self, cache_storage_instance):
        """Test closing Redis cache successfully."""
        cache_storage_instance.close_redis_cache()

        assert cache_storage_instance.redis_client is None

    def test_close_redis_cache_handles_error(self, mocker):
        """Test close_redis_cache handles Redis error."""
        cache = CacheStorage()
        mock_client = MagicMock()
        mock_client.close = MagicMock(side_effect=Exception("Redis error"))
        cache.redis_client = mock_client

        # Should not raise
        cache.close_redis_cache()