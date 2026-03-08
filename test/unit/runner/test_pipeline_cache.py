"""
Unit tests for runner/pipeline_cache.py.

Tests local pipeline cache with weakref caching, memory management,
and intelligent eviction.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import Optional
import threading
import time

from runner.pipeline_cache import (
    LocalPipelineCacheManager,
    _PipelineCacheEntry,
    local_pipeline_cache,
)
from runner.models import Model, ModelProfile, ModelProvider, PipelinePriority
from runner.pipelines.base import BasePipeline
from langchain_core.embeddings import Embeddings


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock(spec=Model)
    model.id = "test-model"
    model.name = "test-model"
    model.provider = ModelProvider.LLAMA_CPP
    model.task = "TextToText"
    model.size = 4 * 1024 * 1024 * 1024  # 4GB
    return model


@pytest.fixture
def mock_model_profile():
    """Create a mock model profile."""
    profile = MagicMock(spec=ModelProfile)
    profile.id = "profile-123"
    profile.model_name = "test-model"
    profile.temperature = 0.7
    profile.max_tokens = 100
    profile.parameters = MagicMock()
    profile.parameters.num_ctx = 4096
    profile.parameters.batch_size = 512
    return profile


@pytest.fixture
def mock_hardware_manager(mocker):
    """Mock hardware manager."""
    mock_manager = MagicMock()
    mock_manager.check_memory_available = MagicMock(return_value=True)
    mock_manager.clear_memory = MagicMock()
    mock_manager.update_all_memory_stats = MagicMock(return_value={})
    mock_manager.gpu_count = 0
    mock_manager.has_gpu = False
    mock_manager.get_gpu_process_info = MagicMock(return_value={})
    return mocker.patch('runner.pipeline_cache.hardware_manager', mock_manager)


@pytest.fixture
def mock_resizer(mocker):
    """Mock resizer for memory estimation."""
    mock_resizer = MagicMock()
    mock_resizer.calculate_memory_breakdown = MagicMock(return_value={
        "model_weights_gpu_gb": 3.0,
        "kv_cache_gb": 0.5,
        "activation_gb": 0.3,
        "overhead_gb": 0.2,
        "clip_model_gb": 0.0,
        "total_gpu_gb": 4.0,
        "gpu_layers_loaded": -1,
    })
    return mocker.patch('runner.pipeline_cache.Resizer', return_value=mock_resizer)


@pytest.fixture
def pipeline_cache(mock_hardware_manager, mock_resizer, mocker):
    """Create a LocalPipelineCacheManager instance."""
    mocker.patch('runner.pipeline_cache.IntelligentOOMRecovery')
    cache = LocalPipelineCacheManager(cache_timeout=60)
    return cache


class TestPipelineCacheEntry:
    """Tests for _PipelineCacheEntry class."""

    def test_entry_initialization(self, mocker):
        """Test cache entry initialization."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)

        assert entry.pipeline == mock_pipeline
        assert entry.priority == PipelinePriority.NORMAL
        assert entry.estimated_memory == 0
        assert entry.is_alive() is True
        assert entry.in_use is False
        assert entry.persistent is False
        assert entry.access_count == 1
        assert entry.use_count == 0

    def test_entry_touch(self, mocker):
        """Test touch method updates last_accessed."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        old_time = entry.last_accessed

        time.sleep(0.01)
        entry.touch()

        assert entry.last_accessed > old_time
        assert entry.access_count == 2

    def test_entry_lock_unlock(self, mocker):
        """Test lock and unlock methods."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)

        entry.lock()
        assert entry.in_use is True
        assert entry.use_count == 1

        entry.lock()
        assert entry.use_count == 2

        entry.unlock()
        assert entry.use_count == 1
        assert entry.in_use is True

        entry.unlock()
        assert entry.use_count == 0
        assert entry.in_use is False

    def test_entry_eviction_score(self, mocker):
        """Test eviction score calculation."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.HIGH)
        entry.touch()

        # High priority should give higher score
        score = entry.eviction_score(time.time(), estimated_memory=0)
        assert score > 0

    def test_entry_eviction_score_with_memory(self, mocker):
        """Test eviction score with memory consideration."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)

        # Small model should get bonus
        small_score = entry.eviction_score(time.time(), estimated_memory=1 * 1024**3)
        # Large model should get penalty
        large_score = entry.eviction_score(time.time(), estimated_memory=15 * 1024**3)

        assert small_score > large_score

    def test_entry_strong_ref_preferred(self, mocker):
        """Test that strong reference is preferred over weak reference."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)

        assert entry.pipeline == mock_pipeline

        # Clear strong ref
        entry._strong_ref = None

        # Should still return via weakref
        assert entry.pipeline == mock_pipeline


class TestLocalPipelineCacheManagerInitialization:
    """Tests for LocalPipelineCacheManager initialization."""

    def test_cache_initialization(self, mocker):
        """Test cache initialization."""
        mocker.patch('runner.pipeline_cache.IntelligentOOMRecovery')
        cache = LocalPipelineCacheManager()

        assert cache._cache == {}
        assert cache._lock is not None
        assert cache._cache_timeout == 300  # Default timeout
        assert cache._cleanup_thread is not None
        assert cache._stop_event is not None

    def test_cache_with_custom_timeout(self, mocker):
        """Test cache initialization with custom timeout."""
        mocker.patch('runner.pipeline_cache.IntelligentOOMRecovery')
        cache = LocalPipelineCacheManager(cache_timeout=120)

        assert cache._cache_timeout == 120

    def test_cache_disables_oom_recovery_on_error(self, mocker):
        """Test cache disables OOM recovery on initialization error."""
        mocker.patch('runner.pipeline_cache.IntelligentOOMRecovery', side_effect=Exception("Init failed"))
        cache = LocalPipelineCacheManager()

        assert cache._oom_recovery is None


class TestIsLocal:
    """Tests for is_local method."""

    def test_is_local_llama_cpp(self, pipeline_cache):
        """Test is_local returns True for llama.cpp."""
        model = MagicMock()
        model.provider = ModelProvider.LLAMA_CPP

        assert pipeline_cache.is_local(model) is True

    def test_is_local_stable_diffusion(self, pipeline_cache):
        """Test is_local returns True for stable diffusion."""
        model = MagicMock()
        model.provider = ModelProvider.STABLE_DIFFUSION_CPP

        assert pipeline_cache.is_local(model) is True

    def test_is_local_remote_provider(self, pipeline_cache):
        """Test is_local returns False for remote providers."""
        model = MagicMock()
        model.provider = ModelProvider.OPENAI

        assert pipeline_cache.is_local(model) is False


class TestGetOrCreate:
    """Tests for get_or_create method."""

    def test_get_or_create_returns_cached(self, pipeline_cache, mock_model, mock_model_profile, mocker):
        """Test get_or_create returns cached pipeline."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        pipeline_cache._cache[str(mock_model_profile.id)] = entry

        create_fn = MagicMock(return_value=None)

        pipeline = pipeline_cache.get_or_create(mock_model, mock_model_profile, PipelinePriority.NORMAL, create_fn)

        assert pipeline == mock_pipeline
        create_fn.assert_not_called()

    def test_get_or_create_creates_new(self, pipeline_cache, mock_model, mock_model_profile, mocker):
        """Test get_or_create creates new pipeline when not cached."""
        mock_pipeline = MagicMock(spec=BasePipeline)
        mock_pipeline.bind_metadata = MagicMock()
        create_fn = MagicMock(return_value=mock_pipeline)

        pipeline = pipeline_cache.get_or_create(mock_model, mock_model_profile, PipelinePriority.NORMAL, create_fn)

        assert pipeline == mock_pipeline
        create_fn.assert_called_once()

    def test_get_or_create_evicts_dead_entry(self, pipeline_cache, mock_model, mock_model_profile, mocker):
        """Test get_or_create evicts dead entry."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        # Make entry dead by clearing both strong and weak refs
        entry._strong_ref = None
        entry._ref = lambda: None  # Weakref returns None
        pipeline_cache._cache[str(mock_model_profile.id)] = entry

        new_pipeline = MagicMock()
        create_fn = MagicMock(return_value=new_pipeline)

        pipeline = pipeline_cache.get_or_create(mock_model, mock_model_profile, PipelinePriority.NORMAL, create_fn)

        assert pipeline == new_pipeline
        assert str(mock_model_profile.id) in pipeline_cache._cache

    def test_get_or_create_memory_insufficient(self, pipeline_cache, mock_model, mock_model_profile, mocker):
        """Test get_or_create raises error when memory insufficient."""
        pipeline_cache._ensure_memory = MagicMock(return_value=False)
        create_fn = MagicMock()

        with pytest.raises(RuntimeError, match="Insufficient memory"):
            pipeline_cache.get_or_create(mock_model, mock_model_profile, PipelinePriority.NORMAL, create_fn)


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clear_cache_single(self, pipeline_cache, mock_model, mocker):
        """Test clearing single cache entry."""
        mock_pipeline = MagicMock()
        mock_pipeline.server_manager = None
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        pipeline_cache._cache["model-1"] = entry
        pipeline_cache._cleanup_pipeline = MagicMock()

        pipeline_cache.clear_cache("model-1")

        assert "model-1" not in pipeline_cache._cache
        pipeline_cache._cleanup_pipeline.assert_called_once_with(mock_pipeline)

    def test_clear_cache_all(self, pipeline_cache, mocker):
        """Test clearing all cache entries."""
        mock_pipeline1 = MagicMock()
        mock_pipeline2 = MagicMock()
        pipeline_cache._cache["model-1"] = _PipelineCacheEntry(mock_pipeline1, PipelinePriority.NORMAL)
        pipeline_cache._cache["model-2"] = _PipelineCacheEntry(mock_pipeline2, PipelinePriority.NORMAL)
        pipeline_cache._cleanup_pipeline = MagicMock()

        pipeline_cache.clear_cache()

        assert pipeline_cache._cache == {}
        assert pipeline_cache._cleanup_pipeline.call_count == 2


class TestClearExpired:
    """Tests for clear_expired method."""

    def test_clear_expired_removes_expired(self, pipeline_cache, mock_model, mocker):
        """Test clear_expired removes expired entries."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        entry.last_accessed = time.time() - 1000  # Expired
        entry.persistent = False
        # Set large estimated_memory to get shorter timeout (300 * 3.0 = 900 seconds)
        entry.estimated_memory = 5 * 1024**3  # 5GB
        pipeline_cache._cache["model-1"] = entry
        pipeline_cache._cleanup_pipeline = MagicMock()

        pipeline_cache.clear_expired()

        assert "model-1" not in pipeline_cache._cache

    def test_clear_expired_keeps_persistent(self, pipeline_cache, mock_model, mocker):
        """Test clear_expired keeps persistent entries."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        entry.last_accessed = time.time() - 1000  # Expired
        entry.persistent = True
        pipeline_cache._cache["model-1"] = entry

        pipeline_cache.clear_expired()

        assert "model-1" in pipeline_cache._cache

    def test_clear_expired_keeps_in_use(self, pipeline_cache, mock_model, mocker):
        """Test clear_expired keeps in-use entries."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        entry.last_accessed = time.time() - 1000  # Expired
        entry.in_use = True
        pipeline_cache._cache["model-1"] = entry

        pipeline_cache.clear_expired()

        assert "model-1" in pipeline_cache._cache


class TestStats:
    """Tests for stats method."""

    def test_stats_returns_correct_format(self, pipeline_cache, mock_model, mocker):
        """Test stats returns correct format."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        entry.in_use = True
        entry.access_count = 5
        entry.last_accessed = time.time()
        entry.estimated_memory = 1 * 1024**3
        pipeline_cache._cache["model-1"] = entry

        stats = pipeline_cache.stats()

        assert "count" in stats
        assert "alive" in stats
        assert "locked" in stats
        assert "entries" in stats
        assert "memory" in stats


class TestLockUnlockPipeline:
    """Tests for lock_pipeline and unlock_pipeline methods."""

    def test_lock_pipeline(self, pipeline_cache, mock_model, mocker):
        """Test lock_pipeline marks entry as in-use."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        pipeline_cache._cache["model-1"] = entry

        result = pipeline_cache.lock_pipeline("model-1")

        assert result is True
        assert entry.in_use is True

    def test_lock_pipeline_not_found(self, pipeline_cache):
        """Test lock_pipeline returns False when not found."""
        result = pipeline_cache.lock_pipeline("nonexistent")

        assert result is False

    def test_unlock_pipeline(self, pipeline_cache, mock_model, mocker):
        """Test unlock_pipeline releases in-use state."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        entry.lock()
        pipeline_cache._cache["model-1"] = entry

        result = pipeline_cache.unlock_pipeline("model-1")

        assert result is True
        assert entry.in_use is False


class TestSetPersistent:
    """Tests for set_persistent method."""

    def test_set_persistent_true(self, pipeline_cache, mock_model, mocker):
        """Test set_persistent marks entry as persistent."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        pipeline_cache._cache["model-1"] = entry

        result = pipeline_cache.set_persistent("model-1", True)

        assert result is True
        assert entry.persistent is True

    def test_set_persistent_false(self, pipeline_cache, mock_model, mocker):
        """Test set_persistent removes persistent marking."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        entry.persistent = True
        pipeline_cache._cache["model-1"] = entry

        result = pipeline_cache.set_persistent("model-1", False)

        assert result is True
        assert entry.persistent is False


class TestGetCacheInfo:
    """Tests for get_cache_info method."""

    def test_get_cache_info_returns_correct_format(self, pipeline_cache, mock_model, mocker):
        """Test get_cache_info returns correct format."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.HIGH)
        entry.estimated_memory = 1 * 1024**3
        entry.in_use = True
        pipeline_cache._cache["model-1"] = entry

        info = pipeline_cache.get_cache_info()

        assert "total_models" in info
        assert "total_memory_gb" in info
        assert "small_models" in info
        assert "large_models" in info
        assert "locked_models" in info
        assert "high_priority_models" in info


class TestForceCleanup:
    """Tests for force_cleanup method."""

    def test_force_cleanup(self, pipeline_cache, mock_model, mocker):
        """Test force_cleanup clears all entries."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        pipeline_cache._cache["model-1"] = entry
        pipeline_cache._cleanup_pipeline = MagicMock()

        count = pipeline_cache.force_cleanup()

        assert count == 1
        assert pipeline_cache._cache == {}
        pipeline_cache._cleanup_pipeline.assert_called_once()


class TestStop:
    """Tests for stop method."""

    def test_stop_stops_cleanup_thread(self, pipeline_cache, mocker):
        """Test stop signals cleanup thread to stop."""
        pipeline_cache._cleanup_thread = threading.Thread(
            target=lambda: None, daemon=True
        )
        pipeline_cache._cleanup_thread.start()

        pipeline_cache.stop(timeout=1.0)

        assert pipeline_cache._stop_event.is_set()


class TestPipelineInUseContextManager:
    """Tests for pipeline_in_use context manager."""

    def test_pipeline_in_use_context(self, pipeline_cache, mock_model, mocker):
        """Test pipeline_in_use context manager locks and unlocks."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        pipeline_cache._cache["model-1"] = entry

        with pipeline_cache.pipeline_in_use("model-1") as locked:
            assert locked is True
            assert entry.in_use is True

        assert entry.in_use is False