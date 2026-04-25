"""
Unit tests for pipeline cache use_count (locking) functionality.

Tests the use_count mechanism that prevents eviction during active inference.
Since use_count is incremented in get() (which requires a full model setup),
these tests directly manipulate _CacheEntry.use_count to simulate locking.
"""

import pytest
from unittest.mock import MagicMock, patch

from runner.pipeline_cache import PipelineCache, _CacheEntry


class TestCacheEntry:
    """Test cases for _CacheEntry use_count functionality."""

    def test_initial_state(self):
        """Test that new cache entries start with use_count=0."""
        mock_pipeline = MagicMock()
        entry = _CacheEntry(pipeline=mock_pipeline)

        assert entry.use_count == 0
        assert entry.estimated_size_bytes == 0
        assert entry.last_accessed > 0

    def test_manual_increment_decrement(self):
        """Test manually incrementing and decrementing use_count."""
        mock_pipeline = MagicMock()
        entry = _CacheEntry(pipeline=mock_pipeline)

        entry.use_count += 1
        assert entry.use_count == 1

        entry.use_count += 1
        assert entry.use_count == 2

        entry.use_count -= 1
        assert entry.use_count == 1

        entry.use_count -= 1
        assert entry.use_count == 0


class TestPipelineCacheUnlock:
    """Test cases for PipelineCache.unlock() functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(PipelineCache, '_cleanup_loop', return_value=None):
            self.cache = PipelineCache()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.__class__.__name__ = "TestPipeline"
        self.model_id = "test-model"

    def teardown_method(self):
        """Clean up after each test."""
        self.cache.clear()

    def _add_test_entry(self, use_count=0):
        """Helper to add a test entry directly to cache."""
        entry = _CacheEntry(pipeline=self.mock_pipeline, estimated_size_bytes=1024)
        entry.use_count = use_count
        with self.cache._lock:
            self.cache._cache[self.model_id] = entry
        return entry

    def test_unlock_existing_entry(self):
        """Test unlock() on an existing entry returns True and decrements use_count."""
        self._add_test_entry(use_count=1)

        result = self.cache.unlock(self.model_id)
        assert result is True

        with self.cache._lock:
            assert self.cache._cache[self.model_id].use_count == 0

    def test_unlock_nonexistent_entry(self):
        """Test unlock() on a non-existing entry returns False."""
        result = self.cache.unlock("nonexistent-model")
        assert result is False

    def test_unlock_does_not_go_negative(self):
        """Test that unlock() on use_count=0 doesn't go negative."""
        self._add_test_entry(use_count=0)

        result = self.cache.unlock(self.model_id)
        assert result is True

        with self.cache._lock:
            assert self.cache._cache[self.model_id].use_count == 0

    def test_unlock_multiple(self):
        """Test multiple unlocks decrement use_count correctly."""
        self._add_test_entry(use_count=3)

        self.cache.unlock(self.model_id)
        with self.cache._lock:
            assert self.cache._cache[self.model_id].use_count == 2

        self.cache.unlock(self.model_id)
        with self.cache._lock:
            assert self.cache._cache[self.model_id].use_count == 1

        self.cache.unlock(self.model_id)
        with self.cache._lock:
            assert self.cache._cache[self.model_id].use_count == 0


class TestPipelineCacheStats:
    """Test cases for PipelineCache.stats() including locked count."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(PipelineCache, '_cleanup_loop', return_value=None):
            self.cache = PipelineCache()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.__class__.__name__ = "TestPipeline"

    def teardown_method(self):
        """Clean up after each test."""
        self.cache.clear()

    def _add_test_entry(self, model_id, use_count=0, size_bytes=1024):
        """Helper to add a test entry directly to cache."""
        entry = _CacheEntry(pipeline=self.mock_pipeline, estimated_size_bytes=size_bytes)
        entry.use_count = use_count
        with self.cache._lock:
            self.cache._cache[model_id] = entry
        return entry

    def test_stats_includes_locked_count(self):
        """Test that stats includes locked count based on use_count > 0."""
        self._add_test_entry("model-a", use_count=0)
        self._add_test_entry("model-b", use_count=2)
        self._add_test_entry("model-c", use_count=1)

        stats = self.cache.stats()

        assert stats["count"] == 3
        assert stats["locked"] == 2
        assert "total_cached_gb" in stats
        assert "entries" in stats
        assert "gpu" in stats

    def test_stats_entry_info(self):
        """Test that stats entries include use_count and metadata."""
        self._add_test_entry("my-model", use_count=1, size_bytes=1_000_000_000)

        stats = self.cache.stats()

        assert "my-model" in stats["entries"]
        entry_info = stats["entries"]["my-model"]
        assert entry_info["use_count"] == 1
        assert "last_accessed" in entry_info
        assert "estimated_size_gb" in entry_info


class TestPipelineCacheClear:
    """Test cases for PipelineCache.clear() functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(PipelineCache, '_cleanup_loop', return_value=None):
            self.cache = PipelineCache()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.__class__.__name__ = "TestPipeline"

    def teardown_method(self):
        """Clean up after each test."""
        self.cache.clear()

    def _add_test_entry(self, model_id):
        """Helper to add a test entry directly to cache."""
        entry = _CacheEntry(pipeline=self.mock_pipeline, estimated_size_bytes=1024)
        with self.cache._lock:
            self.cache._cache[model_id] = entry
        return entry

    def test_clear_specific_entry(self):
        """Test clear(model_id) removes only the specified entry."""
        self._add_test_entry("model-a")
        self._add_test_entry("model-b")

        self.cache.clear("model-a")

        with self.cache._lock:
            assert "model-a" not in self.cache._cache
            assert "model-b" in self.cache._cache

    def test_clear_all_entries(self):
        """Test clear() without args removes all entries."""
        self._add_test_entry("model-a")
        self._add_test_entry("model-b")

        self.cache.clear()

        with self.cache._lock:
            assert len(self.cache._cache) == 0

    def test_clear_nonexistent_entry(self):
        """Test clear(model_id) on non-existing entry doesn't raise."""
        self.cache.clear("nonexistent-model")
        # No exception is the expected behavior
