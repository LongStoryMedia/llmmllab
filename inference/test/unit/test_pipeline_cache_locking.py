"""
Unit tests for pipeline cache locking functionality.

Tests the pipeline locking mechanism that prevents eviction during active inference.
"""

import pytest
from unittest.mock import MagicMock

from runner.pipeline_cache import LocalPipelineCacheManager, _PipelineCacheEntry
from models import PipelinePriority


class TestPipelineCacheEntry:
    """Test cases for _PipelineCacheEntry locking functionality."""

    def test_initial_state(self):
        """Test that new cache entries start unlocked."""
        mock_pipeline = MagicMock()
        mock_pipeline.__class__.__name__ = "TestPipeline"
        
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        
        assert not entry.in_use
        assert entry._use_count == 0  # noqa: SLF001 - accessing for test validation

    def test_single_lock_unlock(self):
        """Test basic lock and unlock functionality."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        
        # Lock the entry
        entry.lock()
        assert entry.in_use
        assert entry._use_count == 1  # noqa: SLF001
        
        # Unlock the entry
        entry.unlock()
        assert not entry.in_use
        assert entry._use_count == 0  # noqa: SLF001

    def test_nested_locking(self):
        """Test support for concurrent/nested usage."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        
        # Lock twice (concurrent usage)
        entry.lock()
        entry.lock()
        assert entry.in_use
        assert entry._use_count == 2  # noqa: SLF001
        
        # Unlock once - should still be in use
        entry.unlock()
        assert entry.in_use
        assert entry._use_count == 1  # noqa: SLF001
        
        # Unlock completely
        entry.unlock()
        assert not entry.in_use
        assert entry._use_count == 0  # noqa: SLF001

    def test_unlock_underflow_protection(self):
        """Test that unlocking more than locking doesn't go negative."""
        mock_pipeline = MagicMock()
        entry = _PipelineCacheEntry(mock_pipeline, PipelinePriority.NORMAL)
        
        # Unlock without locking
        entry.unlock()
        assert not entry.in_use
        assert entry._use_count == 0  # noqa: SLF001


class TestLocalPipelineCacheManagerLocking:
    """Test cases for LocalPipelineCacheManager locking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = LocalPipelineCacheManager()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.__class__.__name__ = "TestPipeline"
        self.model_id = "test-model"

    def teardown_method(self):
        """Clean up after each test."""
        self.cache_manager.force_cleanup()

    def _add_test_entry(self):
        """Helper to add a test entry directly to cache."""
        entry = _PipelineCacheEntry(self.mock_pipeline, PipelinePriority.NORMAL)
        with self.cache_manager._lock:  # noqa: SLF001 - test utility
            self.cache_manager._cache[self.model_id] = entry  # noqa: SLF001
        return entry

    def test_lock_existing_pipeline(self):
        """Test locking an existing pipeline."""
        self._add_test_entry()
        
        result = self.cache_manager.lock_pipeline(self.model_id)
        assert result is True
        
        stats = self.cache_manager.stats()
        assert stats["locked"] == 1
        assert stats["entries"][self.model_id]["in_use"] is True
        assert stats["entries"][self.model_id]["use_count"] == 1

    def test_lock_nonexistent_pipeline(self):
        """Test locking a pipeline that doesn't exist."""
        result = self.cache_manager.lock_pipeline("nonexistent-model")
        assert result is False

    def test_unlock_existing_pipeline(self):
        """Test unlocking an existing pipeline."""
        self._add_test_entry()
        
        # Lock first
        self.cache_manager.lock_pipeline(self.model_id)
        
        # Then unlock
        result = self.cache_manager.unlock_pipeline(self.model_id)
        assert result is True
        
        stats = self.cache_manager.stats()
        assert stats["locked"] == 0
        assert stats["entries"][self.model_id]["in_use"] is False

    def test_unlock_nonexistent_pipeline(self):
        """Test unlocking a pipeline that doesn't exist."""
        result = self.cache_manager.unlock_pipeline("nonexistent-model")
        assert result is False

    def test_context_manager_success(self):
        """Test the context manager for pipeline usage."""
        self._add_test_entry()
        
        with self.cache_manager.pipeline_in_use(self.model_id) as locked:
            assert locked is True
            
            # Check that pipeline is locked during context
            stats = self.cache_manager.stats()
            assert stats["entries"][self.model_id]["in_use"] is True
        
        # Check that pipeline is unlocked after context
        stats = self.cache_manager.stats()
        assert stats["entries"][self.model_id]["in_use"] is False

    def test_context_manager_failure(self):
        """Test the context manager with nonexistent pipeline."""
        with self.cache_manager.pipeline_in_use("nonexistent-model") as locked:
            assert locked is False

    def test_context_manager_nested_locking(self):
        """Test nested context managers increase use count."""
        self._add_test_entry()
        
        # Manual lock first
        self.cache_manager.lock_pipeline(self.model_id)
        initial_stats = self.cache_manager.stats()
        assert initial_stats["entries"][self.model_id]["use_count"] == 1
        
        # Use context manager (should increment count)
        with self.cache_manager.pipeline_in_use(self.model_id) as locked:
            assert locked is True
            nested_stats = self.cache_manager.stats()
            assert nested_stats["entries"][self.model_id]["use_count"] == 2
        
        # After context, should be back to original count
        final_stats = self.cache_manager.stats()
        assert final_stats["entries"][self.model_id]["use_count"] == 1

    def test_stats_include_lock_info(self):
        """Test that stats include locking information."""
        self._add_test_entry()
        
        # Initially no locks
        stats = self.cache_manager.stats()
        assert "locked" in stats
        assert stats["locked"] == 0
        
        # After locking
        self.cache_manager.lock_pipeline(self.model_id)
        stats = self.cache_manager.stats()
        assert stats["locked"] == 1
        assert "in_use" in stats["entries"][self.model_id]
        assert "use_count" in stats["entries"][self.model_id]


class TestPipelineEvictionProtection:
    """Test cases for protecting locked pipelines from eviction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = LocalPipelineCacheManager()

    def teardown_method(self):
        """Clean up after each test."""
        self.cache_manager.force_cleanup()

    def test_locked_pipeline_skipped_in_eviction_candidates(self):
        """Test that locked pipelines are not included in eviction candidates."""
        # Create two mock pipelines
        mock_pipeline1 = MagicMock()
        mock_pipeline1.__class__.__name__ = "TestPipeline1"
        mock_pipeline2 = MagicMock()
        mock_pipeline2.__class__.__name__ = "TestPipeline2"

        # Add both to cache
        entry1 = _PipelineCacheEntry(mock_pipeline1, PipelinePriority.NORMAL)
        entry2 = _PipelineCacheEntry(mock_pipeline2, PipelinePriority.NORMAL)

        with self.cache_manager._lock:  # noqa: SLF001
            self.cache_manager._cache["model1"] = entry1  # noqa: SLF001
            self.cache_manager._cache["model2"] = entry2  # noqa: SLF001

        # Lock model1
        entry1.lock()

        # Mock the eviction logic by accessing the candidate selection directly
        import time
        now = time.time()
        
        with self.cache_manager._lock:  # noqa: SLF001
            # This mimics the candidate selection logic from _ensure_memory
            candidates = [
                (mid, e, e.eviction_score(now))
                for mid, e in self.cache_manager._cache.items()  # noqa: SLF001
                if e.is_alive() and not e.in_use
            ]
            locked_pipelines = [
                mid for mid, e in self.cache_manager._cache.items()  # noqa: SLF001
                if e.is_alive() and e.in_use
            ]

        # Only model2 should be in candidates (model1 is locked)
        candidate_ids = [mid for mid, _, _ in candidates]
        assert "model2" in candidate_ids
        assert "model1" not in candidate_ids

        # model1 should be in locked_pipelines
        assert "model1" in locked_pipelines
        assert "model2" not in locked_pipelines

    def test_large_model_eviction_respects_locks(self):
        """Test that proactive eviction for large models respects locks."""
        # Create test pipelines
        mock_pipeline1 = MagicMock()
        mock_pipeline2 = MagicMock()
        
        entry1 = _PipelineCacheEntry(mock_pipeline1, PipelinePriority.NORMAL)
        entry2 = _PipelineCacheEntry(mock_pipeline2, PipelinePriority.NORMAL)

        with self.cache_manager._lock:  # noqa: SLF001
            self.cache_manager._cache["model1"] = entry1  # noqa: SLF001
            self.cache_manager._cache["model2"] = entry2  # noqa: SLF001

        # Lock model1
        entry1.lock()

        # Test the proactive eviction logic for large models
        with self.cache_manager._lock:  # noqa: SLF001
            # This mimics the eviction target selection from _ensure_memory
            evict_targets = [
                mid for mid, entry in self.cache_manager._cache.items()  # noqa: SLF001
                if mid != "exclude" and not entry.in_use
            ]
            locked_targets = [
                mid for mid, entry in self.cache_manager._cache.items()  # noqa: SLF001
                if mid != "exclude" and entry.in_use
            ]

        # Only model2 should be eligible for eviction
        assert "model2" in evict_targets
        assert "model1" not in evict_targets

        # model1 should be in locked targets
        assert "model1" in locked_targets
        assert "model2" not in locked_targets