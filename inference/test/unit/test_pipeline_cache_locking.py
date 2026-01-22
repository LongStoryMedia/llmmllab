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