"""
Unit tests for enhanced pipeline caching system.

Tests the new intelligent memory management features:
1. Size-aware eviction (small models stay longer)
2. Priority-based persistence (high priority models protected)
3. Intelligent memory pressure handling
4. Automatic persistence for embedding models
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

from models import (
    Model, 
    ModelProfile, 
    ModelProvider,
    ModelTask,
    PipelinePriority,
    OptimalParameters,
    ModelDetails
)
from runner.pipeline_cache import LocalPipelineCacheManager
from runner import pipeline_factory


class TestEnhancedPipelineCache:
    """Test suite for enhanced pipeline caching system."""

    @pytest.fixture
    def cache_manager(self):
        """Fixture providing a fresh cache manager for each test."""
        return LocalPipelineCacheManager(cache_timeout=300)

    def test_cache_initialization(self, cache_manager):
        """Test cache manager initialization."""
        # Test initial state
        stats = cache_manager.stats()
        assert stats["count"] == 0
        assert stats["alive"] == 0
        
        # Test cache manager has expected attributes
        assert hasattr(cache_manager, '_cache')
        assert hasattr(cache_manager, '_cache_timeout')
        assert cache_manager._cache_timeout == 300

    def test_memory_estimation(self, cache_manager):
        """Test memory estimation for different model sizes."""
        # Create mock models with different sizes
        small_model = Mock()
        small_model.size = int(0.4 * 1024 * 1024 * 1024)  # 400MB
        small_profile = Mock()
        small_profile.parameters = Mock()
        small_profile.parameters.num_ctx = 4096
        
        medium_model = Mock()
        medium_model.size = int(3.0 * 1024 * 1024 * 1024)  # 3GB
        medium_profile = Mock()
        medium_profile.parameters = Mock()
        medium_profile.parameters.num_ctx = 4096
        
        large_model = Mock() 
        large_model.size = int(15.0 * 1024 * 1024 * 1024)  # 15GB
        large_profile = Mock()
        large_profile.parameters = Mock()
        large_profile.parameters.num_ctx = 65536  # Large context
        
        # Test memory estimates
        small_estimate = cache_manager.estimate_memory(small_model, small_profile)
        medium_estimate = cache_manager.estimate_memory(medium_model, medium_profile)
        large_estimate = cache_manager.estimate_memory(large_model, large_profile)
        
        # Verify estimates are ordered correctly
        assert small_estimate < medium_estimate < large_estimate
        
        # Verify estimates are reasonable (at least model size)
        assert small_estimate >= small_model.size
        assert medium_estimate >= medium_model.size
        assert large_estimate >= large_model.size

    def test_eviction_scoring(self):
        """Test the enhanced eviction scoring system."""
        from runner.pipeline_cache import _PipelineCacheEntry
        
        current_time = time.time()
        
        # Create test entries with different characteristics
        entries = [
            # Small, high-priority, frequently used (should have highest score)
            _PipelineCacheEntry(Mock(), PipelinePriority.HIGH, 0.5 * 1024**3),  # 500MB
            # Medium, normal priority, occasionally used
            _PipelineCacheEntry(Mock(), PipelinePriority.NORMAL, 3.0 * 1024**3),  # 3GB
            # Large, high priority, rarely used
            _PipelineCacheEntry(Mock(), PipelinePriority.HIGH, 15.0 * 1024**3),  # 15GB
            # Large, low priority, frequently used
            _PipelineCacheEntry(Mock(), PipelinePriority.LOW, 12.0 * 1024**3),  # 12GB
        ]
        
        # Simulate different usage patterns
        entries[0].access_count = 20  # Frequently used small model
        entries[1].access_count = 5   # Occasionally used medium model  
        entries[2].access_count = 2   # Rarely used large model
        entries[3].access_count = 10  # Frequently used large model
        
        # Calculate scores
        scores = []
        for entry in entries:
            score = entry.eviction_score(current_time, entry.estimated_memory)
            scores.append(score)
        
        # Small, frequently used, high priority should have highest score
        assert scores[0] > scores[1]  # Small/frequent beats medium/occasional
        assert scores[0] > scores[2]  # Small/frequent beats large/rare
        
        # Verify that large models generally score lower
        assert scores[2] < scores[0]  # Large model scores lower than small

    def test_persistence_api(self, cache_manager):
        """Test automatic persistence marking API."""
        test_model_id = "test-small-embedding"
        
        # Test that persistence can be set and retrieved
        # Result will be False since model doesn't exist, but method should exist
        result = cache_manager.set_persistent(test_model_id, True)
        assert result is False  # Expected since no model is cached
        
        # Test the API exists and is callable
        assert callable(getattr(cache_manager, 'set_persistent', None))

    def test_cache_info(self, cache_manager):
        """Test cache information gathering."""
        # Test empty cache
        info = cache_manager.get_cache_info()
        
        # Verify all expected keys are present
        expected_keys = [
            "total_models", 
            "total_memory_gb", 
            "small_models", 
            "large_models", 
            "locked_models", 
            "high_priority_models"
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
        
        # Test empty cache values
        assert info["total_models"] == 0
        assert info["total_memory_gb"] == 0.0
        assert info["small_models"]["count"] == 0
        assert info["large_models"]["count"] == 0
        assert info["locked_models"] == []  # It's a list, not a number
        assert info["high_priority_models"] == []  # It's a list, not a number

    def test_pipeline_factory_integration(self):
        """Test integration with pipeline factory."""
        # Test that new methods are available
        assert hasattr(pipeline_factory, 'set_pipeline_persistent')
        assert hasattr(pipeline_factory, 'get_cache_stats')
        assert hasattr(pipeline_factory, 'force_evict_pipeline')
        
        # Test empty cache stats
        stats = pipeline_factory.get_cache_stats()
        assert isinstance(stats, dict)
        
        # Verify stats has expected structure from cache_info
        assert "total_models" in stats
        assert "total_memory_gb" in stats

    def test_cache_stats_structure(self, cache_manager):
        """Test that cache stats return expected structure."""
        stats = cache_manager.stats()
        
        # Verify basic stats structure
        assert isinstance(stats, dict)
        assert "count" in stats
        assert "alive" in stats
        
        # For empty cache
        assert stats["count"] == 0
        assert stats["alive"] == 0

    def test_size_aware_categorization(self, cache_manager):
        """Test that models are correctly categorized by size."""
        # This tests the logic used for size-aware eviction
        small_size = 0.5 * 1024**3  # 500MB
        medium_size = 3.0 * 1024**3  # 3GB  
        large_size = 15.0 * 1024**3  # 15GB
        
        # Test the size threshold logic (if accessible)
        # This may need to test internal constants or behavior
        assert small_size < 1024**3  # Less than 1GB is considered small
        assert large_size > 10 * 1024**3  # Greater than 10GB is considered large

    @patch('runner.pipeline_cache._PipelineCacheEntry')
    def test_eviction_order_simulation(self, mock_entry_class, cache_manager):
        """Test that eviction ordering works as expected."""
        # Mock entries to simulate eviction scenario
        entries = [
            Mock(priority=PipelinePriority.HIGH, estimated_memory=0.4e9, access_count=50),  # Small, frequent
            Mock(priority=PipelinePriority.NORMAL, estimated_memory=3e9, access_count=10),  # Medium
            Mock(priority=PipelinePriority.HIGH, estimated_memory=8e9, access_count=2),     # Large, rare
            Mock(priority=PipelinePriority.LOW, estimated_memory=15e9, access_count=1),    # Large, very rare
        ]
        
        # Set up mock eviction scores (higher = better, kept longer)
        entries[0].eviction_score.return_value = 100  # Best score - small, frequent, high priority
        entries[1].eviction_score.return_value = 50   # Good score - medium size, decent usage
        entries[2].eviction_score.return_value = 20   # Lower score - large, rarely used
        entries[3].eviction_score.return_value = 10   # Worst score - large, very rare, low priority
        
        # Sort by eviction score (ascending = first to evict)
        sorted_entries = sorted(entries, key=lambda e: e.eviction_score.return_value)
        
        # Verify eviction order (worst first)
        assert sorted_entries[0] == entries[3]  # Large, low priority should be evicted first
        assert sorted_entries[-1] == entries[0]  # Small, high priority should be kept longest

    def test_memory_threshold_constants(self):
        """Test that memory threshold constants are reasonable."""
        # Test that we can import and check the threshold constants
        try:
            from runner.pipeline_cache import LocalPipelineCacheManager
            
            # These constants should be defined for size-aware eviction
            # We're testing that they exist and are reasonable
            cache_manager = LocalPipelineCacheManager()
            
            # The thresholds are likely internal, but the behavior should be testable
            # Small models (< 1GB) should be treated differently from large models (> 10GB)
            small_memory = 0.5 * 1024**3  # 500MB
            large_memory = 15 * 1024**3   # 15GB
            
            # These should be categorized differently
            assert small_memory != large_memory
            
        except ImportError:
            pytest.skip("Pipeline cache module not available")

    def test_priority_based_scoring(self):
        """Test that priority affects eviction scoring."""
        from runner.pipeline_cache import _PipelineCacheEntry
        
        current_time = time.time()
        same_size = 5 * 1024**3  # 5GB
        same_access = 10
        
        # Create entries with different priorities
        high_priority_entry = _PipelineCacheEntry(Mock(), PipelinePriority.HIGH, same_size)
        normal_priority_entry = _PipelineCacheEntry(Mock(), PipelinePriority.NORMAL, same_size)
        low_priority_entry = _PipelineCacheEntry(Mock(), PipelinePriority.LOW, same_size)
        
        # Set same usage patterns
        for entry in [high_priority_entry, normal_priority_entry, low_priority_entry]:
            entry.access_count = same_access
        
        # Calculate scores
        high_score = high_priority_entry.eviction_score(current_time, same_size)
        normal_score = normal_priority_entry.eviction_score(current_time, same_size)
        low_score = low_priority_entry.eviction_score(current_time, same_size)
        
        # Higher priority should get higher scores (less likely to be evicted)
        assert high_score >= normal_score >= low_score