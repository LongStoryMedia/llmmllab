"""
Unit tests for PipelineFactory integration with pipeline locking.

Tests that the pipeline factory properly integrates with the cache locking
system to prevent eviction during active inference.
"""

import pytest
from unittest.mock import MagicMock, patch

from runner.pipeline_factory import PipelineFactory
from models import ModelProfile, PipelinePriority, ModelProvider, ModelParameters


class TestPipelineFactoryLocking:
    """Test cases for pipeline factory integration with locking."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a factory with mocked models
        self.factory = PipelineFactory({})
        
        # Mock a local model
        self.mock_local_model = MagicMock()
        self.mock_local_model.id = "test-local-model"
        self.mock_local_model.name = "Test Local Model"
        self.mock_local_model.provider = ModelProvider.LLAMA_CPP
        self.mock_local_model.task = "TextToText"
        self.mock_local_model.pipeline = "TestPipe"
        
        # Mock a remote model
        self.mock_remote_model = MagicMock()
        self.mock_remote_model.id = "test-remote-model"
        self.mock_remote_model.name = "Test Remote Model"
        self.mock_remote_model.provider = "OPENAI"  # Non-local provider
        self.mock_remote_model.task = "TextToText"
        
        # Add models to factory
        self.factory._available_models["test-local-model"] = self.mock_local_model
        self.factory._available_models["test-remote-model"] = self.mock_remote_model

    def teardown_method(self):
        """Clean up after each test."""
        self.factory.local_cache.force_cleanup()

    def _create_test_profile(self, model_name: str) -> ModelProfile:
        """Helper to create a test model profile with required fields."""
        return ModelProfile(
            user_id="test-user",
            name="Test Profile",
            model_name=model_name,
            parameters=ModelParameters(
                temperature=0.7,
                num_ctx=2048,
                num_predict=-1,
                top_k=40,
                top_p=0.9,
                min_p=0.05,
                repeat_penalty=1.1
            ),
            system_prompt="You are a helpful assistant",
            type=0  # ModelProfileTypePrimary
        )

    @patch('runner.pipeline_factory.PipelineFactory.create_pipeline')
    def test_context_manager_locks_local_pipeline(self, mock_create):
        """Test that context manager automatically locks and unlocks local pipelines."""
        # Mock the pipeline creation
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        
        # Mock the cache methods
        with patch.object(self.factory.local_cache, 'is_local', return_value=True):
            with patch.object(self.factory.local_cache, '_ensure_memory', return_value=True):
                with patch.object(self.factory.local_cache, 'lock_pipeline', return_value=True) as mock_lock:
                    with patch.object(self.factory.local_cache, 'unlock_pipeline', return_value=True) as mock_unlock:
                        
                        profile = self._create_test_profile("test-local-model")
                        
                        # Use the context manager
                        with self.factory.pipeline(profile) as pipeline:
                            assert pipeline == mock_pipeline
                        
                        # Verify locking was called during get_pipeline()
                        mock_lock.assert_called_once_with("test-local-model")
                        # Verify unlocking was called when exiting context
                        mock_unlock.assert_called_once_with("test-local-model")

    @patch('runner.pipeline_factory.PipelineFactory.create_pipeline')
    def test_context_manager_skips_locking_remote_pipeline(self, mock_create):
        """Test that context manager skips locking for remote pipelines."""
        # Mock the pipeline creation
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        
        # Mock the cache to return remote model as not local
        with patch.object(self.factory.local_cache, 'is_local', return_value=False):
            with patch.object(self.factory.local_cache, 'lock_pipeline') as mock_lock:
                with patch.object(self.factory.local_cache, 'unlock_pipeline') as mock_unlock:
                    
                    profile = self._create_test_profile("test-remote-model")
                    
                    # Use the context manager
                    with self.factory.pipeline(profile) as pipeline:
                        assert pipeline == mock_pipeline
                    
                    # Verify locking was NOT attempted for remote pipeline
                    mock_lock.assert_not_called()
                    mock_unlock.assert_not_called()

    @patch('runner.pipeline_factory.PipelineFactory.create_pipeline')
    def test_get_pipeline_locks_local_automatically(self, mock_create):
        """Test that get_pipeline automatically locks local pipelines."""
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        
        # Mock local cache methods
        with patch.object(self.factory.local_cache, '_ensure_memory', return_value=True):
            with patch.object(self.factory.local_cache, 'lock_pipeline', return_value=True) as mock_lock:
                
                profile = self._create_test_profile("test-local-model")
                result_pipeline = self.factory.get_pipeline(profile)
                
                assert result_pipeline == mock_pipeline
                mock_lock.assert_called_once_with("test-local-model")

    @patch('runner.pipeline_factory.PipelineFactory.create_pipeline')
    def test_get_pipeline_skips_locking_remote(self, mock_create):
        """Test that get_pipeline skips locking for remote pipelines."""
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        
        with patch.object(self.factory.local_cache, 'lock_pipeline') as mock_lock:
            
            profile = self._create_test_profile("test-remote-model")
            result_pipeline = self.factory.get_pipeline(profile)
            
            assert result_pipeline == mock_pipeline
            mock_lock.assert_not_called()

    def test_unlock_pipeline_local(self):
        """Test that unlock_pipeline works for local pipelines."""
        with patch.object(self.factory.local_cache, 'is_local', return_value=True):
            with patch.object(self.factory.local_cache, 'unlock_pipeline', return_value=True) as mock_unlock:
                
                profile = self._create_test_profile("test-local-model")
                result = self.factory.unlock_pipeline(profile)
                
                assert result is True
                mock_unlock.assert_called_once_with("test-local-model")

    def test_unlock_pipeline_remote(self):
        """Test that unlock_pipeline works for remote pipelines (no-op)."""
        with patch.object(self.factory.local_cache, 'is_local', return_value=False):
            with patch.object(self.factory.local_cache, 'unlock_pipeline') as mock_unlock:
                
                profile = self._create_test_profile("test-remote-model")
                result = self.factory.unlock_pipeline(profile)
                
                assert result is True  # Always returns True for remote
                mock_unlock.assert_not_called()

    def test_context_manager_usage_tracking(self):
        """Test that context manager properly tracks active local uses."""
        with patch.object(self.factory, 'get_pipeline') as mock_get:
            mock_pipeline = MagicMock()
            mock_get.return_value = mock_pipeline
            
            with patch.object(self.factory.local_cache, 'is_local', return_value=True):
                with patch.object(self.factory.local_cache, 'unlock_pipeline', return_value=True) as mock_unlock:
                    
                    profile = self._create_test_profile("test-local-model")
                    
                    initial_uses = self.factory._active_local_uses
                    
                    with self.factory.pipeline(profile) as pipeline:
                        # Should increment during use
                        assert self.factory._active_local_uses == initial_uses + 1
                    
                    # Should decrement after use and unlock called
                    assert self.factory._active_local_uses == initial_uses
                    mock_unlock.assert_called_once_with("test-local-model")