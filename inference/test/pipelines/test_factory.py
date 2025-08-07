import pytest
from unittest.mock import patch, MagicMock

from runner.pipelines.factory import PipelineCacheEntry
from models.model import Model
from runner.pipelines.base_pipeline import BasePipeline


class TestPipelineFactory:
    """Test cases for the PipelineFactory class."""

    @patch("pipelines.factory.PipelineFactory.create_pipeline")
    def test_get_pipeline_caching(self, mock_create_pipeline):
        """Test that get_pipeline properly caches and returns pipelines."""
        # Arrange
        mock_pipeline = MagicMock(spec=BasePipeline)
        mock_create_pipeline.return_value = mock_pipeline
        model_id = "test-model-1"

        # Create a mock model for the _available_models dictionary
        mock_model = MagicMock(spec=Model)
        mock_model.id = model_id
        mock_model.name = "Test Model 1"

        # Set up the available models dictionary
        mock_pipeline._available_models = {model_id: mock_model}
        # Clear any existing pipelines
        mock_pipeline._pipelines = {}

        # Act - first call should create the pipeline
        result1 = mock_pipeline.get_pipeline(model_id)

        # Assert
        assert result1 == mock_pipeline
        mock_create_pipeline.assert_called_once_with(mock_model)
        assert model_id in mock_pipeline._pipelines
        assert mock_pipeline._pipelines[model_id].pipeline == mock_pipeline

        # Reset the mock to check second call
        mock_create_pipeline.reset_mock()

        # Act - second call should use cached pipeline
        result2 = mock_pipeline.get_pipeline(model_id)

        # Assert
        assert result2 == mock_pipeline
        # create_pipeline should not be called again
        mock_create_pipeline.assert_not_called()

    @patch("pipelines.factory.time")
    def test_pipeline_cache_entry_timestamp(self, mock_time):
        """Test that PipelineCacheEntry correctly records and updates timestamps."""
        # Arrange
        mock_pipeline = MagicMock(spec=BasePipeline)
        mock_time.time.return_value = 1000.0

        # Act - create entry with default timestamp
        entry = PipelineCacheEntry(mock_pipeline)

        # Assert
        assert entry.pipeline == mock_pipeline
        assert entry.last_accessed == 1000.0

        # Arrange - new timestamp
        mock_time.time.return_value = 2000.0

        # Act - update timestamp
        entry.last_accessed = mock_time.time()

        # Assert
        assert entry.last_accessed == 2000.0

        # Act - create entry with explicit timestamp
        explicit_entry = PipelineCacheEntry(mock_pipeline, timestamp=1500.0)

        # Assert
        assert explicit_entry.last_accessed == 1500.0

    @patch("threading.Thread")
    @patch("pipelines.factory.time")
    def test_cleanup_expired_entries(self, mock_time, mock_thread):
        """Test that expired entries are correctly cleaned up."""
        # Arrange
        mock_time.time.return_value = 1000.0
        mock_pipeline = MagicMock(spec=BasePipeline)

        # Create mock pipelines
        mock_pipeline1 = MagicMock(spec=BasePipeline)
        mock_pipeline1.model = MagicMock()
        mock_pipeline2 = MagicMock(spec=BasePipeline)
        mock_pipeline2.model = MagicMock()
        mock_pipeline3 = MagicMock(spec=BasePipeline)
        mock_pipeline3.model = MagicMock()

        # Set up cache entries with different timestamps
        mock_pipeline._pipelines = {
            "model1": PipelineCacheEntry(mock_pipeline1, timestamp=800.0),  # expired
            "model2": PipelineCacheEntry(
                mock_pipeline2, timestamp=950.0
            ),  # not expired
            "model3": PipelineCacheEntry(mock_pipeline3, timestamp=600.0),  # expired
        }

        # Set cache timeout to 150 seconds
        mock_pipeline._cache_timeout = 150

        # Act
        mock_pipeline._cleanup_expired_entries()

        # Assert
        assert "model2" in mock_pipeline._pipelines  # should still be in cache
        assert "model1" not in mock_pipeline._pipelines  # should be removed
        assert "model3" not in mock_pipeline._pipelines  # should be removed

        # Check that resource cleanup was called for expired pipelines
        mock_pipeline1.model.to.assert_called_once_with("cpu")
        mock_pipeline3.model.to.assert_called_once_with("cpu")
        mock_pipeline2.model.to.assert_not_called()  # not expired, should not be cleaned up

    def test_set_cache_timeout(self):
        """Test setting the cache timeout."""
        mock_pipeline = MagicMock(spec=BasePipeline)
        # Arrange
        original_timeout = mock_pipeline._cache_timeout
        new_timeout = 600  # 10 minutes

        # Act
        mock_pipeline.set_cache_timeout(new_timeout)

        # Assert
        assert mock_pipeline._cache_timeout == new_timeout

        # Cleanup - restore original timeout
        mock_pipeline._cache_timeout = original_timeout

    def test_clear_cache(self):
        """Test manually clearing the cache."""
        mock_pipeline = MagicMock(spec=BasePipeline)

        # Arrange
        mock_pipeline1 = MagicMock(spec=BasePipeline)
        mock_pipeline1.model = MagicMock()
        mock_pipeline2 = MagicMock(spec=BasePipeline)
        mock_pipeline2.model = MagicMock()

        mock_pipeline._pipelines = {
            "model1": PipelineCacheEntry(mock_pipeline1),
            "model2": PipelineCacheEntry(mock_pipeline2),
        }

        # Act - clear specific model
        mock_pipeline.clear_cache("model1")

        # Assert
        assert "model1" not in mock_pipeline._pipelines
        assert "model2" in mock_pipeline._pipelines
        mock_pipeline1.model.to.assert_called_once_with("cpu")
        mock_pipeline2.model.to.assert_not_called()

        # Reset mocks
        mock_pipeline1.reset_mock()
        mock_pipeline2.reset_mock()

        # Act - clear all
        mock_pipeline.clear_cache()

        # Assert
        assert len(mock_pipeline._pipelines) == 0
        mock_pipeline2.model.to.assert_called_once_with("cpu")

    @patch("importlib.import_module")
    @patch("pipelines.factory.logging.getLogger")
    def test_pipeline_validation(self, mock_logger, mock_import_module):
        """Test that pipelines are validated as BasePipeline instances."""
        mock_pipeline = MagicMock(spec=BasePipeline)
        # Arrange
        model_id = "test-model-validation"
        mock_model = MagicMock(spec=Model)
        mock_model.id = model_id
        mock_model.name = "Test Model Validation"
        mock_model.task = "TextToImage"
        mock_model.pipeline = "StableDiffusion3Pipeline"

        # Setup mock non-BasePipeline object
        mock_non_base_pipeline = MagicMock()

        # Mock BasePipeline class and isinstance check
        mock_base_pipeline_class = MagicMock()
        mock_base_pipeline_module = MagicMock()
        mock_base_pipeline_module.BasePipeline = mock_base_pipeline_class
        mock_import_module.return_value = mock_base_pipeline_module

        # Setup logger
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Setup PipelineFactory
        mock_pipeline._available_models = {model_id: mock_model}

        # Act & Assert - Test with non-BasePipeline instance
        with patch("pipelines.factory.SD3Pipe", return_value=mock_non_base_pipeline):
            with pytest.raises(TypeError) as excinfo:
                mock_pipeline.create_pipeline(mock_model)

            assert "Pipeline must be an instance of BasePipeline" in str(excinfo.value)
            mock_logger_instance.error.assert_called_once()

    def test_cleanup_calls_del_method(self):
        """Test that _cleanup_pipeline_resources calls the pipeline's __del__ method."""
        # Arrange
        mock_pipeline = MagicMock(spec=BasePipeline)

        # Act
        mock_pipeline._cleanup_pipeline_resources(mock_pipeline)

        # Assert
        mock_pipeline.__del__.assert_called_once()
