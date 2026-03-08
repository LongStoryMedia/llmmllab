"""
Unit tests for runner/pipeline_factory.py.

Tests pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import Optional, Type

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from runner.pipeline_factory import PipelineFactory
from runner.models import Model, ModelProfile, ModelProvider, ModelTask, PipelinePriority
from runner.pipelines.base import BasePipeline


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock(spec=Model)
    model.id = "test-model"
    model.name = "test-model"
    model.provider = ModelProvider.LLAMA_CPP
    model.task = ModelTask.TEXTTOTEXT
    model.pipeline = "ChatLlamaCppPipeline"
    return model


@pytest.fixture
def mock_model_remote():
    """Create a mock remote model."""
    model = MagicMock(spec=Model)
    model.id = "openai-gpt4"
    model.name = "gpt-4"
    model.provider = ModelProvider.OPENAI
    model.task = ModelTask.TEXTTOTEXT
    return model


@pytest.fixture
def mock_model_embedding():
    """Create a mock embedding model."""
    model = MagicMock(spec=Model)
    model.id = "embed-model"
    model.name = "embed-model"
    model.provider = ModelProvider.LLAMA_CPP
    model.task = ModelTask.TEXTTOEMBEDDINGS
    return model


@pytest.fixture
def mock_model_image():
    """Create a mock image model."""
    model = MagicMock(spec=Model)
    model.id = "flux-model"
    model.name = "flux-model"
    model.provider = ModelProvider.LLAMA_CPP
    model.task = ModelTask.TEXTTOIMAGE
    model.pipeline = "FluxPipeline"
    return model


@pytest.fixture
def mock_model_image_to_image():
    """Create a mock image-to-image model."""
    model = MagicMock(spec=Model)
    model.id = "flux-kontext-model"
    model.name = "flux-kontext-model"
    model.provider = ModelProvider.LLAMA_CPP
    model.task = ModelTask.IMAGETOIMAGE
    model.pipeline = "FluxKontextPipeline"
    return model


@pytest.fixture
def mock_model_profile(mocker):
    """Create a mock model profile."""
    profile = MagicMock(spec=ModelProfile)
    profile.model_name = "test-model"
    profile.temperature = 0.7
    profile.max_tokens = 100
    profile.id = "profile-123"
    return profile


@pytest.fixture
def mock_local_cache(mocker):
    """Mock local pipeline cache."""
    mock_cache = MagicMock()
    mock_cache.get_or_create = MagicMock()
    mock_cache.unlock_pipeline = MagicMock(return_value=True)
    mock_cache.set_persistent = MagicMock(return_value=True)
    mock_cache.clear_cache = MagicMock()
    mock_cache.is_local = MagicMock(return_value=True)
    mock_cache.lock_pipeline = MagicMock()
    return mock_cache


@pytest.fixture
def pipeline_factory(mock_local_cache, mocker):
    """Create a PipelineFactory instance with mocked cache."""
    mocker.patch('runner.pipeline_factory.LocalPipelineCacheManager', return_value=mock_local_cache)
    mocker.patch('runner.pipeline_factory._GLOBAL_PIPELINE_CACHE', None)
    mocker.patch('runner.pipeline_factory.ModelLoader')
    factory = PipelineFactory({})
    return factory


class TestPipelineFactoryInitialization:
    """Tests for PipelineFactory initialization."""

    def test_factory_initializes_with_models_map(self, mocker):
        """Test factory initialization with models map."""
        models_map = {"model1": MagicMock(spec=Model)}
        mocker.patch('runner.pipeline_factory.LocalPipelineCacheManager')
        mocker.patch('runner.pipeline_factory._GLOBAL_PIPELINE_CACHE', None)

        factory = PipelineFactory(models_map)

        assert factory.models == models_map

    def test_factory_uses_available_models(self, mocker):
        """Test factory uses available models when models_map is empty."""
        mocker.patch('runner.pipeline_factory.LocalPipelineCacheManager')
        mocker.patch('runner.pipeline_factory._GLOBAL_PIPELINE_CACHE', None)

        mock_model = MagicMock(spec=Model)
        mock_model.id = "model1"
        mock_model.name = "model1"
        mock_model.provider = ModelProvider.LLAMA_CPP
        mock_model.task = ModelTask.TEXTTOTEXT
        mock_model.pipeline = "ChatLlamaCppPipeline"
        mock_loader = MagicMock()
        mock_loader.get_available_models.return_value = {"model1": mock_model}
        mocker.patch('runner.pipeline_factory.ModelLoader', return_value=mock_loader)

        factory = PipelineFactory({})

        assert factory._available_models == {"model1": mock_model}

    def test_factory_uses_global_cache_when_available(self, mocker):
        """Test factory uses global pipeline cache when available."""
        mock_global_cache = MagicMock()
        mocker.patch('runner.pipeline_factory._GLOBAL_PIPELINE_CACHE', mock_global_cache)

        factory = PipelineFactory({})

        assert factory.local_cache == mock_global_cache


class TestGetPipeline:
    """Tests for get_pipeline method."""

    def test_get_pipeline_local_provider(self, pipeline_factory, mock_model, mock_model_profile, mocker):
        """Test get_pipeline with local provider."""
        mock_pipeline = MagicMock(spec=BasePipeline)
        pipeline_factory.local_cache.get_or_create = MagicMock(return_value=mock_pipeline)
        pipeline_factory._available_models = {mock_model.id: mock_model}

        pipeline = pipeline_factory.get_pipeline(mock_model_profile)

        assert pipeline == mock_pipeline
        pipeline_factory.local_cache.get_or_create.assert_called_once()

    def test_get_pipeline_remote_provider(self, pipeline_factory, mock_model_remote, mocker):
        """Test get_pipeline with remote provider."""
        mock_pipeline = MagicMock()
        mocker.patch.object(pipeline_factory, 'create_pipeline', return_value=mock_pipeline)
        pipeline_factory._available_models = {mock_model_remote.id: mock_model_remote}

        # Create a profile with the correct model_name matching the remote model
        profile = MagicMock()
        profile.model_name = mock_model_remote.id

        pipeline = pipeline_factory.get_pipeline(profile)

        assert pipeline == mock_pipeline
        pipeline_factory.create_pipeline.assert_called_once()

    def test_get_pipeline_model_not_found(self, pipeline_factory):
        """Test get_pipeline raises error when model not found."""
        profile = MagicMock()
        profile.model_name = "nonexistent-model"

        with pytest.raises(RuntimeError, match="Model with ID 'nonexistent-model' not found."):
            pipeline_factory.get_pipeline(profile)

    def test_get_pipeline_local_cache_failure(self, pipeline_factory, mock_model, mock_model_profile):
        """Test get_pipeline raises error when cache fails."""
        pipeline_factory.local_cache.get_or_create = MagicMock(return_value=None)
        pipeline_factory._available_models = {mock_model.id: mock_model}

        with pytest.raises(RuntimeError, match="Failed to create cached pipeline"):
            pipeline_factory.get_pipeline(mock_model_profile)


class TestCreatePipeline:
    """Tests for create_pipeline method."""

    def test_create_pipeline_text_to_text(self, pipeline_factory, mock_model, mock_model_profile, mocker):
        """Test create_pipeline for text-to-text task."""
        mock_pipeline = MagicMock(spec=BasePipeline)
        mocker.patch.object(pipeline_factory, '_create_text_pipeline', return_value=mock_pipeline)

        result = pipeline_factory.create_pipeline(mock_model, mock_model_profile)

        assert result == mock_pipeline
        pipeline_factory._create_text_pipeline.assert_called_once()

    def test_create_pipeline_text_to_embeddings(self, pipeline_factory, mock_model_embedding, mock_model_profile, mocker):
        """Test create_pipeline for text-to-embeddings task."""
        mock_pipeline = MagicMock()
        mocker.patch.object(pipeline_factory, '_create_embedding_pipeline', return_value=mock_pipeline)
        mock_model_embedding.task = ModelTask.TEXTTOEMBEDDINGS

        result = pipeline_factory.create_pipeline(mock_model_embedding, mock_model_profile)

        assert result == mock_pipeline
        pipeline_factory._create_embedding_pipeline.assert_called_once()

    def test_create_pipeline_text_to_image(self, pipeline_factory, mock_model_image, mock_model_profile, mocker):
        """Test create_pipeline for text-to-image task."""
        mock_pipeline = MagicMock()
        mocker.patch.object(pipeline_factory, '_create_image_pipeline', return_value=mock_pipeline)
        mock_model_image.task = ModelTask.TEXTTOIMAGE

        result = pipeline_factory.create_pipeline(mock_model_image, mock_model_profile)

        assert result == mock_pipeline
        pipeline_factory._create_image_pipeline.assert_called_once()

    def test_create_pipeline_unsupported_task(self, pipeline_factory, mock_model):
        """Test create_pipeline raises error for unsupported task."""
        mock_model.task = "UnsupportedTask"

        with pytest.raises(RuntimeError, match="Unsupported task type"):
            pipeline_factory.create_pipeline(mock_model, MagicMock())


class TestCreateTextPipeline:
    """Tests for _create_text_pipeline method."""

    def test_create_text_pipeline_llama_cpp(self, pipeline_factory, mock_model, mock_model_profile, mocker):
        """Test create_text_pipeline for llama.cpp provider."""
        mock_model.provider = ModelProvider.LLAMA_CPP

        # Mock the ChatLlamaCppPipeline to avoid real pipeline initialization
        mock_pipeline = MagicMock(spec=BasePipeline)
        mocker.patch('runner.pipelines.llamacpp.chat.ChatLlamaCppPipeline', return_value=mock_pipeline)

        pipeline = pipeline_factory._create_text_pipeline(mock_model, mock_model_profile)

        assert pipeline is not None
        assert pipeline == mock_pipeline

    def test_create_text_pipeline_openai(self, pipeline_factory, mock_model, mock_model_profile, mocker):
        """Test create_text_pipeline for OpenAI provider."""
        mock_model.provider = ModelProvider.OPENAI
        mocker.patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})

        pipeline = pipeline_factory._create_text_pipeline(mock_model, mock_model_profile)

        assert pipeline is not None

    def test_create_text_pipeline_anthropic(self, pipeline_factory, mock_model, mock_model_profile, mocker):
        """Test create_text_pipeline for Anthropic provider."""
        mock_model.provider = ModelProvider.ANTHROPIC
        mocker.patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})

        # Mock the ChatAnthropic class - since it's imported inside the function,
        # we need to patch the import system using importlib
        mock_chat_anthropic = MagicMock()

        # Create a mock module that will be returned when langchain_anthropic is imported
        mock_module = MagicMock()
        mock_module.ChatAnthropic = mock_chat_anthropic

        # Patch sys.modules to return our mock when langchain_anthropic is imported
        mocker.patch.dict('sys.modules', {'langchain_anthropic': mock_module})

        pipeline = pipeline_factory._create_text_pipeline(mock_model, mock_model_profile)

        assert pipeline is not None
        mock_chat_anthropic.assert_called_once()

    def test_create_text_pipeline_unsupported_provider(self, pipeline_factory, mock_model, mock_model_profile):
        """Test create_text_pipeline raises error for unsupported provider."""
        mock_model.provider = "UnsupportedProvider"

        with pytest.raises(ValueError, match="Unsupported text provider"):
            pipeline_factory._create_text_pipeline(mock_model, mock_model_profile)


class TestCreateEmbeddingPipeline:
    """Tests for _create_embedding_pipeline method."""

    def test_create_embedding_pipeline(self, pipeline_factory, mock_model_embedding, mock_model_profile, mocker):
        """Test create_embedding_pipeline."""
        mock_model_embedding.task = ModelTask.TEXTTOEMBEDDINGS

        # Mock the EmbedLlamaCppPipeline to avoid real pipeline initialization
        mock_pipeline = MagicMock()
        mocker.patch('runner.pipelines.llamacpp.embed.EmbedLlamaCppPipeline', return_value=mock_pipeline)

        pipeline = pipeline_factory._create_embedding_pipeline(mock_model_embedding, mock_model_profile)

        assert pipeline is not None
        assert pipeline == mock_pipeline


class TestCreateImagePipeline:
    """Tests for _create_image_pipeline method."""

    def test_create_image_pipeline_flux(self, pipeline_factory, mock_model_image, mock_model_profile, mocker):
        """Test create_image_pipeline for Flux pipeline."""
        mock_model_image.pipeline = "FluxPipeline"

        # Mock the FluxPipe to avoid real pipeline initialization
        # The import is relative: from .pipelines.txt2img.flux import FluxPipe
        # We patch it in the pipeline_factory module namespace after import
        mock_pipeline = MagicMock()

        # Create a mock module for txt2img.flux with FluxPipe as a callable that returns our mock
        mock_flux_module = MagicMock()
        mock_flux_module.FluxPipe = MagicMock(return_value=mock_pipeline)

        # Patch sys.modules to return our mock when the module is imported
        mocker.patch.dict('sys.modules', {'runner.pipelines.txt2img.flux': mock_flux_module})

        pipeline = pipeline_factory._create_image_pipeline(mock_model_image, mock_model_profile)

        assert pipeline is not None
        assert pipeline == mock_pipeline

    def test_create_image_pipeline_unsupported(self, pipeline_factory, mock_model_image, mock_model_profile):
        """Test create_image_pipeline for unsupported pipeline type."""
        mock_model_image.pipeline = "UnsupportedPipeline"

        pipeline = pipeline_factory._create_image_pipeline(mock_model_image, mock_model_profile)

        assert pipeline is None


class TestCreateImageToImagePipeline:
    """Tests for _create_image_to_image_pipeline method."""

    def test_create_image_to_image_pipeline_flux_kontext(self, pipeline_factory, mock_model_image_to_image, mock_model_profile, mocker):
        """Test create_image_to_image_pipeline for FluxKontext pipeline."""
        mock_model_image_to_image.pipeline = "FluxKontextPipeline"

        # Mock the FluxKontextPipe to avoid real pipeline initialization
        # The import is relative: from .pipelines.img2img.flux import FluxKontextPipe
        # We patch it in the pipeline_factory module namespace after import
        mock_pipeline = MagicMock()

        # Create a mock module for img2img.flux with FluxKontextPipe as a callable that returns our mock
        mock_flux_module = MagicMock()
        mock_flux_module.FluxKontextPipe = MagicMock(return_value=mock_pipeline)

        # Patch sys.modules to return our mock when the module is imported
        mocker.patch.dict('sys.modules', {'runner.pipelines.img2img.flux': mock_flux_module})

        pipeline = pipeline_factory._create_image_to_image_pipeline(mock_model_image_to_image, mock_model_profile)

        assert pipeline is not None
        assert pipeline == mock_pipeline

    def test_create_image_to_image_pipeline_unsupported(self, pipeline_factory, mock_model_image_to_image, mock_model_profile):
        """Test create_image_to_image_pipeline for unsupported pipeline type."""
        mock_model_image_to_image.pipeline = "UnsupportedPipeline"

        pipeline = pipeline_factory._create_image_to_image_pipeline(mock_model_image_to_image, mock_model_profile)

        assert pipeline is None


class TestGetEmbeddingPipeline:
    """Tests for get_embedding_pipeline method."""

    def test_get_embedding_pipeline_local(self, pipeline_factory, mock_model_embedding, mock_model_profile, mocker):
        """Test get_embedding_pipeline with local provider."""
        mock_pipeline = MagicMock(spec=Embeddings)
        pipeline_factory.local_cache.get_or_create = MagicMock(return_value=mock_pipeline)

        # Use a model that matches the profile's model_name
        mock_model_embedding.id = "test-model"
        mock_model_embedding.name = "test-model"
        pipeline_factory._available_models = {mock_model_embedding.id: mock_model_embedding}

        pipeline = pipeline_factory.get_embedding_pipeline(mock_model_profile)

        assert pipeline == mock_pipeline
        assert isinstance(pipeline, Embeddings)

    def test_get_embedding_pipeline_not_embedding_model(self, pipeline_factory, mock_model, mock_model_profile):
        """Test get_embedding_pipeline raises error for non-embedding model."""
        pipeline_factory._available_models = {mock_model.id: mock_model}

        with pytest.raises(ValueError, match="is not an embedding model"):
            pipeline_factory.get_embedding_pipeline(mock_model_profile)

    def test_get_embedding_pipeline_wrong_type(self, pipeline_factory, mock_model_embedding, mock_model_profile, mocker):
        """Test get_embedding_pipeline raises error when pipeline is not Embeddings."""
        mock_pipeline = MagicMock()
        pipeline_factory.local_cache.get_or_create = MagicMock(return_value=mock_pipeline)

        # Use a model that matches the profile's model_name
        mock_model_embedding.id = "test-model"
        mock_model_embedding.name = "test-model"
        pipeline_factory._available_models = {mock_model_embedding.id: mock_model_embedding}

        with pytest.raises(ValueError, match="Expected Embeddings instance"):
            pipeline_factory.get_embedding_pipeline(mock_model_profile)


class TestUnlockPipeline:
    """Tests for unlock_pipeline method."""

    def test_unlock_pipeline_local(self, pipeline_factory, mock_model, mock_model_profile):
        """Test unlock_pipeline with local provider."""
        pipeline_factory.local_cache.unlock_pipeline = MagicMock(return_value=True)
        pipeline_factory._available_models = {mock_model.id: mock_model}

        result = pipeline_factory.unlock_pipeline(mock_model_profile)

        assert result is True
        pipeline_factory.local_cache.unlock_pipeline.assert_called_once()

    def test_unlock_pipeline_remote(self, pipeline_factory, mock_model_remote, mock_model_profile, mocker):
        """Test unlock_pipeline with remote provider returns True."""
        # Mock the _get_model_by_id to return the remote model
        mocker.patch.object(pipeline_factory, '_get_model_by_id', return_value=mock_model_remote)
        pipeline_factory._available_models = {mock_model_remote.id: mock_model_remote}

        result = pipeline_factory.unlock_pipeline(mock_model_profile)

        assert result is True


class TestSetPipelinePersistent:
    """Tests for set_pipeline_persistent method."""

    def test_set_pipeline_persistent_local(self, pipeline_factory, mock_model, mock_model_profile):
        """Test set_pipeline_persistent with local provider."""
        pipeline_factory.local_cache.set_persistent = MagicMock(return_value=True)
        pipeline_factory._available_models = {mock_model.id: mock_model}

        result = pipeline_factory.set_pipeline_persistent(mock_model_profile, True)

        assert result is True
        pipeline_factory.local_cache.set_persistent.assert_called_once()

    def test_set_pipeline_persistent_remote(self, pipeline_factory, mock_model_remote, mock_model_profile, mocker):
        """Test set_pipeline_persistent with remote provider returns True."""
        # Mock the _get_model_by_id to return the remote model
        mocker.patch.object(pipeline_factory, '_get_model_by_id', return_value=mock_model_remote)
        pipeline_factory._available_models = {mock_model_remote.id: mock_model_remote}

        result = pipeline_factory.set_pipeline_persistent(mock_model_profile, True)

        assert result is True


class TestForceEvictPipeline:
    """Tests for force_evict_pipeline method."""

    def test_force_evict_pipeline_local(self, pipeline_factory, mock_model, mock_model_profile):
        """Test force_evict_pipeline with local provider."""
        pipeline_factory.local_cache.clear_cache = MagicMock()
        pipeline_factory._available_models = {mock_model.id: mock_model}

        result = pipeline_factory.force_evict_pipeline(mock_model_profile)

        assert result is True
        pipeline_factory.local_cache.clear_cache.assert_called_once()

    def test_force_evict_pipeline_remote(self, pipeline_factory, mock_model_remote, mock_model_profile):
        """Test force_evict_pipeline with remote provider returns False."""
        pipeline_factory._available_models = {mock_model_remote.id: mock_model_remote}

        result = pipeline_factory.force_evict_pipeline(mock_model_profile)

        assert result is False


class TestGetCacheStats:
    """Tests for get_cache_stats method."""

    def test_get_cache_stats(self, pipeline_factory):
        """Test get_cache_stats returns cache info."""
        expected_stats = {"cache_size": 10, "hits": 100, "misses": 50}
        pipeline_factory.local_cache.get_cache_info = MagicMock(return_value=expected_stats)

        stats = pipeline_factory.get_cache_stats()

        assert stats == expected_stats


class TestPipelineContextManager:
    """Tests for pipeline context manager."""

    def test_pipeline_context_manager(self, pipeline_factory, mock_model, mock_model_profile, mocker):
        """Test pipeline context manager locks and unlocks properly."""
        mock_pipeline = MagicMock()
        mocker.patch.object(pipeline_factory, 'get_pipeline', return_value=mock_pipeline)
        pipeline_factory._available_models = {mock_model.id: mock_model}

        with pipeline_factory.pipeline(mock_model_profile) as pipeline:
            assert pipeline == mock_pipeline

        # Verify unlock was called after context exit
        pipeline_factory.local_cache.unlock_pipeline.assert_called_once()


class TestGetModelById:
    """Tests for _get_model_by_id method."""

    def test_get_model_by_id_found(self, pipeline_factory, mock_model):
        """Test get_model_by_id returns model when found."""
        pipeline_factory._available_models = {mock_model.id: mock_model}

        model = pipeline_factory._get_model_by_id(mock_model.id)

        assert model == mock_model

    def test_get_model_by_id_not_found(self, pipeline_factory):
        """Test get_model_by_id returns None when model not found."""
        pipeline_factory._available_models = {}

        model = pipeline_factory._get_model_by_id("nonexistent")

        assert model is None

    def test_get_model_by_id_empty_models(self, pipeline_factory):
        """Test get_model_by_id handles empty models dictionary."""
        pipeline_factory._available_models = {}

        model = pipeline_factory._get_model_by_id("test")

        assert model is None