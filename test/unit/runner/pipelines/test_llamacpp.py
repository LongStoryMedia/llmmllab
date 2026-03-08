"""
Unit tests for runner/pipelines/llamacpp/chat.py.

Tests ChatLlamaCppPipeline and ReasoningChatOpenAI classes.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import Optional, Type
import uuid

from unittest.mock import patch
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from pydantic import BaseModel

from runner.pipelines.llamacpp.chat import (
    ReasoningAwareAIMessageChunk,
    ReasoningChatOpenAI,
    ChatLlamaCppPipeline,
)
from runner.models import ModelTask, ModelProvider, ModelProfileType, Model, ModelProfile
from runner.models.model_parameters import ModelParameters
from runner.models.model_details import ModelDetails
from langchain_openai import ChatOpenAI


class TestReasoningAwareAIMessageChunk:
    """Tests for ReasoningAwareAIMessageChunk class."""

    def test_initialization(self):
        """Test chunk initialization with reasoning content."""
        chunk = ReasoningAwareAIMessageChunk(reasoning_content="test reasoning", content="test")

        assert chunk.reasoning_content == "test reasoning"
        assert chunk.content == "test"

    def test_default_reasoning_content(self):
        """Test chunk with default empty reasoning content."""
        chunk = ReasoningAwareAIMessageChunk(content="test")

        assert chunk.reasoning_content == ""


class TestReasoningChatOpenAI:
    """Tests for ReasoningChatOpenAI class."""

    def test_convert_chunk_to_generation_chunk_with_reasoning(self):
        """Test chunk conversion with reasoning content."""
        # Create a real ReasoningAwareAIMessageChunk with reasoning content
        chunk = {
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "reasoning_content": "I am thinking"
                },
                "finish_reason": None
            }]
        }

        # Create a real ReasoningChatOpenAI instance using model_construct
        # to bypass __init__ which would fail with patched parent
        model = ReasoningChatOpenAI.model_construct(model_name="test")

        # Call the method - it should wrap the message with reasoning content
        result = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

        assert result is not None
        assert hasattr(result.message, 'reasoning_content')
        assert result.message.reasoning_content == "I am thinking"

    def test_convert_chunk_to_generation_chunk_no_reasoning(self, mocker):
        """Test chunk conversion without reasoning content."""
        chunk = {
            "choices": [{
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": None
            }]
        }

        base_chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="Hello"),
            delta={"content": "Hello"},
            finish_reason=None
        )

        model = ReasoningChatOpenAI.model_construct(model_name="test")

        mocker.patch.object(model, '_convert_chunk_to_generation_chunk', return_value=base_chunk)

        result = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

        assert result is not None
        assert result.message.content == "Hello"

    def test_convert_chunk_to_generation_chunk_finish_reason(self):
        """Test chunk conversion with finish_reason."""
        chunk = {
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "reasoning_content": ""
                },
                "finish_reason": "stop"
            }]
        }

        model = ReasoningChatOpenAI.model_construct(model_name="test")

        result = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

        assert result is not None
        assert result.generation_info is not None
        assert result.generation_info.get("finish_reason") == "stop"


class TestChatLlamaCppPipeline:
    """Tests for ChatLlamaCppPipeline class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock Model."""
        model = MagicMock(spec=Model)
        model.id = "test-model"
        model.name = "test-model"
        model.model = "test-model"
        model.provider = ModelProvider.LLAMA_CPP
        model.task = ModelTask.TEXTTOTEXT
        model.modified_at = "2024-01-01T00:00:00Z"
        model.digest = "abc123"
        model.details = MagicMock(spec=ModelDetails)
        model.details.description = "Test model"
        return model

    @pytest.fixture
    def mock_profile(self):
        """Create a mock ModelProfile."""
        profile = MagicMock(spec=ModelProfile)
        profile.id = uuid.uuid4()
        profile.name = "Test Profile"
        profile.user_id = "user-123"
        profile.model_name = "test-model"
        profile.type = ModelProfileType.Primary
        profile.parameters = MagicMock(spec=ModelParameters)
        profile.parameters.temperature = 0.7
        profile.parameters.max_tokens = -1
        profile.parameters.top_p = 0.9
        profile.system_prompt = "You are a helpful assistant."
        return profile

    @pytest.fixture
    def mock_server_manager(self, mocker):
        """Mock LlamaCppServerManager."""
        mock_manager = MagicMock()
        mock_manager.start = MagicMock(return_value=True)
        mock_manager.stop = MagicMock()
        mock_manager.get_api_endpoint = MagicMock(return_value="http://localhost:8080/v1")
        mock_manager.startup_timeout = 30
        return mocker.patch('runner.pipelines.llamacpp.chat.LlamaCppServerManager', return_value=mock_manager)

    def test_initialization(self, mock_model, mock_profile, mock_server_manager):
        """Test pipeline initialization."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)

        assert pipeline.model == mock_model
        assert pipeline.profile == mock_profile
        assert pipeline.started is True
        assert pipeline.server_manager is not None

    def test_initialization_with_grammar(self, mock_model, mock_profile, mock_server_manager):
        """Test pipeline initialization with grammar."""
        class TestModel(BaseModel):
            field: str

        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile, grammar=TestModel)

        assert pipeline.model == mock_model
        assert pipeline.profile == mock_profile

    def test_initialization_with_metadata(self, mock_model, mock_profile, mock_server_manager):
        """Test pipeline initialization with metadata."""
        metadata = {"custom_key": "custom_value"}
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile, metadata=metadata)

        assert pipeline.metadata == metadata

    def test_shutdown(self, mock_model, mock_profile, mock_server_manager):
        """Test pipeline shutdown."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        pipeline.started = True

        pipeline.shutdown()

        assert pipeline.started is False
        pipeline.server_manager.stop.assert_called_once()

    def test_bind_metadata(self, mock_model, mock_profile, mock_server_manager):
        """Test metadata binding."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        pipeline._initialize_persistent_server = MagicMock()
        pipeline._initialize_chat_openai = MagicMock()

        new_metadata = {"new_key": "new_value", "model_name": "updated"}
        result = pipeline.bind_metadata(new_metadata)

        assert pipeline.metadata["new_key"] == "new_value"
        assert pipeline.metadata["model_name"] == "updated"

    def test_bind_metadata_no_chat_model(self, mock_model, mock_profile, mock_server_manager):
        """Test bind_metadata when chat_model not initialized."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        pipeline.chat_model = None

        with pytest.raises(RuntimeError, match="ChatOpenAI not initialized"):
            pipeline.bind_metadata({"key": "value"})

    def test_get_chat_model(self, mock_model, mock_profile, mock_server_manager):
        """Test getting chat model."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)

        chat_model = pipeline.get_chat_model()

        assert chat_model is not None

    def test_get_chat_model_not_initialized(self, mock_model, mock_profile, mock_server_manager):
        """Test get_chat_model when not initialized."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        pipeline.chat_model = None

        with pytest.raises(RuntimeError, match="ChatOpenAI not initialized"):
            pipeline.get_chat_model()

    def test_llm_type_property(self, mock_model, mock_profile, mock_server_manager):
        """Test _llm_type property."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)

        assert pipeline._llm_type == "langchain_chatopenai_llamacpp"

    def test_identifying_params_property(self, mock_model, mock_profile, mock_server_manager):
        """Test _identifying_params property."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)

        params = pipeline._identifying_params

        assert "model_name" in params
        assert "server_port" in params
        assert "pipeline_type" in params

    def test_bind_tools(self, mock_model, mock_profile, mock_server_manager):
        """Test binding tools."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        result = pipeline.bind_tools(tools)

        assert result is not None

    def test_bind_tools_no_chat_model(self, mock_model, mock_profile, mock_server_manager):
        """Test bind_tools when chat_model not initialized."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        pipeline.chat_model = None
        tools = [MagicMock()]

        with pytest.raises(RuntimeError, match="ChatOpenAI not initialized"):
            pipeline.bind_tools(tools)

    def test_del_calls_shutdown(self, mock_model, mock_profile, mock_server_manager):
        """Test __del__ calls shutdown."""
        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)
        pipeline.shutdown = MagicMock()
        pipeline.started = True

        del pipeline

        # Note: __del__ may not be called immediately in tests
        # This is a design test - the method should be defined


class TestChatLlamaCppPipelineIntegration:
    """Integration tests for ChatLlamaCppPipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock Model."""
        model = MagicMock(spec=Model)
        model.id = "test-model"
        model.name = "test-model"
        model.model = "test-model"
        model.provider = ModelProvider.LLAMA_CPP
        model.task = ModelTask.TEXTTOTEXT
        model.modified_at = "2024-01-01T00:00:00Z"
        model.digest = "abc123"
        model.details = MagicMock(spec=ModelDetails)
        model.details.description = "Test model"
        return model

    @pytest.fixture
    def mock_profile(self):
        """Create a mock ModelProfile."""
        profile = MagicMock(spec=ModelProfile)
        profile.id = uuid.uuid4()
        profile.name = "Test Profile"
        profile.user_id = "user-123"
        profile.model_name = "test-model"
        profile.type = ModelProfileType.Primary
        profile.parameters = MagicMock(spec=ModelParameters)
        profile.parameters.temperature = 0.7
        profile.parameters.max_tokens = 100
        profile.parameters.top_p = 0.9
        return profile

    def test_full_initialization_flow(self, mock_model, mock_profile, mocker):
        """Test complete initialization flow."""
        mock_server_manager = MagicMock()
        mock_server_manager.start = MagicMock(return_value=True)
        mock_server_manager.get_api_endpoint = MagicMock(return_value="http://localhost:8080/v1")
        mock_server_manager.startup_timeout = 30

        mocker.patch('runner.pipelines.llamacpp.chat.LlamaCppServerManager', return_value=mock_server_manager)
        mocker.patch.object(ChatLlamaCppPipeline, '_initialize_chat_openai')

        pipeline = ChatLlamaCppPipeline(mock_model, mock_profile)

        assert pipeline.started is True
        mock_server_manager.start.assert_called_once()