import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import logging

# Import the modules to be tested
from pipelines.factory import PipelineFactory
from models.model import Model


class TestPipelineFactory:
    """Test cases for the PipelineFactory class."""

    @patch('pipelines.factory.model_service')
    @patch('pipelines.factory.PipelineFactory.create_pipeline')
    def test_get_pipeline(self, mock_create_pipeline, mock_model_service, sd3_model):
        """Test getting a pipeline for a valid model."""
        # Set up the mock model service
        mock_model_service.models = {
            sd3_model.id: sd3_model
        }
        mock_create_pipeline.return_value = MagicMock()

        # Call the method under test and directly check the model was found
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            pipe = PipelineFactory.get_pipeline(sd3_model.id)

            # We want to verify the model was properly found
            mock_logger.error.assert_not_called()

        # Since we patched at the class level, check if it's been called with any model
        assert mock_create_pipeline.called
        # Get the first call argument
        call_args = mock_create_pipeline.call_args
        if call_args:
            # Compare model IDs since the model objects may not be identical
            assert call_args[0][0].id == sd3_model.id
        # Assert that a pipeline was returned
        assert pipe is not None

    @patch('pipelines.factory.model_service')
    def test_get_pipeline_model_not_found(self, mock_model_service, sd3_model):
        """Test getting a pipeline for a model that doesn't exist."""
        # Set up the mock model service
        mock_model_service.models = {
            sd3_model.id: sd3_model
        }

        # Call the method under test
        pipe = PipelineFactory.get_pipeline("nonexistent_model_id")

        # Assert that no pipeline was returned
        assert pipe is None

    @patch('pipelines.factory.model_service')
    def test_get_pipeline_model_service_not_initialized(self, mock_model_service):
        """Test getting a pipeline when the model service is not initialized."""
        # Set the models attribute to None to simulate an uninitialized service
        mock_model_service.models = None

        # Call the method under test
        pipe = PipelineFactory.get_pipeline("test_model_id")

        # Assert that no pipeline was returned
        assert pipe is None

    @patch('pipelines.sd3.SD3Pipe')
    def test_create_pipeline_sd3(self, mock_sd3_pipe, sd3_model):
        """Test creating an SD3 pipeline."""
        # Set up the mock SD3 pipe
        mock_sd3_instance = MagicMock()
        mock_sd3_pipe.load.return_value = mock_sd3_instance

        # Call the method under test
        pipe = PipelineFactory.create_pipeline(sd3_model)

        # Assert that the SD3 pipe was loaded with the test model
        mock_sd3_pipe.load.assert_called_once_with(sd3_model)
        # Assert that the returned pipeline is the mock SD3 instance
        assert pipe == mock_sd3_instance

    @patch('pipelines.sdxl.SDXLPipe')
    def test_create_pipeline_sdxl(self, mock_sdxl_pipe, sdxl_model):
        """Test creating an SDXL pipeline."""
        # Set up the mock SDXL pipe
        mock_sdxl_instance = MagicMock()
        mock_sdxl_pipe.load.return_value = mock_sdxl_instance

        # Call the method under test
        pipe = PipelineFactory.create_pipeline(sdxl_model)

        # Assert that the SDXL pipe was loaded with the test model
        mock_sdxl_pipe.load.assert_called_once_with(sdxl_model)
        # Assert that the returned pipeline is the mock SDXL instance
        assert pipe == mock_sdxl_instance

    @patch('pipelines.flux.FluxPipe')
    def test_create_pipeline_flux(self, mock_flux_pipe, flux_model):
        """Test creating a Flux pipeline."""
        # Set up the mock Flux pipe
        mock_flux_instance = MagicMock()
        mock_flux_pipe.load.return_value = mock_flux_instance

        # Call the method under test
        pipe = PipelineFactory.create_pipeline(flux_model)

        # Assert that the Flux pipe was loaded with the test model
        mock_flux_pipe.load.assert_called_once_with(flux_model)
        # Assert that the returned pipeline is the mock Flux instance
        assert pipe == mock_flux_instance

    def test_create_pipeline_unsupported(self, test_models):
        """Test creating a pipeline with an unsupported type."""
        # Create a test model with an unsupported pipeline type
        sd3_model = next(model for _, model in test_models.items()
                         if model.pipeline == "StableDiffusion3Pipeline")

        # Copy the model and change the pipeline type
        from copy import deepcopy
        unsupported_model = deepcopy(sd3_model)
        unsupported_model.pipeline = "UnsupportedPipeline"

        # Call the method under test
        pipe = PipelineFactory.create_pipeline(unsupported_model)

        # Assert that no pipeline was returned
        assert pipe is None

    @patch('pipelines.factory.PipelineFactory.load_lora_weights')
    def test_load_lora_weights_called(self, mock_load_lora_weights, model_with_lora):
        """Test that load_lora_weights is called when a pipeline is created."""
        # Determine which pipeline class to mock based on the model's pipeline type
        pipeline_type = model_with_lora.pipeline

        # Create a mock pipeline instance
        mock_pipe_instance = MagicMock()

        # Patch the appropriate pipeline class
        if pipeline_type == "StableDiffusion3Pipeline":
            with patch('pipelines.sd3.SD3Pipe') as mock_pipe_class:
                mock_pipe_class.load.return_value = mock_pipe_instance
                # Call the method under test
                pipe = PipelineFactory.create_pipeline(model_with_lora)
        elif pipeline_type == "StableDiffusionXLPipeline":
            with patch('pipelines.sdxl.SDXLPipe') as mock_pipe_class:
                mock_pipe_class.load.return_value = mock_pipe_instance
                # Call the method under test
                pipe = PipelineFactory.create_pipeline(model_with_lora)
        elif pipeline_type == "FluxPipeline":
            with patch('pipelines.flux.FluxPipe') as mock_pipe_class:
                # Also mock the _setup_quantization_config to avoid bitsandbytes dependency
                mock_pipe_class._setup_quantization_config = MagicMock(return_value=MagicMock())
                mock_pipe_class.load.return_value = mock_pipe_instance
                # Call the method under test
                pipe = PipelineFactory.create_pipeline(model_with_lora)
        else:
            pytest.fail(f"Unsupported pipeline type: {pipeline_type}")

        # Assert that load_lora_weights was called with the pipe and model
        mock_load_lora_weights.assert_called_once_with(mock_pipe_instance, model_with_lora)

    def test_load_lora_weights(self, model_with_lora):
        """Test loading LoRA weights into a pipeline."""
        # Create a mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.load_lora_weights = MagicMock()

        # Call the method under test
        PipelineFactory.load_lora_weights(mock_pipeline, model_with_lora)

        # Assert that the pipeline's load_lora_weights method was called with the correct arguments
        lora_weight = model_with_lora.lora_weights[0]
        mock_pipeline.load_lora_weights.assert_called_once_with(
            lora_weight.name,
            weight_name=lora_weight.weight_name,
            adapter_name=lora_weight.adapter_name
        )

    def test_load_lora_weights_no_method(self, model_with_lora):
        """Test loading LoRA weights into a pipeline that doesn't support it."""
        # Create a mock pipeline without the load_lora_weights method
        mock_pipeline = MagicMock()
        delattr(mock_pipeline, 'load_lora_weights')

        # Call the method under test - should not raise an exception
        PipelineFactory.load_lora_weights(mock_pipeline, model_with_lora)

        # No assertions needed - we're just checking it doesn't raise an exception

    def test_load_lora_weights_no_weights(self, sd3_model):
        """Test loading LoRA weights when there are none."""
        # Create a mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.load_lora_weights = MagicMock()

        # Call the method under test
        PipelineFactory.load_lora_weights(mock_pipeline, sd3_model)

        # Assert that the pipeline's load_lora_weights method was not called
        mock_pipeline.load_lora_weights.assert_not_called()
