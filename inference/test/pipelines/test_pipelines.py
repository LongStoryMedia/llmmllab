import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import logging

# Import the modules to be tested
from pipelines.txt2img.sd3 import SD3Pipe
from pipelines.txt2img.sdxl import SDXLPipe
from inference.pipelines.txt2img.flux import FluxPipe
from models.model import Model
from diffusers.quantizers.quantization_config import BitsAndBytesConfig


class TestSD3Pipe:
    """Test cases for the SD3Pipe class."""

    @patch('pipelines.sd3.StableDiffusion3Pipeline')
    @patch('pipelines.sd3.SD3Transformer2DModel')
    @patch('pipelines.sd3.get_dtype')
    def test_load_sd3_pipeline(self, mock_get_dtype, mock_transformer, mock_sd3_pipeline, sd3_model):
        """Test loading an SD3 pipeline."""
        # Set up mocks
        mock_get_dtype.return_value = torch.bfloat16
        mock_transformer_instance = MagicMock()
        mock_transformer.from_pretrained.return_value = mock_transformer_instance
        mock_pipeline_instance = MagicMock()
        mock_sd3_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Mock the quantization config
        bnb_config = MagicMock(spec=BitsAndBytesConfig)

        # Call the method under test with patch to avoid actual loading
        with patch('pipelines.sd3.SD3Pipe._setup_quantization_config', return_value=bnb_config):
            pipe = SD3Pipe.load(sd3_model)

        # Assert transformer was loaded with the correct arguments
        mock_transformer.from_pretrained.assert_called()
        kwargs = mock_transformer.from_pretrained.call_args.kwargs
        assert kwargs['subfolder'] == 'transformer'
        assert kwargs['torch_dtype'] == torch.bfloat16
        # Check that quantization_config was passed, but not check its exact type
        # since we're mocking and avoiding actual BitsAndBytesConfig instantiation
        assert 'quantization_config' in kwargs

        # Assert pipeline was loaded with the correct arguments
        mock_sd3_pipeline.from_pretrained.assert_called_with(
            sd3_model.model,
            transformer=mock_transformer_instance,
            torch_dtype=torch.bfloat16
        )

        # Assert memory optimization methods were called
        mock_pipeline_instance.enable_model_cpu_offload.assert_called_once()

        # Assert the correct pipeline is returned
        assert pipe == mock_pipeline_instance

    @patch('pipelines.sd3.StableDiffusion3Pipeline')
    @patch('pipelines.sd3.SD3Transformer2DModel')
    @patch('pipelines.sd3.get_dtype')
    def test_load_sd3_pipeline_no_quantization(self, mock_get_dtype, mock_transformer, mock_sd3_pipeline, sd3_model):
        """Test loading an SD3 pipeline without quantization."""
        # Create a copy of the model to avoid side effects on other tests
        from copy import deepcopy
        sd3_model_copy = deepcopy(sd3_model)
        sd3_model_copy.details.quantization_level = None

        # Set up mocks
        mock_get_dtype.return_value = torch.float32
        mock_transformer_instance = MagicMock()
        mock_transformer.from_pretrained.return_value = mock_transformer_instance
        mock_pipeline_instance = MagicMock()
        mock_sd3_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Call the method under test with patch to avoid actual loading
        with patch('pipelines.sd3.SD3Pipe._setup_quantization_config', return_value=None):
            # Also patch get_dtype to ensure it returns what we expect
            pipe = SD3Pipe.load(sd3_model_copy)

        # Assert transformer was loaded with the correct arguments
        mock_transformer.from_pretrained.assert_called()
        kwargs = mock_transformer.from_pretrained.call_args.kwargs
        assert kwargs['subfolder'] == 'transformer'
        # We've mocked get_dtype to return float32 so this should match
        assert kwargs['torch_dtype'] == torch.float32

        # Assert pipeline was loaded
        mock_sd3_pipeline.from_pretrained.assert_called()

        # Assert the correct pipeline is returned
        assert pipe == mock_pipeline_instance


class TestSDXLPipe:
    """Test cases for the SDXLPipe class."""

    @patch('pipelines.sdxl.StableDiffusionXLPipeline')
    @patch('pipelines.sdxl.get_dtype')
    @patch('pipelines.sdxl.get_precision')
    def test_load_sdxl_pipeline(self, mock_get_precision, mock_get_dtype, mock_sdxl_pipeline, sdxl_model):
        """Test loading an SDXL pipeline."""
        # Set up mocks
        mock_get_dtype.return_value = torch.float16
        mock_get_precision.return_value = "fp16"
        mock_pipeline_instance = MagicMock()
        mock_sdxl_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Call the method under test
        pipe = SDXLPipe.load(sdxl_model)

        # Assert pipeline was loaded with the correct arguments
        mock_sdxl_pipeline.from_pretrained.assert_called_with(
            sdxl_model.model,
            device_map="balanced",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            safety_checker=None
        )

        # Assert the correct pipeline is returned
        assert pipe == mock_pipeline_instance


class TestFluxPipe:
    """Test cases for the FluxPipe class."""

    @patch('pipelines.flux.FluxPipeline')
    @patch('pipelines.flux.FluxTransformer2DModel')
    def test_load_flux_pipeline(self, mock_transformer, mock_flux_pipeline, flux_model):
        """Test loading a Flux pipeline."""
        # Set up mocks
        mock_transformer_instance = MagicMock()
        mock_transformer.from_pretrained.return_value = mock_transformer_instance
        mock_pipeline_instance = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Mock the quantization config
        bnb_config = MagicMock(spec=BitsAndBytesConfig)

        # Call the method under test with patch to avoid actual loading
        with patch('pipelines.flux.FluxPipe._setup_quantization_config', return_value=bnb_config):
            pipe = FluxPipe.load(flux_model)

        # Assert transformer was loaded with the correct arguments
        mock_transformer.from_pretrained.assert_called()
        kwargs = mock_transformer.from_pretrained.call_args.kwargs
        assert kwargs['subfolder'] == 'transformer'
        assert kwargs['torch_dtype'] == torch.bfloat16
        # Check that quantization_config was passed, but not check its exact type
        # since we're mocking and avoiding actual BitsAndBytesConfig instantiation
        assert 'quantization_config' in kwargs

        # Assert pipeline was loaded with the correct arguments
        mock_flux_pipeline.from_pretrained.assert_called_with(
            flux_model.name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            transformer=mock_transformer_instance
        )

        # Assert memory optimization methods were called
        mock_pipeline_instance.enable_model_cpu_offload.assert_called_once()

        # Assert the correct pipeline is returned
        assert pipe == mock_pipeline_instance
