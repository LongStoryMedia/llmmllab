import pytest
import torch

from pipelines.helpers import get_dtype, get_precision
from models.model import Model
from models.model_details import ModelDetails


class TestHelpers:
    """Test cases for the helper functions."""

    def test_get_dtype_from_models_json(self, test_models):
        """Test get_dtype with models from models.json."""
        for model_id, model in test_models.items():
            # Call the function under test
            dtype = get_dtype(model)

            # Assert dtype is a torch.dtype
            assert isinstance(dtype, torch.dtype)

            # Check mapping is correct based on model's dtype
            if model.details and model.details.dtype:
                if model.details.dtype.lower() in ["float16", "fp16"]:
                    assert dtype == torch.float16
                elif model.details.dtype.lower() in ["bfloat16", "bfp16"]:
                    assert dtype == torch.bfloat16
                elif model.details.dtype.lower() in ["float32", "fp32"]:
                    assert dtype == torch.float32

    def test_get_precision_from_models_json(self, test_models):
        """Test get_precision with models from models.json."""
        for model_id, model in test_models.items():
            # Call the function under test
            precision = get_precision(model)

            # Check mapping is correct based on model's dtype
            if model.details and model.details.dtype:
                if model.details.dtype.lower() in ["float16", "fp16"]:
                    assert precision == "fp16"
                elif model.details.dtype.lower() in ["bfloat16", "bfp16"]:
                    assert precision == "bfp16"
                elif model.details.dtype.lower() in ["float32", "fp32"]:
                    assert precision == "fp32"
                else:
                    assert precision is None

    @pytest.mark.parametrize("dtype_str,expected", [
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bfp16", torch.bfloat16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("unsupported", torch.float32),  # Default case
    ])
    def test_get_dtype_parametrized(self, dtype_str, expected):
        """Test get_dtype with parametrized inputs."""
        # Create model details with the given dtype
        model_details = ModelDetails(
            parent_model="test_parent_model",
            format="test_format",
            family="test_family",
            families=["test_family"],
            parameter_size="1B",
            dtype=dtype_str,
            quantization_level=None,
            specialization="TextToImage"
        )

        # Create a model with the details
        model = Model(
            id="test_model_id",
            name="test_model",
            model="test_model_path",
            modified_at="2025-06-30",
            size=1000,
            digest="test_digest",
            pipeline="StableDiffusion3Pipeline",
            details=model_details,
            lora_weights=[]
        )

        # Call the function under test
        dtype = get_dtype(model)

        # Assert that the correct dtype is returned
        assert dtype == expected

    @pytest.mark.parametrize("dtype_str,expected", [
        ("float16", "fp16"),
        ("fp16", "fp16"),
        ("bfloat16", "bfp16"),
        ("bfp16", "bfp16"),
        ("float32", "fp32"),
        ("fp32", "fp32"),
        ("unsupported", None),  # Default case
    ])
    def test_get_precision_parametrized(self, dtype_str, expected):
        """Test get_precision with parametrized inputs."""
        # Create model details with the given dtype
        model_details = ModelDetails(
            parent_model="test_parent_model",
            format="test_format",
            family="test_family",
            families=["test_family"],
            parameter_size="1B",
            dtype=dtype_str,
            quantization_level=None,
            specialization="TextToImage"
        )

        # Create a model with the details
        model = Model(
            id="test_model_id",
            name="test_model",
            model="test_model_path",
            modified_at="2025-06-30",
            size=1000,
            digest="test_digest",
            pipeline="StableDiffusion3Pipeline",
            details=model_details,
            lora_weights=[]
        )

        # Call the function under test
        precision = get_precision(model)

        # Assert that the correct precision is returned
        assert precision == expected

    def test_get_dtype_no_dtype(self):
        """Test get_dtype with no dtype in details."""
        # Create model details with no dtype
        model_details = ModelDetails(
            parent_model="test_parent_model",
            format="test_format",
            family="test_family",
            families=["test_family"],
            parameter_size="1B",
            dtype=None,
            quantization_level=None,
            specialization="TextToImage"
        )

        # Create a model with the details
        model = Model(
            id="test_model_id",
            name="test_model",
            model="test_model_path",
            modified_at="2025-06-30",
            size=1000,
            digest="test_digest",
            pipeline="StableDiffusion3Pipeline",
            details=model_details,
            lora_weights=[]
        )

        # Call the function under test
        dtype = get_dtype(model)

        # Assert that the default dtype is returned
        assert dtype == torch.float32

    def test_get_precision_no_dtype(self):
        """Test get_precision with no dtype in details."""
        # Create model details with no dtype
        model_details = ModelDetails(
            parent_model="test_parent_model",
            format="test_format",
            family="test_family",
            families=["test_family"],
            parameter_size="1B",
            dtype=None,
            quantization_level=None,
            specialization="TextToImage"
        )

        # Create a model with the details
        model = Model(
            id="test_model_id",
            name="test_model",
            model="test_model_path",
            modified_at="2025-06-30",
            size=1000,
            digest="test_digest",
            pipeline="StableDiffusion3Pipeline",
            details=model_details,
            lora_weights=[]
        )

        # Call the function under test
        precision = get_precision(model)

        # Assert that None is returned
        assert precision is None
