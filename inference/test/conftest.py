import os
import sys
import pytest
from unittest.mock import MagicMock
from typing import Dict

import torch
from models.model import Model
from models.model_details import ModelDetails
from models.lora_weight import LoraWeight
from test.utils import load_test_models

# Add the parent directory to the sys.path to make imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def test_models() -> Dict[str, Model]:
    """Fixture for loading models from models.json for testing."""
    return load_test_models()


@pytest.fixture
def sd3_model(test_models) -> Model:
    """Fixture for a Stable Diffusion 3 model."""
    # Find a model with StableDiffusion3Pipeline pipeline
    for model_id, model in test_models.items():
        if model.pipeline == "StableDiffusion3Pipeline":
            return model

    # If no SD3 model is found, create a basic one
    details = ModelDetails(
        parent_model="test_parent_model",
        format="test_format",
        family="test_family",
        families=["test_family"],
        parameter_size="8B",
        dtype="BFP16",
        quantization_level="nf4",
        specialization="TextToImage"
    )

    return Model(
        id="test-sd3-model",
        name="test-sd3-model",
        model="stabilityai/stable-diffusion-3-medium",
        modified_at="2025-06-30",
        size=1000,
        digest="test_digest",
        pipeline="StableDiffusion3Pipeline",
        details=details,
        lora_weights=[],
        task="TextToImage",
    )


@pytest.fixture
def sdxl_model(test_models) -> Model:
    """Fixture for a Stable Diffusion XL model."""
    # Find a model with StableDiffusionXLPipeline pipeline
    for model_id, model in test_models.items():
        if model.pipeline == "StableDiffusionXLPipeline":
            return model

    # If no SDXL model is found, create a basic one
    details = ModelDetails(
        parent_model="test_parent_model",
        format="test_format",
        family="test_family",
        families=["test_family"],
        parameter_size="2.5B",
        dtype="FP16",
        specialization="TextToImage"
    )

    return Model(
        id="test-sdxl-model",
        name="test-sdxl-model",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        modified_at="2025-06-30",
        size=1000,
        digest="test_digest",
        pipeline="StableDiffusionXLPipeline",
        details=details,
        lora_weights=[],
        task="TextToImage",
    )


@pytest.fixture
def flux_model(test_models) -> Model:
    """Fixture for a Flux model."""
    # Find a model with FluxPipeline pipeline
    for model_id, model in test_models.items():
        if model.pipeline == "FluxPipeline":
            return model

    # If no Flux model is found, create a basic one
    details = ModelDetails(
        parent_model="test_parent_model",
        format="test_format",
        family="test_family",
        families=["test_family"],
        parameter_size="12B",
        dtype="BFP16",
        quantization_level="nf4",
        specialization="TextToImage"
    )

    return Model(
        id="test-flux-model",
        name="test-flux-model",
        model="black-forest-labs/FLUX.1-dev",
        modified_at="2025-06-30",
        size=1000,
        digest="test_digest",
        pipeline="FluxPipeline",
        details=details,
        lora_weights=[],
        task="TextToImage",
    )


@pytest.fixture
def model_with_lora(test_models) -> Model:
    """Fixture for a model with LoRA weights."""
    # Find a model with LoRA weights
    for model_id, model in test_models.items():
        if model.lora_weights and len(model.lora_weights) > 0:
            return model

    # If no model with LoRA weights is found, create one
    details = ModelDetails(
        parent_model="test_parent_model",
        format="test_format",
        family="test_family",
        families=["test_family"],
        parameter_size="12B",
        dtype="BFP16",
        quantization_level="nf4",
        specialization="TextToImage"
    )

    lora_weight = LoraWeight(
        id="test_lora_id",
        name="test_lora_weight",
        weight_name="lora.safetensors",
        adapter_name="uncensored",
        parent_model="test_model_path"
    )

    return Model(
        id="test-model-with-lora",
        name="test-model-with-lora",
        model="test-model-with-lora",
        modified_at="2025-06-30",
        size=1000,
        digest="test_digest",
        pipeline="StableDiffusion3Pipeline",
        details=details,
        lora_weights=[lora_weight],
        task="TextToImage",
    )


@pytest.fixture
def mock_pipeline():
    """Fixture for creating a mock pipeline."""
    mock = MagicMock()
    mock.load_lora_weights = MagicMock()
    return mock
