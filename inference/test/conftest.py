import os
import sys
import pytest
from unittest.mock import MagicMock
from typing import Dict

from models.model import Model
from models.model_details import ModelDetails
from models.lora_weight import LoraWeight

# Ensure parent directory is in path before local imports (idempotent)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import logging

def load_test_models() -> Dict[str, Model]:
    """Inline loader for models.json to avoid import path issues in tests."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(project_root, ".models.json")
    models: Dict[str, Model] = {}
    try:
        if not os.path.exists(models_file):
            logging.error(f"Models config file not found: {models_file}")
            return {}
        with open(models_file, "r", encoding="utf-8") as f:
            models_data = json.load(f)
        for model_data in models_data:
            lora_weights = model_data.get("lora_weights", []) or []
            loras = []
            for lora in lora_weights:
                if lora:
                    loras.append(
                        LoraWeight(
                            id=lora.get("id", ""),
                            name=lora.get("name", ""),
                            weight_name=lora.get("weight_name", ""),
                            adapter_name=lora.get("adapter_name", ""),
                            parent_model=lora.get("parent_model", ""),
                        )
                    )
            details_dict = model_data.get("details", {}) or {}
            details = ModelDetails(
                parent_model=details_dict.get("parent_model", ""),
                format=details_dict.get("format", ""),
                family=details_dict.get("family", ""),
                families=details_dict.get("families", []),
                parameter_size=details_dict.get("parameter_size", ""),
                dtype=details_dict.get("dtype", "float32"),
                quantization_level=details_dict.get("quantization_level", ""),
                specialization=details_dict.get("specialization", ""),
            )
            model = Model(
                id=model_data.get("id"),
                name=model_data["name"],
                model=model_data.get("model"),
                modified_at=model_data.get("modified_at", ""),
                size=model_data.get("size", 0),
                digest=model_data.get("digest", ""),
                pipeline=model_data.get("pipeline"),
                lora_weights=loras,
                details=details,
            )
            models[model_data["id"]] = model
        logging.info(f"Loaded {len(models)} models from config for testing")
        return models
    except Exception as e:
        logging.error(f"Error loading models from config: {e}")
        return {}


@pytest.fixture
def test_models() -> Dict[str, Model]:
    """Fixture for loading models from models.json for testing."""
    return load_test_models()


@pytest.fixture
def sd3_model() -> Model:
    """Fixture for a Stable Diffusion 3 model."""
    # Find a model with StableDiffusion3Pipeline pipeline
    models_fixture = load_test_models()
    for _model_id, model in models_fixture.items():
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
def sdxl_model() -> Model:
    """Fixture for a Stable Diffusion XL model."""
    # Find a model with StableDiffusionXLPipeline pipeline
    models_fixture = load_test_models()
    for _model_id, model in models_fixture.items():
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
def flux_model() -> Model:
    """Fixture for a Flux model."""
    # Find a model with FluxPipeline pipeline
    models_fixture = load_test_models()
    for _model_id, model in models_fixture.items():
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
def model_with_lora() -> Model:
    """Fixture for a model with LoRA weights."""
    # Find a model with LoRA weights
    models_fixture = load_test_models()
    for _model_id, model in models_fixture.items():
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
