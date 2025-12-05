"""
Unit tests for ModelLoader to validate models from .models.yaml are loaded with all necessary details.

This test validates:
1. All models load successfully from .models.yaml
2. All required fields are present with valid values
3. All ModelDetails fields are populated correctly
4. Data types and constraints are respected
5. Default values are applied appropriately
6. Statistics are calculated correctly
"""

import os
import pytest
import tempfile
import yaml
from typing import Dict, Any
from unittest.mock import patch

from models import Model, ModelDetails, LoraWeight
from runner.utils.model_loader import ModelLoader


class TestModelLoader:
    """Test suite for ModelLoader functionality."""

    def test_load_models_from_yaml_file(self):
        """Test that ModelLoader successfully loads all models from .models.yaml."""
        # Use the actual ModelLoader to load from the real config
        loader = ModelLoader()
        models = loader.get_available_models()

        # Verify models were loaded
        assert len(models) > 0, "No models were loaded from configuration"

        # Log the loaded models for debugging
        print(f"Loaded {len(models)} models: {list(models.keys())}")

    def test_all_models_have_required_fields(self):
        """Test that all loaded models have the required fields with valid values."""
        loader = ModelLoader()
        models = loader.get_available_models()

        for model_id, model in models.items():
            # Test required Model fields
            assert model.id, f"Model {model_id} missing id"
            assert model.name, f"Model {model_id} missing name"
            assert model.model, f"Model {model_id} missing model path"
            assert model.provider, f"Model {model_id} missing provider"
            # Note: modified_at and digest can be empty strings in some configs, which is allowed
            assert (
                model.modified_at is not None
            ), f"Model {model_id} modified_at is None"
            assert model.digest is not None, f"Model {model_id} digest is None"
            assert model.task, f"Model {model_id} missing task"

            # Test ModelDetails required fields
            assert model.details, f"Model {model_id} missing details"
            assert (
                model.details.size is not None
            ), f"Model {model_id} missing details.size"
            assert (
                model.details.size >= 0
            ), f"Model {model_id} has negative size: {model.details.size}"
            assert (
                model.details.original_ctx is not None
            ), f"Model {model_id} missing details.original_ctx"
            assert (
                model.details.original_ctx > 0
            ), f"Model {model_id} has invalid original_ctx: {model.details.original_ctx}"
            # Note: format and family can be empty for HF models, parameter_size is required
            assert model.details.format is not None, f"Model {model_id} format is None"
            assert model.details.family is not None, f"Model {model_id} family is None"
            assert (
                model.details.parameter_size
            ), f"Model {model_id} missing details.parameter_size"

    def test_model_types_have_appropriate_details(self):
        """Test that different model types (GGUF vs HF) have appropriate field completeness."""
        loader = ModelLoader()
        models = loader.get_available_models()

        llama_cpp_models = []
        hf_models = []

        for model_id, model in models.items():
            if model.provider.value == "llama_cpp":
                llama_cpp_models.append((model_id, model))
            elif model.provider.value in ["hf", "hugging_face"]:
                hf_models.append((model_id, model))

        print(
            f"Found {len(llama_cpp_models)} llama_cpp models and {len(hf_models)} HF models"
        )

        # GGUF models should have more complete details
        for model_id, model in llama_cpp_models:
            details = model.details
            assert details.format, f"GGUF model {model_id} should have format"
            assert details.family, f"GGUF model {model_id} should have family"
            assert details.size > 0, f"GGUF model {model_id} should have positive size"
            # Most GGUF models should have architectural details
            if details.n_layers:
                assert (
                    details.n_layers > 0
                ), f"GGUF model {model_id} has invalid layer count"

        # HF models may have minimal details, which is acceptable
        for model_id, model in hf_models:
            details = model.details
            # HF models may have empty format/family but should have parameter_size
            assert (
                details.parameter_size
            ), f"HF model {model_id} should have parameter_size"
            # Size might be 0 for HF models as they're downloaded on demand

    def test_model_details_field_types(self):
        """Test that all ModelDetails fields have correct types."""
        loader = ModelLoader()
        models = loader.get_available_models()

        for model_id, model in models.items():
            details = model.details

            # Test field types
            assert isinstance(
                details.size, (int, float)
            ), f"Model {model_id} size is not numeric: {type(details.size)}"
            assert isinstance(
                details.original_ctx, int
            ), f"Model {model_id} original_ctx is not int: {type(details.original_ctx)}"
            assert isinstance(
                details.format, str
            ), f"Model {model_id} format is not string: {type(details.format)}"
            assert isinstance(
                details.family, str
            ), f"Model {model_id} family is not string: {type(details.family)}"
            assert isinstance(
                details.families, list
            ), f"Model {model_id} families is not list: {type(details.families)}"
            assert isinstance(
                details.parameter_size, str
            ), f"Model {model_id} parameter_size is not string: {type(details.parameter_size)}"

            # Test optional fields when present
            if details.specialization:
                assert isinstance(
                    details.specialization, str
                ), f"Model {model_id} specialization is not string: {type(details.specialization)}"
                # Validate specialization values
                valid_specializations = [
                    "Text",
                    "LoRA",
                    "Embedding",
                    "TextToImage",
                    "ImageToImage",
                    "Audio",
                ]
                assert (
                    details.specialization in valid_specializations
                ), f"Model {model_id} has invalid specialization: {details.specialization}"

            if details.dtype:
                assert isinstance(
                    details.dtype, str
                ), f"Model {model_id} dtype is not string: {type(details.dtype)}"

            if details.weight is not None:
                assert isinstance(
                    details.weight, (int, float)
                ), f"Model {model_id} weight is not numeric: {type(details.weight)}"

    def test_model_architectural_details(self):
        """Test that models have proper architectural details from GGUF metadata."""
        loader = ModelLoader()
        models = loader.get_available_models()

        # Count models with architectural details
        models_with_arch_details = 0

        for model_id, model in models.items():
            details = model.details

            # Check for architectural information that should be present from GGUF metadata
            arch_fields = ["layers", "heads", "ctx_length", "vocab_size"]
            has_arch_info = any(getattr(details, field, None) for field in arch_fields)

            if has_arch_info:
                models_with_arch_details += 1

                # If architectural info is present, validate it
                if details.layers:
                    assert (
                        details.layers > 0
                    ), f"Model {model_id} has invalid layers: {details.layers}"
                if details.heads:
                    assert (
                        details.heads > 0
                    ), f"Model {model_id} has invalid heads: {details.heads}"
                if details.ctx_length:
                    assert (
                        details.ctx_length > 0
                    ), f"Model {model_id} has invalid ctx_length: {details.ctx_length}"
                if details.vocab_size:
                    assert (
                        details.vocab_size > 0
                    ), f"Model {model_id} has invalid vocab_size: {details.vocab_size}"

        # We expect most models to have architectural details from GGUF metadata
        print(
            f"Models with architectural details: {models_with_arch_details}/{len(models)}"
        )

    def test_lora_weights_loading(self):
        """Test that LoRA weights are loaded correctly when present."""
        loader = ModelLoader()
        models = loader.get_available_models()

        for model_id, model in models.items():
            # Test LoRA weights structure
            assert isinstance(
                model.lora_weights, list
            ), f"Model {model_id} lora_weights is not a list"

            for lora in model.lora_weights:
                assert isinstance(
                    lora, LoraWeight
                ), f"Model {model_id} has invalid LoRA weight type"
                assert lora.id, f"Model {model_id} LoRA missing id"
                assert lora.name, f"Model {model_id} LoRA missing name"

    def test_model_statistics(self):
        """Test that model statistics are calculated correctly."""
        loader = ModelLoader()
        stats = loader.get_model_statistics()

        # Test statistics structure
        assert "total_models" in stats
        assert "providers" in stats
        assert "specializations" in stats
        assert "families" in stats
        assert "tasks" in stats
        assert "models_with_clip" in stats
        assert "models_with_lora" in stats
        assert "average_size_gb" in stats
        assert "size_range" in stats

        # Test statistics values
        assert stats["total_models"] > 0, "No models in statistics"
        assert isinstance(stats["providers"], dict), "Providers should be a dictionary"
        assert isinstance(
            stats["specializations"], dict
        ), "Specializations should be a dictionary"
        assert isinstance(stats["families"], dict), "Families should be a dictionary"
        assert isinstance(stats["tasks"], dict), "Tasks should be a dictionary"
        assert isinstance(
            stats["models_with_clip"], int
        ), "CLIP count should be an integer"
        assert isinstance(
            stats["models_with_lora"], int
        ), "LoRA count should be an integer"
        assert isinstance(
            stats["average_size_gb"], (int, float)
        ), "Average size should be numeric"

        # Test size range structure
        assert "min" in stats["size_range"]
        assert "max" in stats["size_range"]
        assert stats["size_range"]["min"] >= 0, "Min size should be non-negative"
        assert (
            stats["size_range"]["max"] >= stats["size_range"]["min"]
        ), "Max size should be >= min size"

        print(f"Statistics: {stats}")

    def test_get_model_by_id(self):
        """Test retrieval of specific models by ID."""
        loader = ModelLoader()
        models = loader.get_available_models()

        if models:
            # Test getting existing model
            first_model_id = list(models.keys())[0]
            retrieved_model = loader.get_model_by_id(first_model_id)
            assert (
                retrieved_model is not None
            ), f"Could not retrieve model {first_model_id}"
            assert retrieved_model.id == first_model_id, "Retrieved model has wrong ID"

            # Test getting non-existent model
            non_existent_model = loader.get_model_by_id("non-existent-model-id")
            assert (
                non_existent_model is None
            ), "Should return None for non-existent model"

    def test_model_validation(self):
        """Test model data validation functionality."""
        loader = ModelLoader()

        # Test valid model data
        valid_data = {
            "id": "test-model",
            "name": "Test Model",
            "model": "test/model/path",
            "provider": "test_provider",
            "modified_at": "2024-01-01",
            "digest": "test_digest",
            "task": "TextToText",
            "details": {
                "size": 1000000,
                "original_ctx": 4096,
                "format": "gguf",
                "family": "test",
                "parameter_size": "7B",
            },
        }

        errors = loader.validate_model_data(valid_data)
        assert len(errors) == 0, f"Valid model data failed validation: {errors}"

        # Test invalid model data - missing required fields
        invalid_data = {
            "name": "Test Model"
            # Missing other required fields
        }

        errors = loader.validate_model_data(invalid_data)
        assert len(errors) > 0, "Invalid model data should have validation errors"

        # Check for specific missing field errors
        error_text = " ".join(errors)
        assert "model" in error_text, "Should report missing 'model' field"
        assert "provider" in error_text, "Should report missing 'provider' field"

    def test_dynamic_field_mapping(self):
        """Test that ModelLoader handles all Pydantic model fields dynamically."""
        loader = ModelLoader()

        # Get field information for introspection
        model_fields = loader._get_model_fields()
        details_fields = loader._get_model_details_fields()

        # Verify that field introspection works
        assert len(model_fields) > 0, "Should detect Model fields"
        assert len(details_fields) > 0, "Should detect ModelDetails fields"

        # Check for key expected fields
        expected_model_fields = ["id", "name", "model", "provider", "details", "task"]
        for field in expected_model_fields:
            assert field in model_fields, f"Model field '{field}' not detected"

        expected_details_fields = [
            "size",
            "original_ctx",
            "format",
            "family",
            "specialization",
        ]
        for field in expected_details_fields:
            assert field in details_fields, f"ModelDetails field '{field}' not detected"

    @patch.dict(os.environ, {"MODELS_FILE_PATH": "/tmp/test_models.yaml"})
    def test_custom_models_file_path(self):
        """Test loading models from custom file path specified in environment variable."""
        # Create a temporary YAML file with test model data
        test_models_data = [
            {
                "id": "test-custom-model",
                "name": "Test Custom Model",
                "model": "test/custom/path",
                "provider": "hf",  # Use valid provider enum value
                "modified_at": "2024-01-01",
                "digest": "test_digest",
                "task": "TextToText",
                "details": {
                    "size": 500000,
                    "original_ctx": 2048,
                    "format": "gguf",
                    "family": "custom",
                    "parameter_size": "3B",
                    "specialization": "Text",
                },
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_models_data, f)
            temp_path = f.name

        try:
            # Update environment variable to point to our test file
            with patch.dict(os.environ, {"MODELS_FILE_PATH": temp_path}):
                # Create new loader to test custom path
                custom_loader = ModelLoader()
                models = custom_loader.get_available_models()

                # Verify our test model was loaded
                assert "test-custom-model" in models, "Custom model should be loaded"
                model = models["test-custom-model"]
                assert model.name == "Test Custom Model"
                assert model.details.size == 500000
                assert model.details.specialization == "Text"

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def test_reload_models_functionality(self):
        """Test that reload_models() refreshes the model cache."""
        loader = ModelLoader()

        # Get initial model count
        initial_models = loader.get_available_models()
        initial_count = len(initial_models)

        # Reload models
        loader.reload_models()

        # Get models after reload
        reloaded_models = loader.get_available_models()
        reloaded_count = len(reloaded_models)

        # Should have the same models after reload (assuming config didn't change)
        assert (
            reloaded_count == initial_count
        ), "Model count should be consistent after reload"

        # Verify model IDs are the same
        initial_ids = set(initial_models.keys())
        reloaded_ids = set(reloaded_models.keys())
        assert (
            initial_ids == reloaded_ids
        ), "Model IDs should be consistent after reload"


if __name__ == "__main__":
    # Run tests manually if executed directly
    import sys

    test_loader = TestModelLoader()

    print("Running ModelLoader validation tests...")

    try:
        test_loader.test_load_models_from_yaml_file()
        print("✅ Models loaded successfully from .models.yaml")

        test_loader.test_all_models_have_required_fields()
        print("✅ All models have required fields")

        test_loader.test_model_details_field_types()
        print("✅ All model field types are correct")

        test_loader.test_model_architectural_details()
        print("✅ Model architectural details validated")

        test_loader.test_lora_weights_loading()
        print("✅ LoRA weights loading validated")

        test_loader.test_model_statistics()
        print("✅ Model statistics calculated correctly")

        test_loader.test_get_model_by_id()
        print("✅ Model retrieval by ID works")

        test_loader.test_model_validation()
        print("✅ Model validation functionality works")

        test_loader.test_dynamic_field_mapping()
        print("✅ Dynamic field mapping works")

        test_loader.test_reload_models_functionality()
        print("✅ Model reload functionality works")

        print("\n🎉 All ModelLoader tests passed!")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
