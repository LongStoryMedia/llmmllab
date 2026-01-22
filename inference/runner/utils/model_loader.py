import json
import os
import yaml
from typing import Any, Dict, List, Optional

from models import LoraWeight, Model, ModelDetails

from utils.logging import llmmllogger


class ModelLoader:
    def __init__(self):
        self.logger = llmmllogger.bind(module="ModelLoader")
        self._available_models: Dict[str, Model] = {}
        self._load_available_models()

    def _get_model_details_fields(self) -> Dict[str, Any]:
        """Get all fields from ModelDetails with their default values and types."""
        fields = {}
        for field_name, field_info in ModelDetails.model_fields.items():
            fields[field_name] = {
                "default": field_info.default,
                "required": field_info.is_required(),
                "annotation": field_info.annotation,
            }
        return fields

    def _get_model_fields(self) -> Dict[str, Any]:
        """Get all fields from Model with their default values and types."""
        fields = {}
        for field_name, field_info in Model.model_fields.items():
            fields[field_name] = {
                "default": field_info.default,
                "required": field_info.is_required(),
                "annotation": field_info.annotation,
            }
        return fields

    def _load_available_models(self) -> None:
        """Load available models from YAML (preferred) or JSON (legacy) configuration.

        Order of precedence:
        1. MODELS_FILE_PATH env var (can point to YAML or JSON)
        2. /app/.models.yaml
        3. /app/.models.json (legacy fallback)
        """
        candidates: List[str] = []
        env_path = os.environ.get("MODELS_FILE_PATH")
        if env_path:
            candidates.append(env_path)
        candidates.extend(["/app/.models.yaml", "/app/.models.json"])

        chosen_path: Optional[str] = None
        for path in candidates:
            if path and os.path.exists(path):
                chosen_path = path
                break

        if not chosen_path:
            self.logger.error(
                "No models configuration file found (checked env, .models.yaml, .models.json)"
            )
            return

        # Attempt parsing based on extension; allow YAML for .yaml/.yml, otherwise JSON
        try:
            with open(chosen_path, "r", encoding="utf-8") as f:
                if chosen_path.endswith((".yaml", ".yml")):
                    models_data = yaml.safe_load(f)
                else:
                    models_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to parse models file {chosen_path}: {e}")
            return

        if not isinstance(models_data, list):
            self.logger.error(
                f"Models config {chosen_path} is not a list; ignoring contents"
            )
            return

        loaded_count = 0
        for data in models_data:
            try:
                model = self._create_model_from_data(data)
                if model:
                    id_key = str(data.get("id") or model.id or "")
                    if not id_key:
                        self.logger.error(
                            f"Skipping model with missing id: {getattr(model, 'name', 'unknown')}"
                        )
                        continue
                    self._available_models[id_key] = model
                    loaded_count += 1
            except Exception as e:
                self.logger.error(
                    f"Error creating model from {data.get('id', 'unknown')}: {e}"
                )

        self.logger.info(
            f"Loaded {loaded_count}/{len(models_data)} models from config ({os.path.basename(chosen_path)})"
        )

    def _create_model_from_data(self, data: Dict[str, Any]) -> Optional[Model]:
        """Create a Model instance from configuration data with dynamic field mapping."""
        # LoRA weights
        loras: List[LoraWeight] = []
        for lw in data.get("lora_weights", []) or []:
            try:
                loras.append(
                    LoraWeight(
                        id=lw.get("id", ""),
                        name=lw.get("name", ""),
                        weight_name=lw.get("weight_name", ""),
                        adapter_name=lw.get("adapter_name", ""),
                        parent_model=lw.get("parent_model", ""),
                    )
                )
            except Exception:
                continue

        details_dict = data.get("details", {}) or {}

        # Dynamic ModelDetails creation with all available fields
        try:
            model_details_fields = self._get_model_details_fields()
            details_data = {}

            for field_name, field_info in model_details_fields.items():
                value = details_dict.get(field_name)

                # Handle special cases for field mapping and validation
                if field_name == "specialization":
                    # Map invalid specialization values to valid ones
                    if value:
                        specialization_map = {
                            "ImageTextToText": "Text",
                            "TextGeneration": "Text",
                            "TextSummarization": "Text",
                            "TextToText": "Text",
                            "TextToEmbeddings": "Embedding",
                            # Valid values remain unchanged
                            "Text": "Text",
                            "LoRA": "LoRA",
                            "Embedding": "Embedding",
                            "TextToImage": "TextToImage",
                            "ImageToImage": "ImageToImage",
                            "Audio": "Audio",
                        }
                        value = specialization_map.get(value, "Text")

                elif field_name in ["format", "family", "parameter_size", "dtype"]:
                    # Ensure these are strings
                    value = (
                        str(value)
                        if value is not None
                        else (
                            field_info["default"] if not field_info["required"] else ""
                        )
                    )

                elif field_name == "families":
                    # Ensure this is a list
                    value = (
                        list(value)
                        if value is not None
                        else (
                            field_info["default"] if not field_info["required"] else []
                        )
                    )

                elif field_name == "weight":
                    # Ensure this is a float
                    value = (
                        float(value)
                        if value is not None
                        else (
                            field_info["default"] if not field_info["required"] else 1.0
                        )
                    )

                # Handle required fields with appropriate defaults or errors
                if value is None and field_info["required"]:
                    if field_name == "size":
                        # Size is required but might be missing in older configs
                        self.logger.warning(
                            f"Missing required field 'size' for model {data.get('id', 'unknown')}, using 0"
                        )
                        value = 0
                    elif field_name == "original_ctx":
                        # Original context is required but might be missing
                        self.logger.warning(
                            f"Missing required field 'original_ctx' for model {data.get('id', 'unknown')}, using 4096"
                        )
                        value = 4096
                    elif field_name == "format":
                        value = "gguf"  # Default format
                    elif field_name == "family":
                        value = "unknown"  # Default family
                    elif field_name == "families":
                        value = []
                    elif field_name == "parameter_size":
                        value = "unknown"  # Default parameter size
                    else:
                        self.logger.error(
                            f"Missing required field '{field_name}' for model {data.get('id', 'unknown')}"
                        )
                        return None

                # Set the value, using default if None and field is optional
                if value is None and not field_info["required"]:
                    value = (
                        field_info["default"]
                        if field_info["default"] is not None
                        else None
                    )

                details_data[field_name] = value

            details = ModelDetails(**details_data)

        except Exception as e:
            self.logger.error(f"Invalid model details for {data.get('id')}: {e}")
            return None

        # Dynamic Model creation with all available fields
        try:
            model_fields = self._get_model_fields()
            model_data = {}

            for field_name, field_info in model_fields.items():
                if field_name == "details":
                    value = details
                elif field_name == "lora_weights":
                    value = loras
                else:
                    value = data.get(field_name)

                    # Handle required fields
                    if value is None and field_info["required"]:
                        if field_name == "task":
                            value = "TextToText"  # Default task
                        else:
                            self.logger.error(
                                f"Missing required field '{field_name}' for model {data.get('id', 'unknown')}"
                            )
                            return None

                model_data[field_name] = value

            model = Model(**model_data)

        except Exception as e:
            self.logger.error(f"Invalid model entry: {e}")
            return None

        return model

    def get_available_models(self) -> Dict[str, Model]:
        """Get all available models."""
        return self._available_models.copy()

    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Get a specific model by its ID."""
        return self._available_models.get(model_id)

    def reload_models(self) -> None:
        """Reload models from configuration file."""
        self._available_models.clear()
        self._load_available_models()

    def validate_model_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate model data and return list of validation errors."""
        errors = []

        # Check required top-level fields
        required_fields = ["name", "model", "provider", "modified_at", "digest"]
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")

        # Check details section
        details = data.get("details", {})
        if not details:
            errors.append("Missing details section")
        else:
            # Check for required details fields
            if "size" not in details:
                errors.append("Missing required field: details.size")
            if "original_ctx" not in details:
                errors.append("Missing required field: details.original_ctx")

        return errors

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded models."""
        models = self._available_models

        stats = {
            "total_models": len(models),
            "providers": {},
            "specializations": {},
            "families": {},
            "tasks": {},
            "models_with_clip": 0,
            "models_with_lora": 0,
            "average_size_gb": 0,
            "size_range": {"min": float("inf"), "max": 0},
        }

        total_size = 0
        for model in models.values():
            # Provider stats
            provider = str(model.provider)
            stats["providers"][provider] = stats["providers"].get(provider, 0) + 1

            # Specialization stats
            spec = model.details.specialization or "Unknown"
            stats["specializations"][spec] = stats["specializations"].get(spec, 0) + 1

            # Family stats
            family = model.details.family or "Unknown"
            stats["families"][family] = stats["families"].get(family, 0) + 1

            # Task stats
            task = str(model.task)
            stats["tasks"][task] = stats["tasks"].get(task, 0) + 1

            # CLIP model count
            if model.details.clip_model_path:
                stats["models_with_clip"] += 1

            # LoRA count
            if model.lora_weights:
                stats["models_with_lora"] += 1

            # Size stats
            size = model.details.size
            if size:
                total_size += size
                stats["size_range"]["min"] = min(stats["size_range"]["min"], size)
                stats["size_range"]["max"] = max(stats["size_range"]["max"], size)

        if len(models) > 0:
            stats["average_size_gb"] = round((total_size / len(models)) / (1024**3), 2)

        if stats["size_range"]["min"] == float("inf"):
            stats["size_range"]["min"] = 0

        # Convert size range to GB
        stats["size_range"]["min"] = round(stats["size_range"]["min"] / (1024**3), 2)
        stats["size_range"]["max"] = round(stats["size_range"]["max"] / (1024**3), 2)

        return stats
