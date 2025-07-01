import os
import json
import logging
from typing import Dict, List, Optional

from models.model import Model
from models.model_details import ModelDetails
from models.lora_weight import LoraWeight


def load_test_models() -> Dict[str, Model]:
    """
    Load models from the models.json file for testing purposes.

    Returns:
        Dict[str, Model]: A dictionary of Model instances indexed by model ID.
    """
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(project_root, 'config', 'models.json')
    models = {}

    try:
        if not os.path.exists(models_file):
            logging.error(f"Models config file not found: {models_file}")
            return {}

        with open(models_file, 'r') as f:
            models_data: List[dict] = json.load(f)

        for model_data in models_data:
            # Create Model instance
            lora_weights: List[dict] = model_data.get("lora_weights", [])
            loras: List[LoraWeight] = []
            if len(lora_weights) > 0:
                for lora in lora_weights:
                    # Load LoRA weights if available
                    if lora is not None:
                        lora_weight_instance = LoraWeight(
                            id=lora.get("id", ""),
                            name=lora.get("name", ""),
                            weight_name=lora.get("weight_name", ""),
                            adapter_name=lora.get("adapter_name", ""),
                            parent_model=lora.get("parent_model", ""),
                        )
                        loras.append(lora_weight_instance)

            details_dict: dict = model_data.get("details", {})
            details = ModelDetails(
                parent_model=details_dict.get("parent_model", ""),
                format=details_dict.get("format", ""),
                family=details_dict.get("family", ""),
                families=details_dict.get("families", []),
                parameter_size=details_dict.get("parameter_size", ""),
                dtype=details_dict.get("dtype", "float32"),
                quantization_level=details_dict.get("quantization_level", ""),
                specialization=details_dict.get("specialization", "")
            )

            model = Model(
                id=model_data.get("id"),
                name=model_data["name"],
                model=model_data["model"],
                modified_at=model_data.get("modified_at", ""),
                size=model_data.get("size", 0),
                digest=model_data.get("digest", ""),
                pipeline=model_data.get("pipeline"),
                lora_weights=loras,
                details=details
            )

            models[model_data["id"]] = model

        logging.info(f"Loaded {len(models)} models from config for testing")
        return models
    except Exception as e:
        logging.error(f"Error loading models from config: {e}")
        return {}
