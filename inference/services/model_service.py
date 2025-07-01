import gc
import os
import json
import logging
from pickle import NONE
from typing import Callable, Dict, Any, Optional, List, Union, Tuple
from unittest.mock import DEFAULT

import pip
from regex import D, F
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from transformers import T5EncoderModel

# Import config to use configuration values
import config
from models.lora_weight import LoraWeight
from services.hardware_manager import hardware_manager
from models.model import Model
from models.model_details import ModelDetails


class ModelService:
    """
    Service for managing diffusion models, including loading and unloading from GPU.
    """

    def __init__(self):
        """Initialize the model service."""
        self.logger = logging.getLogger(__name__)

        # Use config values instead of environment variables
        self.models_file = config.MODELS_CONFIG_PATH

        self.logger.info(f"Using models config from {self.models_file}")

        # Load models config
        self.models: Dict[str, Model] = {}
        self.loras: Dict[str, Model] = {}  # Placeholder for LoRA models
        self.load_models_config()

    def load_models_config(self) -> None:
        """Load model configuration from the models.json file."""
        try:
            if not os.path.exists(self.models_file):
                self.logger.error(
                    f"Models config file not found: {self.models_file}")
                return

            with open(self.models_file, 'r') as f:
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
                    quantization_level=details_dict.get("quantization_level", ""),
                    specialization=details_dict.get("specialization", "")
                )

                model = Model(
                    id=model_data.get("id"),
                    name=model_data["name"],
                    model=model_data["model"],
                    modified_at=model_data["modified_at"],
                    size=model_data["size"],
                    digest=model_data["digest"],
                    pipeline=model_data.get("pipeline"),
                    lora_weights=loras,
                    details=details  # You may want to provide a ModelDetails instance here
                )

                self.models[model_data["id"]] = model

            self.logger.info(f"Loaded {len(self.models)} models from config")
        except Exception as e:
            self.logger.error(f"Error loading models config: {e}")

    def get_models(self) -> Dict[str, Model]:
        """Get the loaded models."""
        return self.models


# Create a singleton instance
model_service = ModelService()
