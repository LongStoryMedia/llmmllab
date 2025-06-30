import gc
import os
import json
import logging
from pickle import NONE
from typing import Callable, Dict, Any, Optional, List, Union, Tuple
from unittest.mock import DEFAULT

import pip
from regex import F
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
from services.hardware_manager import hardware_manager
from models.model import Model
from models.model_details import ModelDetails

from enum import Enum


class QuantizationLevel(Enum):
    """Enum for quantization levels."""
    FOUR_BIT = "4-bit"
    EIGHT_BIT = "8-bit"
    NONE = "none"


def get_sd3l_pipeline() -> StableDiffusion3Pipeline:
    """Get the StableDiffusion3Pipeline instance."""

    c = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # load_in_8bit=True,
        # llm_int8_threshold=6.0,
    )
    # e = T5EncoderModel.from_pretrained(
    #     "diffusers/t5-nf4",
    #     torch_dtype=torch.bfloat16,
    # )
    t = SD3Transformer2DModel.from_pretrained(
        model_service.get_active_model().name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=c,
        # text_encoder=e,
        # device_map="balanced",  # Apply device_map here on the full pipeline
    )
    p = StableDiffusion3Pipeline.from_pretrained(
        model_service.get_active_model().name,
        transformer=t,
        torch_dtype=torch.bfloat16,
        # device_map="balanced",  # Apply device_map here on the full pipeline
    )
    # Enable sequential CPU offloading for low memory
    p.enable_model_cpu_offload()
    p.enable_attention_slicing()  # Enable attention slicing for memory efficiency
    # # if hasattr(p, 'enable_xformers_memory_efficient_attention'):
    # # p.enable_xformers_memory_efficient_attention()
    # if hasattr(p, 'enable_vae_slicing'):
    #     p.enable_vae_slicing()  # Enable VAE slicing for memory efficiency

    return p


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

        # Keep track of the active model details
        self.active_model: str = config.DEFAULT_MODEL_ID
        self.active_pipeline: Optional[DiffusionPipeline] = None
        self.device = hardware_manager.device
        self.last_device = None  # Track the last device used to detect changes

    def get_active_model(self) -> Model:
        """Get the currently active model.

        Returns:
            Model: The active model instance.
        """
        return self.models.get(
            self.active_model, self.models[config.DEFAULT_MODEL_ID])

    def set_active_model(self, model_id: str) -> bool:
        """Set the active model by ID.

        Args:
            model_id (str): The ID of the model to set as active.

        Returns:
            bool: True if the model was set successfully, False otherwise.
        """
        if model_id not in self.models:
            self.logger.error(f"Model not found: {model_id}")
            return False

        # If already active, no need to change
        if model_id == self.active_model and self.active_pipeline is not None:
            self.logger.info(f"Model {model_id} is already active")
            return True

        self.active_model = model_id
        self.active_pipeline = None  # Reset pipeline when changing model
        return True

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
                model = Model(
                    id=model_data.get("id"),
                    name=model_data["name"],
                    model=model_data["model"],
                    modified_at=model_data["modified_at"],
                    size=model_data["size"],
                    digest=model_data["digest"],
                    pipeline=model_data.get("pipeline"),
                    details=ModelDetails(
                        parent_model='',
                        format='gguf',
                        family='llama',
                        families=[],
                        parameter_size='7.2B',
                        quantization_level='Q4_0',
                        specialization='TextToImage'
                    )  # You may want to provide a ModelDetails instance here
                )

                self.models[model_data["id"]] = model

            self.logger.info(f"Loaded {len(self.models)} models from config")
        except Exception as e:
            self.logger.error(f"Error loading models config: {e}")

    def get_models(self) -> List[ModelDetails]:
        """Get a list of all available models with their details."""
        models_list = []
        for _, model in self.models.items():
            details = ModelDetails(
                format=getattr(model, "format", ""),
                family=getattr(model, "family", "") or "",
                families=getattr(model, "families", []),
                parameter_size=getattr(model, "parameter_size", ""),
                quantization_level=getattr(model, "quantization_level", "")
            )
            models_list.append(details)
        return models_list

    def unload_active_model(self) -> None:
        """Unload the active model from GPU memory."""
        if not self.active_model or self.active_pipeline is None:
            return

        try:
            self.logger.info(f"Unloading model: {self.active_model}")

            # Delete the pipeline object
            del self.active_pipeline
            self.active_pipeline = None

            self.logger.info(
                f"Model {self.active_model} successfully unloaded")
        except Exception as e:
            self.logger.error(f"Error unloading model: {e}")
            # Reset the pipeline reference anyway to avoid further issues
            self.active_pipeline = None

        finally:
            # Try to clear memory even if unloading failed
            hardware_manager.clear_memory(aggressive=True)

    def configure_pipeline(self, model: Model, quantization_level: QuantizationLevel, dtype: torch.dtype) -> None:
        """Quantize the model to the specified quantization level."""
        kwargs = {
            "torch_dtype": dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
            "trust_remote_code": True,
        }
        transformer_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": dtype,
        }

        if model.name.find("stable-diffusion-3.5") != -1 or model.name.find("sd3.5") != -1:
            # If the model is SD3.5, we need to use StableDiffusion3Pipeline
            self.logger.info(
                f"Using StableDiffusion3Pipeline for model {model.name}")
            quantization_level = QuantizationLevel.FOUR_BIT  # Default to 8-bit for SD3.5

        if quantization_level == QuantizationLevel.FOUR_BIT:
            self.logger.info(f"Quantizing model {model.name} to 4-bit")
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype
            )
            transformer_kwargs["quantization_config"] = config

        elif quantization_level == QuantizationLevel.EIGHT_BIT:
            self.logger.info(f"Quantizing model {model.name} to 8-bit")
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_8bit_compute_dtype=dtype
            )
            transformer_kwargs["quantization_config"] = config

        t5_nf4 = T5EncoderModel.from_pretrained(
            "diffusers/t5-nf4", torch_dtype=dtype)
        kwargs["text_encoder_3"] = t5_nf4

        if quantization_level != QuantizationLevel.NONE:
            self.logger.info(
                f"adding SD3Transformer2DModel to {model.name} with quantization level {quantization_level}")
            kwargs["transformer"] = SD3Transformer2DModel.from_pretrained(
                model.name,
                **transformer_kwargs
            )

        self.active_pipeline = AutoPipelineForText2Image.from_pretrained(
            model.name, **kwargs)

        if self.active_pipeline is not None:
            self.active_pipeline.enable_model_cpu_offload()
        else:
            raise RuntimeError(
                f"Failed to load model {model.name} with quantization level {quantization_level}")

    def run_pipeline(self, prompt: str, q: QuantizationLevel = QuantizationLevel.NONE, **kwargs) -> Any:
        """Run the pipeline with enhanced memory management."""
        hardware_manager.clear_memory(aggressive=True)
        self.set_active_model(self.get_active_model().name)

        try:
            self.load_active_model(q)
            pipeline = self.get_pipeline(q)

            if pipeline is None:
                raise RuntimeError("Pipeline could not be retrieved.")
            self.logger.info(
                f"Using kwargs for pipeline: {kwargs} with quantization level {q.name}"
            )
            res = pipeline(
                prompt=prompt,
                **kwargs,
            )    # type: ignore
            return res
        except (torch.OutOfMemoryError, RuntimeError) as e:
            self.logger.warning(
                f"Out of memory error for model {self.get_active_model().name} (quantization is {q}): {e}"
            )

            # Enhanced memory clearing logic
            self.logger.info("Unloading active model to free memory.")
            self.unload_active_model()

            self.logger.info("Performing aggressive memory clearing.")
            hardware_manager.clear_memory(aggressive=True)

            # Introduce a short delay to allow system stabilization
            import time
            time.sleep(30)

            # Retry with reduced parameters or fallback
            if q == QuantizationLevel.NONE:
                self.logger.info("Retrying with 8-bit quantization.")
                return self.run_pipeline(prompt, QuantizationLevel.EIGHT_BIT, **kwargs)
            elif q == QuantizationLevel.EIGHT_BIT:
                self.logger.info("Retrying with further reduced parameters.")
                return self.run_pipeline(prompt, QuantizationLevel.FOUR_BIT, **kwargs)

            self.logger.error("Pipeline failed after multiple retries.")
            raise RuntimeError(
                "Pipeline execution failed due to memory issues.")

    def load_active_model(self, q: QuantizationLevel = QuantizationLevel.NONE) -> None:
        """Load the active model into GPU memory."""
        if not self.active_model:
            raise ValueError("No active model set")

        if self.active_pipeline is not None:
            self.logger.info(f"Model {self.active_model} already loaded")
            return

        model = self.get_active_model()
        self.logger.info(
            f"Loading model {self.active_model} ({model.name})")

        # Use bfloat16 precision instead of float32/float16 for better memory efficiency
        # bfloat16 has better numerical stability than float16 and uses less memory than float32
        use_bfloat16 = hardware_manager.has_gpu and torch.cuda.is_available(
        ) and torch.cuda.is_bf16_supported()
        use_fp16 = config.USE_FP16_PRECISION and not use_bfloat16
        if config.FORCE_FP32_ON_CPU and self.device.type != "cuda":
            torch_dtype = torch.float32
        elif use_bfloat16:
            torch_dtype = torch.bfloat16
        elif use_fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        self.configure_pipeline(model, q, torch_dtype)

    def get_pipeline(self, q: QuantizationLevel) -> DiffusionPipeline:
        """Get the currently loaded pipeline, loading it if necessary."""
        if self.active_model is None:
            self.set_active_model(self.get_active_model().name)

        # Load model if not already loaded
        if self.active_pipeline is None:
            self.load_active_model(q)

        if self.active_pipeline is None:
            raise RuntimeError("Failed to load active model pipeline")

        return self.active_pipeline

    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a specific model by ID."""
        return self.models.get(model_id)

    def list_models(self) -> List[str]:
        """List all available model IDs."""
        return list(self.models.keys())

    def add_model(self, name: str, source: str, description: str = "") -> Model:
        """
        Add a new model to the service.
        This does not load the model, it just adds it to the config.
        """
        if name in self.models:
            raise ValueError(f"Model with name '{name}' already exists")

        # Create a new Model instance
        model = Model(
            name=name,
            model=source,
            modified_at="",
            size=0,
            digest="",
            details=ModelDetails(
                parent_model='',
                format='',
                family='',
                families=[],
                parameter_size='',
                quantization_level='',
                specialization='TextToImage'
            )
        )

        # Add to models dictionary
        self.models[name] = model

        # Save to config file
        self.save_models_config()

        return model

    def save_models_config(self) -> None:
        """Save the current models configuration to the models.json file."""
        try:
            models_data = [model for model in self.models.values()]
            with open(self.models_file, 'w') as f:
                json.dump(models_data, f, indent=4)
            self.logger.info(f"Models config saved to {self.models_file}")
        except Exception as e:
            self.logger.error(f"Error saving models config: {e}")

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model by ID.
        This does not unload the model, it just removes it from the config.
        """
        if model_id not in self.models:
            self.logger.error(f"Model with ID {model_id} not found")
            return False

        # If this is the active model, unload it first
        if model_id == self.active_model:
            self.unload_active_model()

        # Remove from models dictionary
        del self.models[model_id]

        # Save updated config
        self.save_models_config()

        self.logger.info(f"Model {model_id} removed successfully")
        return True


# Create a singleton instance
model_service = ModelService()
