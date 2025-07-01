from ..helpers import get_dtype
from models.model import Model
from typing import Optional
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel


class SD3Pipe:
    @staticmethod
    def _setup_quantization_config(model: Model) -> Optional[BitsAndBytesConfig]:
        """
        Set up the quantization configuration based on the model details.

        Args:
            model (Model): The model configuration.

        Returns:
            Optional[BitsAndBytesConfig]: The quantization configuration or None.
        """
        if model.details is not None and model.details.quantization_level is not None:
            if model.details.quantization_level.lower().startswith(("q4", "int4", "nf4")):
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif model.details.quantization_level.lower().startswith(("q8", "int8")):
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        return None

    @staticmethod
    def load(model: Model) -> StableDiffusion3Pipeline:
        """        
        Load the Stable Diffusion 3 pipeline with quantization and memory optimizations.

        Args:
            model (Model): The model configuration to load.
        Returns:
            StableDiffusion3Pipeline: The loaded pipeline with quantization and optimizations.
        """
        dtype = get_dtype(model)
        quantization_config = SD3Pipe._setup_quantization_config(model)
        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
        }

        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = quantization_config

        # Load the transformer model
        transformer = SD3Transformer2DModel.from_pretrained(
            model.model,
            **transformer_kwargs,
        )

        # Load the full pipeline
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model.model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        # Apply memory optimization techniques
        pipeline.enable_model_cpu_offload()
        # pipeline.enable_attention_slicing()

        return pipeline
