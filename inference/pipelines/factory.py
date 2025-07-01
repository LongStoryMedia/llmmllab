from dataclasses import dataclass, field
import logging
from typing import Any, List, Optional

import torch

from services.model_service import model_service
from models.model import Model


# Basic Formula: A rough estimate for VRAM usage (in GB) is: Number of Parameters (in billions) * (Precision / 8) * 1.2.
# Inference Rule of Thumb (Half Precision): For LLMs in half precision (FP16), estimate approximately 2GB of GPU memory per 1B parameters.
# Inference Estimate: VRAM (in GB) = 2x model parameters (in billions) + 1x context length (in thousands).
# Training Estimate: VRAM (in GB) = 40x model parameters (in billions).


class PipelineFactory:
    """
    Factory class to create pipelines based on model configurations.
    """

    @staticmethod
    def get_pipeline(model_id: str) -> Optional[Any]:
        """
        Get the appropriate pipeline for the given model.

        Args:
            model (Model): The model configuration to create the pipeline for.

        Returns:
            The created pipeline or None if the model type is unsupported.
        """
        if model_service.models is None:
            logging.getLogger(__name__).error("Model service is not initialized or models are not loaded.")
            return None

        if model_id not in model_service.models:
            logging.getLogger(__name__).error(f"Model with ID '{model_id}' not found in the model service.")
            return None

        model = model_service.models[model_id]
        logging.getLogger(__name__).info(f"Creating pipeline for model: {model.name} (ID: {model.id})")

        pipe = PipelineFactory.create_pipeline(model)

        if pipe is None:
            logging.getLogger(__name__).error(f"Failed to create pipeline for model {model.name}.")
            return None

        return pipe

    @staticmethod
    def create_pipeline(model: Model) -> Any:
        """
        Factory method to create and return the appropriate pipeline based on the model configuration.

        Returns:
            torch.nn.Module: The created pipeline or None if the model type is unsupported.
        """
        pipe = None
        if model.pipeline == "StableDiffusion3Pipeline":
            from pipelines.txt2img.sd3 import SD3Pipe
            pipe = SD3Pipe.load(model)
        elif model.pipeline == "StableDiffusionXLPipeline":
            from pipelines.txt2img.sdxl import SDXLPipe
            pipe = SDXLPipe.load(model)
        elif model.pipeline == "FluxPipeline":
            from pipelines.txt2img.flux import FluxPipe
            pipe = FluxPipe.load(model)
        elif model.pipeline == "StableDiffusionXLImg2ImgPipeline":
            from pipelines.img2img.sdxl import SDXLRefinerPipe
            pipe = SDXLRefinerPipe.load(model)
        else:
            logging.getLogger(__name__).error(f"Unsupported pipeline type '{model.pipeline}' for model {model.name}.")
            return None

        PipelineFactory.load_lora_weights(pipe, model)
        return pipe

    @staticmethod
    def load_lora_weights(pipeline: Any, model: Model) -> None:
        """
        Load LoRA weights into the pipeline if available.

        Args:
            pipeline (Any): The pipeline to load LoRA weights into.
            model (Model): The model containing the LoRA weight information.
        """
        if not hasattr(pipeline, 'load_lora_weights'):
            logging.getLogger(__name__).warning(f"Pipeline {pipeline.__class__.__name__} does not support LoRA weights.")
            return

        if model.lora_weights is not None and len(model.lora_weights) > 0:
            for lora_weight in model.lora_weights:
                lw_kwargs = {}
                if lora_weight.weight_name:
                    lw_kwargs['weight_name'] = lora_weight.weight_name
                if lora_weight.adapter_name:
                    lw_kwargs['adapter_name'] = lora_weight.adapter_name

                logging.getLogger(__name__).info(
                    f"Loading LoRA weight '{lora_weight.name}' for model '{model.name}' with kwargs: {lw_kwargs}"
                )

                # Load the LoRA weights into the pipeline
                pipeline.load_lora_weights(
                    lora_weight.name,
                    **lw_kwargs
                )
