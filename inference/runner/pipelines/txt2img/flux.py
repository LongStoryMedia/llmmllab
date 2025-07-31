import logging
from ..helpers import get_dtype
from models.model import Model
from models import Message
from typing import Optional, List, Any
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from ..base_pipeline import BasePipeline


class FluxPipe(BasePipeline):
    def __init__(self, model: Model):
        """
        Initialize a FluxPipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
        """
        self.model = model
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading Flux pipeline for model: {model.name} (ID: {model.id}, dtype: {torch.bfloat16})")
        quantization_config = self._setup_quantization_config(model)
        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
        }

        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = quantization_config

        # Load the transformer model
        transformer = FluxTransformer2DModel.from_pretrained(
            model.model,
            **transformer_kwargs,
        )

        # Load the full pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            model.name,
            # device_map="balanced",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            transformer=transformer,
        )

        self.pipeline.enable_model_cpu_offload()

        if not hasattr(self.pipeline, 'load_lora_weights'):
            self.logger.warning(f"Pipeline {type(self.pipeline).__name__} does not support LoRA weights.")
            return

        if model.lora_weights is not None and len(model.lora_weights) > 0:
            for lora_weight in model.lora_weights:
                lw_kwargs = {}
                if lora_weight.weight_name:
                    lw_kwargs['weight_name'] = lora_weight.weight_name
                if lora_weight.adapter_name:
                    lw_kwargs['adapter_name'] = lora_weight.adapter_name

                self.logger.info(
                    f"Loading LoRA weight '{lora_weight.name}' for model '{model.name}' with kwargs: {lw_kwargs}"
                )

                # Load the LoRA weights into the pipeline
                self.pipeline.load_lora_weights(
                    lora_weight.name,
                    **lw_kwargs
                )
        # self.pipeline.enable_sequential_cpu_offload()

    def _setup_quantization_config(self, model: Model) -> Optional[BitsAndBytesConfig]:
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

    def run(self, messages: List[Message]) -> Any:
        """
        Process the input messages and generate an image using the Flux pipeline.

        Args:
            messages (List[Message]): The list of messages to process.

        Returns:
            Any: The generated image.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call load() first.")

        # Extract prompt from messages
        prompt = ""
        for message in messages:
            if message.role == "user" and isinstance(message.content, str):
                prompt = message.content
                break

        # Generate image with the pipeline
        result = self.pipeline(prompt=prompt)
        return result

    def __del__(self) -> None:
        """
        Clean up resources used by the FluxPipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to('cpu')
                self.logger.debug(f"FluxPipe for {self.model.name}: Resources moved to CPU during cleanup")
        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up FluxPipe resources: {str(e)}")
