from ..helpers import get_dtype
from models.model import Model
from models import ChatReq
from typing import Optional, Any
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from ..base_pipeline import BasePipeline


class SD3Pipe(BasePipeline):
    def __init__(self, model: Model):
        """
        Initialize a SD3Pipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
        """
        super().__init__()
        self.model = model
        self.model_def = model

        # Use the get_dtype function directly in pipeline initialization instead of storing
        quantization_config = self._setup_quantization_config()
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
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model.model,
            transformer=transformer,
            torch_dtype=get_dtype(model),  # Use the dtype from the model details
        )

        # Apply memory optimization techniques
        self.pipeline.enable_model_cpu_offload()
        # self.pipeline.enable_attention_slicing()

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Set up the quantization configuration based on the model details.

        Returns:
            Optional[BitsAndBytesConfig]: The quantization configuration or None.
        """
        return super()._setup_quantization_config()

    def run(self, req: ChatReq) -> Any:
        """
        Process the input messages and generate an image using the SD3 pipeline.

        Args:
            req (ChatReq): The chat request containing messages and parameters.

        Returns:
            Any: The generated image.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call load() first.")

        messages = req.messages

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
        Clean up resources used by the SD3Pipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to("cpu")
                if hasattr(self, "model") and hasattr(self.model, "name"):
                    print(
                        f"SD3Pipe for {self.model.name}: Resources moved to CPU during cleanup"
                    )
        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up SD3Pipe resources: {str(e)}")
