from ..helpers import get_dtype
from models.model import Model
from models import Message
from typing import Optional, List, Any
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
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
        self.model = model

        dtype = get_dtype(model)
        quantization_config = self._setup_quantization_config(model)
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
            torch_dtype=torch.bfloat16,
        )

        # Apply memory optimization techniques
        self.pipeline.enable_model_cpu_offload()
        # self.pipeline.enable_attention_slicing()

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
        Process the input messages and generate an image using the SD3 pipeline.

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
        Clean up resources used by the SD3Pipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to('cpu')
                if hasattr(self, 'model') and hasattr(self.model, 'name'):
                    print(f"SD3Pipe for {self.model.name}: Resources moved to CPU during cleanup")
        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up SD3Pipe resources: {str(e)}")
