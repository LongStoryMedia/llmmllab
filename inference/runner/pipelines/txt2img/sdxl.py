from ..helpers import get_dtype, get_precision
from models.model import Model
from models import ChatReq
from typing import Any
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from ..base_pipeline import BasePipeline


class SDXLPipe(BasePipeline):
    def __init__(self, model: Model):
        """
        Initialize a SDXLPipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
        """
        super().__init__()
        self.model = model

        # Load the full pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model.model,
            device_map="balanced",
            torch_dtype=get_dtype(model),
            use_safetensors=True,
            variant=get_precision(model),  # Use the precision from the model details
            safety_checker=None,  # Disable safety checker for now
        )
        # self.pipeline.to(hardware_manager.device)

    def run(self, req: ChatReq) -> Any:
        """
        Process the input request and generate an image using the SDXL pipeline.

        Args:
            req (ChatReq): The chat request containing messages and options.

        Returns:
            Any: The generated image.
        """
        messages = req.messages
        # We can access options later if needed: options = req.options
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
        Clean up resources used by the SDXLPipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to("cpu")
                if hasattr(self, "model") and hasattr(self.model, "name"):
                    print(
                        f"SDXLPipe for {self.model.name}: Resources moved to CPU during cleanup"
                    )
        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up SDXLPipe resources: {str(e)}")
