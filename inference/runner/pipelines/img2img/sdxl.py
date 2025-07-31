import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.utils.loading_utils import load_image
from ..helpers import get_dtype, get_precision
from models.model import Model
from models import Message
from typing import List, Any
from ..base_pipeline import BasePipeline


class SDXLRefinerPipe(BasePipeline):
    def __init__(self, model: Model):
        """
        Initialize a SDXLRefinerPipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
        """
        self.model = model

        # Load the full pipeline
        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model.model,
            device_map="balanced",
            torch_dtype=get_dtype(model),
            use_safetensors=True,
            variant=get_precision(model),  # Use the precision from the model details
            safety_checker=None,  # Disable safety checker for now
        )
        # self.pipeline.to(hardware_manager.device)

    def run(self, messages: List[Message]) -> Any:
        """
        Process the input messages and generate an image using the SDXL Img2Img pipeline.

        Args:
            messages (List[Message]): The list of messages to process.

        Returns:
            Any: The generated image.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call load() first.")

        # Extract prompt and image from messages
        prompt = ""
        image = None

        for message in messages:
            if message.role == "user":
                if isinstance(message.content, str):
                    prompt = message.content
                elif isinstance(message.content, list):
                    # Handle multimodal input where content is a list of parts
                    for part in message.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            prompt = part.get("text", "")
                        elif isinstance(part, dict) and part.get("type") == "image":
                            # Get image URL
                            image_url = part.get("url", "")
                            if image_url:
                                image = load_image(image_url)

        if not image:
            raise ValueError("No image provided in messages")

        # Generate image with the pipeline
        result = self.pipeline(prompt=prompt, image=image)
        return result

    def __del__(self) -> None:
        """
        Clean up resources used by the SDXLRefinerPipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to('cpu')
                if hasattr(self, 'model') and hasattr(self.model, 'name'):
                    print(f"SDXLRefinerPipe for {self.model.name}: Resources moved to CPU during cleanup")
        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up SDXLRefinerPipe resources: {str(e)}")
