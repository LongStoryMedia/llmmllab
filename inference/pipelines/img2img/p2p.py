from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
from diffusers.utils.loading_utils import load_image
import torch
from ..helpers import get_dtype, get_precision
from models.model import Model
from models import Message
from typing import List, Any
from ..base_pipeline import BasePipeline


class Pix2PixPipe(BasePipeline):
    def __init__(self, model: Model):
        """
        Initialize a Pix2PixPipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
        """
        self.model = model

        # Load the full pipeline
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model.model,
            device_map="balanced",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant=get_precision(model),  # Use the precision from the model details
            safety_checker=None,  # Disable safety checker for now
            attn_implementation="eager",
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

        # self.pipeline.to(hardware_manager.device)

    def run(self, messages: List[Message]) -> Any:
        """
        Process the input messages and generate an image using the Instruct Pix2Pix pipeline.

        Args:
            messages (List[Message]): The list of messages to process.

        Returns:
            Any: The generated image.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call load() first.")

        # Extract prompt, instruction and image from messages
        prompt = ""
        instruction = ""
        image = None

        for message in messages:
            if message.role == "user":
                if isinstance(message.content, str):
                    # In p2p pipeline, the content should be the instruction
                    instruction = message.content
                elif isinstance(message.content, list):
                    # Handle multimodal input where content is a list of parts
                    for part in message.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            # The text is the instruction
                            instruction = part.get("text", "")
                        elif isinstance(part, dict) and part.get("type") == "image":
                            # Get image URL
                            image_url = part.get("url", "")
                            if image_url:
                                image = load_image(image_url)

        if not image:
            raise ValueError("No image provided in messages")

        if not instruction:
            raise ValueError("No instruction provided in messages")

        # Generate image with the pipeline
        result = self.pipeline(prompt="", image=image, instruction=instruction)
        return result

    def __del__(self) -> None:
        """
        Clean up resources used by the Pix2PixPipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to('cpu')
                if hasattr(self, 'model') and hasattr(self.model, 'name'):
                    print(f"Pix2PixPipe for {self.model.name}: Resources moved to CPU during cleanup")
        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up Pix2PixPipe resources: {str(e)}")
