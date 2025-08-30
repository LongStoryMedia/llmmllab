"""
Pipeline for SDXL image-to-image models.
Clean implementation with only essential methods for public API.
"""

import datetime
import logging
import time
from typing import List, Optional

import torch
from langchain_core.tools import BaseTool
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils.loading_utils import load_image

from models import (
    Model,
    ModelProfile,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ChatResponse,
)
from ..base import BasePipelineCore


def get_dtype(model: Model) -> torch.dtype:
    """Simple helper to get dtype from model."""
    _ = model  # Unused for now
    return torch.bfloat16


def get_precision(model: Model) -> str:
    """Simple helper to get precision from model."""
    _ = model  # Unused for now  
    return "fp16"


class SDXLRefinerPipe(BasePipelineCore[ChatResponse]):
    """
    Image-to-image pipeline for SDXL refiner models.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)
        self.pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None
        
        self.logger.info(f"Initialized SDXL Refiner pipeline: {model.name}")

    def _initialize_pipeline(self) -> None:
        """Initialize the SDXL Img2Img pipeline."""
        if self.pipeline is not None:
            return
            
        self.logger.info(f"Loading SDXL Refiner pipeline for model: {self.model.name}")

        # Load the full pipeline
        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model.model,
            device_map="balanced",
            torch_dtype=get_dtype(self.model),
            use_safetensors=True,
            variant=get_precision(self.model),
            safety_checker=None,
        )

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> ChatResponse:
        """Process messages and generate an image-to-image response."""
        # Initialize pipeline if needed
        if self.pipeline is None:
            self._initialize_pipeline()
            
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        # Extract prompt and input image from messages
        prompt_text = ""
        image = None

        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                for part in msg.content:
                    if hasattr(part, "text") and part.text:
                        prompt_text += str(part.text) + "\n"
                    if hasattr(part, "type") and part.type == MessageContentType.IMAGE:
                        if hasattr(part, "url") and part.url:
                            image = load_image(part.url)

        prompt_text = prompt_text.strip() or "Refine this image"
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        if image is None:
            return ChatResponse(
                created_at=start_time,
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="No input image provided in messages",
                        ),
                    ],
                ),
                finish_reason="error",
            )

        try:
            # Run the pipeline for image-to-image generation
            self.pipeline(prompt=prompt_text, image=image)
            # In a real implementation, you would save the image and return its URL
            image_url = f"refined_image_{int(time.time())}.png"
            
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)

            return ChatResponse(
                created_at=start_time,
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Refined image based on prompt: {prompt_text}",
                        ),
                        MessageContent(type=MessageContentType.IMAGE, url=image_url),
                    ],
                    created_at=end_time,
                ),
                finish_reason="stop",
                total_duration=(end_time - start_time).total_seconds() * 1000.0,
            )
            
        except Exception as e:
            self.logger.error(f"Error refining image: {e}")
            return ChatResponse(
                created_at=start_time,
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error refining image: {e}",
                        ),
                    ],
                ),
                finish_reason="error",
            )

    def create_graph(self, tools: Optional[List[BaseTool]] = None):  # type: ignore[override]
        """Image pipelines do not use LangGraph graphs."""
        raise NotImplementedError("Image pipelines do not use LangGraph graphs")
