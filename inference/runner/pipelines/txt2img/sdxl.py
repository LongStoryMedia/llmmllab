"""
Pipeline for Stable Diffusion XL text-to-image models.
Clean implementation with only essential methods for public API.
"""

import datetime
import logging
from typing import List, Optional

import torch
from langchain_core.tools import BaseTool
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

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


class SDXLPipe(BasePipelineCore[ChatResponse]):
    """
    Pipeline class for Stable Diffusion XL text-to-image models.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize a SDXLPipe instance."""
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)
        self.pipeline: Optional[StableDiffusionXLPipeline] = None

        self.logger.info(f"Initialized SDXL pipeline: {model.name}")

    def _initialize_pipeline(self) -> None:
        """Initialize the SDXL pipeline."""
        if self.pipeline is not None:
            return

        self.logger.info(f"Loading SDXL pipeline for model: {self.model.name}")

        # Load the full pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model.model,
            device_map="balanced",
            torch_dtype=get_dtype(self.model),
            use_safetensors=True,
            variant=get_precision(self.model),
            safety_checker=None,  # Disable safety checker
        )

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> ChatResponse:
        """Process messages and generate an image response."""
        # Initialize pipeline if needed
        if self.pipeline is None:
            self._initialize_pipeline()

        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        # Extract prompt from messages
        prompt = ""
        for message in messages:
            if hasattr(message, "content") and message.content:
                for content in message.content:
                    if hasattr(content, "text") and content.text:
                        prompt += str(content.text) + " "
        prompt = prompt.strip() or "A beautiful landscape"

        try:
            # Generate image with the pipeline
            self.pipeline(prompt=prompt)
            # In a real implementation, you would save the image and return its URL
            image_url = "generated_image.png"

            return ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Generated SDXL image for prompt: {prompt}",
                        ),
                        MessageContent(type=MessageContentType.IMAGE, url=image_url),
                    ],
                ),
            )
        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            return ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error generating image: {str(e)}",
                        )
                    ],
                ),
                finish_reason="error",
            )

    def create_graph(self, tools: Optional[List[BaseTool]] = None):  # type: ignore[override]
        """Image pipelines do not use LangGraph graphs."""
        raise NotImplementedError("Image pipelines do not use LangGraph graphs")
