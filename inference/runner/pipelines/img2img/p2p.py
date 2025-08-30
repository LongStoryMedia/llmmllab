"""
Pipeline for Pix2Pix (InstructPix2Pix) image-to-image models.
Clean implementation with only essential methods for public API.
"""

import datetime
import logging
from typing import Optional, List

import torch
from langchain_core.tools import BaseTool
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.utils.loading_utils import load_image

from models import (
    Model,
    Message,
    ChatResponse,
    MessageRole,
    MessageContent,
    MessageContentType,
    ModelProfile,
)
from ..base import BasePipelineCore


class Pix2PixPipe(BasePipelineCore[ChatResponse]):
    """
    Pipeline for InstructPix2Pix image editing models.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize a Pix2PixPipe instance."""
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)
        self.pipeline: Optional[StableDiffusionInstructPix2PixPipeline] = None

        self.logger.info(f"Initialized Pix2Pix pipeline: {model.name}")

    def _initialize_pipeline(self) -> None:
        """Initialize the Pix2Pix pipeline."""
        if self.pipeline is not None:
            return

        self.logger.info(f"Loading Pix2Pix pipeline for model: {self.model.name}")

        # Load the full pipeline
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model.model,
            device_map="balanced",
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            attn_implementation="eager",
        )

        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> ChatResponse:
        """Process messages and generate an image editing response."""
        # Initialize pipeline if needed
        if self.pipeline is None:
            self._initialize_pipeline()

        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        instruction = ""
        image = None

        for message in messages:
            if hasattr(message, "content") and message.content:
                for part in message.content:
                    if (
                        hasattr(part, "type")
                        and part.type == MessageContentType.TEXT
                        and hasattr(part, "text")
                        and part.text
                    ):
                        instruction = part.text
                    if (
                        hasattr(part, "type")
                        and part.type == MessageContentType.IMAGE
                        and hasattr(part, "url")
                        and part.url
                    ):
                        image = load_image(part.url)

        if not image or not instruction:
            return ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="Missing image or instruction for Pix2Pix",
                        )
                    ],
                ),
                finish_reason="error",
            )

        try:
            # Run the pipeline for instruction-based image editing
            self.pipeline(prompt="", image=image, instruction=instruction)
            # In a real implementation, you would save the image and return its URL
            image_url = "p2p_result.png"

            return ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(type=MessageContentType.IMAGE, url=image_url)
                    ],
                ),
            )
        except Exception as e:
            self.logger.error(f"Error editing image: {e}")
            return ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error editing image: {e}",
                        )
                    ],
                ),
                finish_reason="error",
            )

    def create_graph(self, tools: Optional[List[BaseTool]] = None):  # type: ignore[override]
        """Image pipelines do not use LangGraph graphs."""
        raise NotImplementedError("Image pipelines do not use LangGraph graphs")
