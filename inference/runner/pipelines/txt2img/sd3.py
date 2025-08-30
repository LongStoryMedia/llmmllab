"""
Pipeline for Stable Diffusion 3 text-to-image models.
Clean implementation with only essential methods for public API.
"""

import datetime
import logging
from typing import List, Optional

import torch
from langchain_core.tools import BaseTool
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

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


class SD3Pipe(BasePipelineCore[ChatResponse]):
    """
    Pipeline class for Stable Diffusion 3 text-to-image models.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize a SD3Pipe instance."""
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)
        self.pipeline: Optional[StableDiffusion3Pipeline] = None

        self.logger.info(f"Initialized SD3 pipeline: {model.name}")

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Set up the quantization configuration."""
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )

    def _initialize_pipeline(self) -> None:
        """Initialize the SD3 pipeline."""
        if self.pipeline is not None:
            return

        self.logger.info(f"Loading SD3 pipeline for model: {self.model.name}")

        # Setup quantization and transformer
        quantization_config = self._setup_quantization_config()
        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
        }

        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = quantization_config

        # Load the transformer model
        transformer = SD3Transformer2DModel.from_pretrained(
            self.model.model,
            **transformer_kwargs,
        )

        # Load the full pipeline
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model.model,
            transformer=transformer,
            torch_dtype=get_dtype(self.model),
        )

        # Apply memory optimization techniques
        self.pipeline.enable_model_cpu_offload()

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
                            text=f"Generated SD3 image for prompt: {prompt}",
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
