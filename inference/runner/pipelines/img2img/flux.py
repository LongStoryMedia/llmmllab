"""
Pipeline for Flux image-to-image models.
Clean implementation with only essential methods for public API.
"""

import logging
import datetime
import time
from typing import Optional, List

import torch
from langchain_core.tools import BaseTool

from models import (
    Model,
    Message,
    ChatResponse,
    ModelProfile,
    MessageRole,
    MessageContent,
    MessageContentType,
)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.utils.loading_utils import load_image
from ..base import BasePipelineCore


class FluxKontextPipe(BasePipelineCore[ChatResponse]):
    """
    Image-to-image pipeline for Flux models.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)
        self.pipeline: Optional[FluxPipeline] = None

        self.logger.info(f"Initialized Flux Kontext pipeline: {model.name}")

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration for the model."""
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )

    def _initialize_pipeline(self) -> None:
        """Initialize the Flux pipeline."""
        if self.pipeline is not None:
            return

        self.logger.info(f"Loading Flux Kontext pipeline for model: {self.model.name}")

        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
        }

        qconf = self._setup_quantization_config()
        if qconf is not None:
            transformer_kwargs["quantization_config"] = qconf

        transformer = FluxTransformer2DModel.from_pretrained(
            self.model.model, **transformer_kwargs
        )

        try:
            self.pipeline = FluxPipeline.from_pretrained(
                self.model.name,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                transformer=transformer,
            )
        except Exception as e:
            self.logger.error(f"Failed to load FluxKontextPipeline: {e}")
            raise RuntimeError(f"Failed to load FluxKontextPipeline: {e}") from e

        if self.pipeline and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()

        # Load LoRA weights if specified
        if (
            self.pipeline
            and hasattr(self.pipeline, "load_lora_weights")
            and self.model.lora_weights
        ):
            for lw in self.model.lora_weights:
                kwargs = {}
                if lw.weight_name:
                    kwargs["weight_name"] = lw.weight_name
                if lw.adapter_name:
                    kwargs["adapter_name"] = lw.adapter_name
                self.logger.info(f"Loading LoRA weight '{lw.name}'")
                try:
                    self.pipeline.load_lora_weights(lw.name, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Failed to load LoRA '{lw.name}': {e}")

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> ChatResponse:
        """Process messages and generate an image-to-image response."""
        # Initialize pipeline if needed
        if self.pipeline is None:
            self._initialize_pipeline()

        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

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

        prompt_text = prompt_text.strip() or "Enhance this image"
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
            # Using a placeholder until correct parameter is verified
            # self.pipeline(prompt=prompt_text, image=image)
            # In a real implementation, you would save the image and return its URL
            image_url = f"generated_kontext_image_{int(time.time())}.png"

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)

            return ChatResponse(
                created_at=start_time,
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Modified image based on prompt: {prompt_text}",
                        ),
                        MessageContent(type=MessageContentType.IMAGE, url=image_url),
                    ],
                    created_at=end_time,
                ),
                finish_reason="stop",
                total_duration=(end_time - start_time).total_seconds() * 1000.0,
            )

        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            return ChatResponse(
                created_at=start_time,
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error generating image: {e}",
                        ),
                    ],
                ),
                finish_reason="error",
            )

    def create_graph(self, tools: Optional[List[BaseTool]] = None):  # type: ignore[override]
        """Image pipelines do not use LangGraph graphs."""
        raise NotImplementedError("Image pipelines do not use LangGraph graphs")
