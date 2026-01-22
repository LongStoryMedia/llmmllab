"""
Pipeline for Flux text-to-image models.
Clean implementation with only essential methods for public API.
"""

import logging
import datetime
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
from ..base import BasePipelineCore


class FluxPipe(BasePipelineCore[ChatResponse]):
    """
    Pipeline class for Flux text-to-image models.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)
        self._pipeline: Optional[FluxPipeline] = None

        self.logger.info(f"Initialized Flux pipeline: {model.name}")

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration for the model."""
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )

    def _initialize_pipeline(self) -> None:
        """Initialize the Flux pipeline."""
        if self._pipeline is not None:
            return

        self.logger.info(f"Loading Flux pipeline for model: {self.model.name}")

        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
        }

        quantization_config = self._setup_quantization_config()
        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = quantization_config

        transformer = FluxTransformer2DModel.from_pretrained(
            self.model.model, **transformer_kwargs
        )

        self._pipeline = FluxPipeline.from_pretrained(
            self.model.name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            transformer=transformer,
        )

        if self._pipeline:
            self._pipeline.enable_model_cpu_offload()

            # Load LoRA weights if specified
            if hasattr(self._pipeline, "load_lora_weights") and self.model.lora_weights:
                for lora_weight in self.model.lora_weights:
                    lw_kwargs = {}
                    if lora_weight.weight_name:
                        lw_kwargs["weight_name"] = lora_weight.weight_name
                    if lora_weight.adapter_name:
                        lw_kwargs["adapter_name"] = lora_weight.adapter_name
                    self.logger.info(f"Loading LoRA weight '{lora_weight.name}'")
                    self._pipeline.load_lora_weights(lora_weight.name, **lw_kwargs)

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> ChatResponse:
        """Process messages and generate an image response."""
        # Initialize pipeline if needed
        if self._pipeline is None:
            self._initialize_pipeline()

        if not self._pipeline:
            raise ValueError("Pipeline not initialized")

        # Extract prompt text from messages
        prompt_text = ""
        for message in messages:
            if hasattr(message, "content") and message.content:
                for content in message.content:
                    if hasattr(content, "text") and content.text:
                        prompt_text += str(content.text) + "\n"
        prompt_text = prompt_text.strip() or "A beautiful landscape"

        try:
            # Generate image
            self._pipeline(prompt=prompt_text)
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
                            text=f"Generated image for prompt: {prompt_text}",
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
