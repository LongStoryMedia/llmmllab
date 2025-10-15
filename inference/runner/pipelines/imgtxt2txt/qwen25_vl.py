"""
Qwen 2.5 VL pipeline - Simplified implementation following qwen3 pattern.
"""

import logging
from typing import Union, Dict, Any

from models import (
    Model,
    ChatResponse,
    ModelProfile,
)
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen25VLPipeline(BaseLlamaCppPipeline):
    """Qwen 2.5 VL pipeline with LangGraph support."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        **kwargs,
    ):
        """Initialize a Qwen25VLPipeline instance."""
        super().__init__(model, profile, **kwargs)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info("Qwen25VLPipeline initialized")

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "qwen25-vl-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "qwen25-vl",
                "vision_capable": True,
            }
        )
        return base_params
