"""
Qwen 2.5 VL pipeline - Simplified implementation following qwen3 pattern.
"""

from typing import Optional, Type, Dict, Any

from models import (
    Model,
    ModelProfile,
)
from pydantic import BaseModel
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen25VLPipeline(BaseLlamaCppPipeline):
    """Qwen 2.5 VL pipeline with LangGraph support."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs
    ):
        super().__init__(model, profile, grammar, **kwargs)

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
