"""
Qwen3 MoE pipeline as BaseChatModel implementation.
Provides custom model-specific optimizations for Qwen MoE models.
"""

from typing import Dict, Any, Optional, Type
from pydantic import BaseModel

from models import Model, ModelProfile
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3Moe(BaseLlamaCppPipeline):
    """
    Qwen3 MoE chat model implementation.

    Features:
    - Optimized for Qwen3 MoE models (e.g., Qwen2.5-Coder-32B-Instruct)
    - Custom chat format for Qwen models
    - Hardware optimization for MoE architecture
    - <think> tag processing for reasoning models
    """

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
        return "qwen3-moe-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "qwen3-moe",
                "chat_format": "chatml",
            }
        )
        return base_params


__all__ = ["Qwen3Moe"]
