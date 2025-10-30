"""
Qwen3 VL pipeline for multimodal (text + image) generation.
Optimized for Qwen3 VL models with vision capabilities.
"""

import os
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel

from models import Model, ModelProfile
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3VLPipeline(BaseLlamaCppPipeline):
    """
    Qwen3 VL multimodal chat model implementation.

    Features:
    - Optimized for Qwen3 VL models (e.g., Qwen3-VL-32B-Thinking-abliterated)
    - Vision capabilities with multimodal processing
    - Custom chat format for Qwen3 VL models
    - Hardware optimization for large VL models
    - <think> tag processing for reasoning models
    - Supports image and video inputs
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
        return "qwen3-vl-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "qwen3-vl",
                "vision_capable": True,
                "multimodal": True,
                "chat_format": "chatml",
                "supports_thinking": True,
            }
        )
        return base_params

    def _get_mmproj_path(self) -> Optional[str]:
        """Get the mmproj file path for multimodal processing."""
        # Check if mmproj file is available in the model directory
        model_dir = os.path.dirname(self._get_gguf_path())
        mmproj_path = os.path.join(model_dir, "mmproj-model-f16.gguf")
        
        if os.path.exists(mmproj_path):
            return mmproj_path
        
        # Try alternative naming patterns
        alternative_paths = [
            os.path.join(model_dir, "mmproj.gguf"),
            os.path.join(model_dir, "mmproj-f16.gguf"),
            os.path.join(model_dir, "multimodal.gguf"),
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                return path
        
        self._logger.warning(f"No mmproj file found in {model_dir}")
        return None

    def _get_chat_format(self) -> str:
        """Get the appropriate chat format for Qwen3 VL."""
        # Qwen3 VL uses a custom chat format with vision tokens
        # For now, we'll use None to let llama.cpp auto-detect
        # This may need to be updated based on the specific model requirements
        return None


__all__ = ["Qwen3VLPipeline"]