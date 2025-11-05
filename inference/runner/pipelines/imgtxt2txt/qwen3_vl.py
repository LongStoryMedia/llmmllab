"""
Qwen3 VL pipeline for multimodal (text + image) generation.
Optimized for Qwen3 VL models with vision capabilities.
"""

import os
import json
from typing import Dict, Any, Optional, Type, List
from llama_cpp.llama import Llama
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler, Qwen25VLChatHandler
from pydantic import BaseModel  # noqa: F401

# llama_cpp imported lazily within methods to reduce unnecessary top-level dependencies
# Pillow not required for text-only stabilization; multimodal image loading currently disabled.

from models import Model, ModelProfile, OptimalParameters
from models.default_configs import DEFAULT_GPU_CONFIG
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3VLPipeline(BaseLlamaCppPipeline):
    """Qwen3 VL multimodal chat model implementation."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ):
        self._multimodal_chat_handler = None
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

    def initialize_llama_with_optimization(self) -> Llama:
        """Override to handle multimodal chat handler with optimization."""
        if self.model.details.clip_model_path:
            # Create the vision handler
            handler = Qwen25VLChatHandler(
                clip_model_path=self.model.details.clip_model_path,
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
            )
            
            # Use the parent optimization logic with our custom vision handler
            gguf_path = self._get_gguf_path()
            
            # Let the base class handle ALL parameter optimization
            # We just need to pass the vision handler
            self._logger.info("🔍 DEBUG Qwen3VL delegating to base class with vision handler")
            return self._initialize_llama(gguf_path, handler=handler)
        else:
            # No vision support, use parent method
            return super().initialize_llama_with_optimization()


__all__ = ["Qwen3VLPipeline"]
