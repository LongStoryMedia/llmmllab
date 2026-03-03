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

from runner.models import Model, ModelProfile, OptimalParameters
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

    def _initialize_llama(
        self,
        gguf_path: str,
        h: LlamaChatCompletionHandler | None = None,
        force_params: Optional[OptimalParameters] = None,
    ) -> Llama:
        if self.model.details.clip_model_path:
            handler = Qwen25VLChatHandler(
                clip_model_path=self.model.details.clip_model_path,
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
            )
            return super()._initialize_llama(gguf_path, handler, force_params)
        else:
            return super()._initialize_llama(gguf_path, h, force_params)


__all__ = ["Qwen3VLPipeline"]
