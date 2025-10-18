"""
Simplified OpenAI GPT OSS pipeline implementation - pure LLM interface.
Just calls LLM directly with messages, configuration, and hardware management.
"""

from typing import Dict, Any, Optional, Type
from pydantic import BaseModel

from models import (
    Model,
    ModelProfile,
)
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class OpenAIGptOssPipeline(BaseLlamaCppPipeline):
    """
    Simplified OpenAI GPT OSS pipeline - pure LLM interface.

    Just calls LLM directly with messages, configuration, and hardware management.
    No orchestration, no graphs - exactly what composer needs.
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
        return "openai-gpt-oss-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "openai-gpt-oss",
                "chat_format": "openai-gpt",
            }
        )
        return base_params
