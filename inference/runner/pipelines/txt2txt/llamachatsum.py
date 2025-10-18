"""
Simplified Llama Chat Summary pipeline - pure LLM interface, no orchestration.
Replaced 641 lines of complex LangGraph orchestration with direct LLM calls.
"""

from typing import Dict, Any, Optional, Type

from pydantic import BaseModel
from models import (
    Model,
    ModelProfile,
)
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class LlamaChatSummPipe(BaseLlamaCppPipeline):
    """
    Simplified Llama Chat Summary pipeline - direct LLM calls for summarization.

    Features:
    - Direct LlamaCpp initialization for summarization models
    - Clean prompt formatting optimized for summary generation
    - Automatic text preprocessing for better summaries
    - Hardware optimization for Llama 3.2 3B models
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
        return "llama-chat-summary-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "llama-chat-summary",
                "task": "summarization",
            }
        )
        return base_params
