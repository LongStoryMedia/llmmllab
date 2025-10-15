"""
Simplified OpenAI GPT OSS pipeline implementation - pure LLM interface.
Just calls LLM directly with messages, configuration, and hardware management.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator, Dict, Any, ClassVar

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.language_models import BaseChatModel

try:
    from langchain_community.chat_models.llamacpp import ChatLlamaCpp
except ImportError:
    ChatLlamaCpp = None
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from runner.pipelines.base import GrammarInput
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class OpenAIGptOssPipeline(BaseLlamaCppPipeline):
    """
    Simplified OpenAI GPT OSS pipeline - pure LLM interface.

    Just calls LLM directly with messages, configuration, and hardware management.
    No orchestration, no graphs - exactly what composer needs.
    """

    # Override to allow ChatResponse return type
    allowed_return_types: ClassVar = (ChatResponse,)
    default_return_type: ClassVar = ChatResponse

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        **kwargs,
    ):
        super().__init__(model, profile, **kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)

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
