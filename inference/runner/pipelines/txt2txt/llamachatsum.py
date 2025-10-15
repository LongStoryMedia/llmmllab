"""
Simplified Llama Chat Summary pipeline - pure LLM interface, no orchestration.
Replaced 641 lines of complex LangGraph orchestration with direct LLM calls.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator, Dict, Any

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
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
from utils.message import extract_message_text


class LlamaChatSummPipe(BaseLlamaCppPipeline):
    """
    Simplified Llama Chat Summary pipeline - direct LLM calls for summarization.

    Features:
    - Direct LlamaCpp initialization for summarization models
    - Clean prompt formatting optimized for summary generation
    - Automatic text preprocessing for better summaries
    - Hardware optimization for Llama 3.2 3B models
    """

    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(model, profile, **kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)

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
