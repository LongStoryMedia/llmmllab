"""
Base pipeline class for processing data in a structured manner.
"""

import json
import multiprocessing
import os
import re
import time

from typing import Optional, List, Any, Dict, Iterator, Type, Tuple, cast

from pydantic import BaseModel
import llama_cpp
from llama_cpp import llama_grammar
from llama_cpp.llama_types import CreateChatCompletionResponse
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall as LangChainToolCall,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools.base import BaseTool

from models import Model, ModelProfile, OptimalParameters
from models.default_configs import DEFAULT_GPU_CONFIG
from utils.logging import llmmllogger
from runner.utils.hardware_manager import hardware_manager
from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery


class BasePipeline(BaseChatModel):
    """
    Custom BaseChatModel implementation using llama-cpp-python directly.

    Features:
    - Direct Llama class instantiation from llama-cpp-python
    - Hardware optimization with GPU layers and context fallback
    - Grammar constraints support (GBNF/Pydantic)
    - Tool calling support through prompt formatting
    - Streaming and non-streaming chat completion
    """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "allow"

    model: Model
    profile: ModelProfile
    grammar: Optional[Type[BaseModel]]

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]],
        **kwargs,
    ):
        """Base LlamaCpp pipeline implementation.

        Experiment 4 adds optional single-GPU isolation to rule out mixed compute capability issues.
        Enable with environment variable:
            EXPERIMENT_SINGLE_GPU=true (forces CUDA_VISIBLE_DEVICES to EXPERIMENT_SINGLE_GPU_ID or '1')
            EXPERIMENT_SINGLE_GPU_ID=1 (defaults to 1 if unset)
        """

        # Pass the required fields to the parent constructor for Pydantic validation
        super().__init__(model=model, profile=profile, grammar=grammar, **kwargs)  # type: ignore
