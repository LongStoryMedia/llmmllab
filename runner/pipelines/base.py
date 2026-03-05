"""
Base pipeline class for processing data in a structured manner.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Iterator, Optional, Type
from pydantic import BaseModel

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.language_models import BaseChatModel


from runner.models import Model, ModelProfile


# Enable HTTP logging for debugging
if os.getenv("LOG_LEVEL", "").lower() == "trace":
    logging.getLogger("openai").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
else:
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class BasePipeline(BaseChatModel, ABC):
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
        metadata: Optional[dict] = None,
    ):
        """Base LlamaCpp pipeline implementation.

        Experiment 4 adds optional single-GPU isolation to rule out mixed compute capability issues.
        Enable with environment variable:
            EXPERIMENT_SINGLE_GPU=true (forces CUDA_VISIBLE_DEVICES to EXPERIMENT_SINGLE_GPU_ID or '1')
            EXPERIMENT_SINGLE_GPU_ID=1 (defaults to 1 if unset)
        """

        # Pass the required fields to the parent constructor for Pydantic validation
        super().__init__(
            name=model.name,
            verbose=os.getenv("LOG_LEVEL", "warning").lower() == "trace",
            output_version="v1",
            tags=[
                model.task.value,
                model.provider.value,
            ],
            metadata={
                "model_id": model.id,
                "profile_id": profile.id,
                "grammar": grammar.__name__ if grammar else "None",
                **(metadata or {}),
            },
            model=model,  # type: ignore
            profile=profile,  # type: ignore
            grammar=grammar,  # type: ignore
        )

    @abstractmethod
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs
    ) -> ChatResult:
        """Generate chat completions given input messages."""

    @abstractmethod
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions given input messages."""

    @abstractmethod
    def bind_tools(self, tools: list[BaseModel], **kwargs) -> BaseChatModel:
        """Bind tools to the pipeline for tool calling support."""

    @abstractmethod
    def bind_metadata(self, metadata: dict) -> BaseChatModel:
        """Bind additional metadata to the pipeline."""

    @abstractmethod
    def shutdown(self):
        """Shutdown the pipeline and release resources."""

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        self.shutdown()
