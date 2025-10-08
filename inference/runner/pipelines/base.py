"""
Simplified pipeline architecture without LangGraph - pure LLM interface.
Clean, focused pipeline abstractions for direct model interactions.
"""

from abc import ABC, abstractmethod
import logging
from typing import (
    List,
    Any,
    Optional,
    Union,
    Type,
    TypeVar,
    Generic,
)
from pathlib import Path

from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from models import (
    Message,
    ModelProfile,
    Model,
    ChatResponse,
)

Embeddings = List[List[float]]
PipeReturn = Union[str, Embeddings, ChatResponse]
GrammarInput = Union[str, Path, Type[BaseModel], None]

logger = logging.getLogger(__name__)


# Type variable for return types
T = TypeVar("T")
PipeReturn = TypeVar("PipeReturn")


class SimplePipelineCore(ABC, Generic[T]):
    """
    Simplified pipeline core - pure LLM interface without orchestration.

    Responsibilities:
    - Direct model interaction
    - Grammar-constrained generation
    - Streaming support
    - Type safety

    Does NOT handle:
    - Graph construction
    - Complex orchestration
    - Multi-step workflows
    - State management beyond single calls
    """

    # Subclasses should override to restrict supported return types
    allowed_return_types: tuple[type, ...] = (str, ChatResponse, list)
    default_return_type: Optional[type] = None

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
    ):
        """Initialize the pipeline with model definition and parameters."""
        self.model = model
        self.profile = profile
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Resolve expected type and validate against allowed types
        self.expected_return_type: Optional[type] = (
            expected_return_type
            if expected_return_type is not None
            else self.default_return_type
        )

        if self.expected_return_type is not None and not self._allows_return_type(
            self.expected_return_type
        ):
            allowed = ", ".join(t.__name__ for t in self.allowed_return_types)
            raise TypeError(
                f"{self.__class__.__name__} does not support return type "
                f"{getattr(self.expected_return_type, '__name__', str(self.expected_return_type))}. "
                f"Allowed: {allowed}"
            )

    def _allows_return_type(self, t: type) -> bool:
        """Check if pipeline supports this return type."""
        return t in self.allowed_return_types

    def allows_return_type(self, t: type) -> bool:
        """Public check for whether this pipeline supports a return type."""
        return self._allows_return_type(t)

    def validate_return_value(self, value: Any) -> None:
        """Validate that a produced value matches the configured expected return type."""

        def _is_embeddings(v: Any) -> bool:
            # A very light structural check: list of lists of floats (or empty)
            if not isinstance(v, list):
                return False
            if not v:
                return True
            return isinstance(v[0], list)

        expected = self.expected_return_type
        if expected is ChatResponse and not isinstance(value, ChatResponse):
            raise TypeError(
                f"Expected ChatResponse from {self.__class__.__name__}, got {type(value).__name__}"
            )
        if expected is str and not isinstance(value, str):
            raise TypeError(
                f"Expected str from {self.__class__.__name__}, got {type(value).__name__}"
            )
        if expected is list and not _is_embeddings(value):
            raise TypeError(
                f"Expected embeddings (List[List[float]]) from {self.__class__.__name__}, got {type(value).__name__}"
            )

        if expected is None:
            # No explicit expectation; permit anything within allowed types
            if isinstance(value, ChatResponse) and not self._allows_return_type(
                ChatResponse
            ):
                raise TypeError(
                    f"{self.__class__.__name__} is not configured to return ChatResponse"
                )
            if isinstance(value, str) and not self._allows_return_type(str):
                raise TypeError(
                    f"{self.__class__.__name__} is not configured to return str"
                )
            if _is_embeddings(value) and not self._allows_return_type(list):
                raise TypeError(
                    f"{self.__class__.__name__} is not configured to return embeddings"
                )

    @abstractmethod
    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> Any:
        """
        Direct pipeline invocation - single call interface.

        Args:
            messages: Input messages
            tools: Optional tools (implementation-specific handling)
            grammar: Optional grammar constraint
            **kwargs: Additional pipeline-specific parameters

        Returns:
            Pipeline result based on expected_return_type
        """

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> Any:
        """
        Stream pipeline execution (optional, not all pipelines support streaming).

        Args:
            messages: Input messages
            tools: Optional tools
            grammar: Optional grammar constraint
            **kwargs: Additional parameters

        Returns:
            AsyncIterator of streaming chunks
        """
        # Default implementation falls back to invoke
        result = await self.invoke(messages, tools, grammar, **kwargs)
        yield result

    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            # Subclasses should override to add specific cleanup
            self._cleanup_resources()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _cleanup_resources(self) -> None:
        """Subclass-specific cleanup. Override as needed."""
        # Default does nothing

    def get_common_args(self) -> dict:
        """Return common arguments for this pipeline."""
        return {
            "model": self.model.model_dump() if self.model else None,
            "profile": self.profile.model_dump() if self.profile else None,
        }


class SimpleChatPipeline(SimplePipelineCore):
    """
    Simple chat pipeline - direct chat model interface.
    """

    # Chat pipelines return ChatResponse
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=ChatResponse)
        self.llm: Optional[BaseChatModel] = None

    @abstractmethod
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the underlying LLM. Must be implemented by subclasses."""

    def get_llm(self) -> BaseChatModel:
        """Get or initialize the LLM."""
        if not self.llm:
            self.llm = self._initialize_llm()
        return self.llm

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """Direct chat invocation."""
        # Convert messages to LangChain format and invoke
        # Implementation varies by pipeline type
        raise NotImplementedError("Subclasses must implement invoke")

    def _cleanup_resources(self) -> None:
        """Clean up chat pipeline specific resources."""
        try:
            if self.llm:
                if hasattr(self.llm, "cleanup"):
                    self.llm.cleanup()  # type: ignore
                self.llm = None
        except Exception as e:
            self.logger.error(f"Error cleaning up chat pipeline resources: {e}")


class SimpleEmbeddingPipeline(SimplePipelineCore):
    """Simple embedding pipeline - direct embedding interface."""

    # Embedding pipelines return embeddings (list[list[float]])
    allowed_return_types: tuple[type, ...] = (list,)
    default_return_type: Optional[type] = list

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=list)

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for input messages."""
        # Implementation varies by embedding model
        raise NotImplementedError("Subclasses must implement invoke")

    # Many callers (e.g., embed_pipeline helper) expect embedding pipelines to
    # expose a process_messages method mirroring older orchestration-era APIs.
    # Provide a default passthrough to maintain compatibility.
    async def process_messages(  # type: ignore[override]
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> List[List[float]]:
        return await self.invoke(messages, tools=tools, grammar=grammar, **kwargs)


class SimpleTextPipeline(SimplePipelineCore):
    """Base class for simple text completion pipelines."""

    allowed_return_types = (str,)
    default_return_type = str

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=str)

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> str:
        """Generate text response."""
        # Implementation varies by text model
        raise NotImplementedError("Subclasses must implement invoke")


# Backwards compatibility aliases for old pipeline_factory
BasePipelineCore = SimplePipelineCore
EmbeddingPipeline = SimpleEmbeddingPipeline
PipeReturn = Any
