"""
Base pipelines which implement LLM and BaseChatModel from langchain.
Creates a simple framework to run inference on models from various sources.
"""

from abc import ABC, abstractmethod
import asyncio
from typing import (
    List,
    Any,
    Union,
    AsyncGenerator,
    Iterator,
    Optional,
    TypeVar,
    Dict,
    Generic,
    AsyncIterator,
)
import logging
from datetime import datetime

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import (
    GenerationChunk,
    LLMResult,
    Generation,
    ChatResult,
    ChatGeneration,
    ChatGenerationChunk,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ChatMessage,
)

from models import (
    Message,
    MessageContent,
    MessageRole,
    MessageContentType,
    ModelProfile,
    Model,
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for return type
T = TypeVar("T", bound=Any)


class BasePipelineCore(ABC, Generic[T]):
    """
    Core pipeline functionality without LangChain inheritance conflicts.
    """

    model: Model
    profile: ModelProfile

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
    ):
        """Initialize the pipeline with model definition and parameters."""
        self.model = model
        self.profile = profile

    @abstractmethod
    def run(self, messages: List[Message], *args, **kwargs) -> AsyncGenerator[T, None]:
        """
        Process the messages and generate a response.

        Args:
            messages: List of messages to process
            params: Optional model parameters to override defaults

        Returns:
            AsyncIterator[T]: Async iterator yielding response chunks of type T
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    async def get(self, messages: List[Message], *args, **kwargs) -> T:
        """
        Get a complete response for the given messages.

        Args:
            messages: The list of messages to process
            params: Optional model parameters to override defaults

        Returns:
            T: The complete response (type depends on the model implementation)
        """
        chunks = []
        async for chunk in self.run(messages):
            chunks.append(chunk)

        # For string responses, concatenate chunks
        if chunks and isinstance(chunks[0], str):
            return "".join(chunks)  # type: ignore

        # For other types, return the last chunk or combine as appropriate
        return chunks[-1] if chunks else None  # type: ignore

    # ===== Helper Methods =====

    def create_user_message(self, text: str) -> Message:
        """Create a user message from text."""
        return Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text=text, url=None)],
            tool_calls=None,
            thinking=None,
            id=None,
            created_at=datetime.now(),
        )

    def convert_langchain_messages(self, messages: List[BaseMessage]) -> List[Message]:
        """Convert LangChain messages to native Message format."""
        native_messages = []

        for msg in messages:
            # Map LangChain message types to our roles
            role_mapping = {
                SystemMessage: MessageRole.SYSTEM,
                AIMessage: MessageRole.ASSISTANT,
                HumanMessage: MessageRole.USER,
                FunctionMessage: MessageRole.TOOL,
                ToolMessage: MessageRole.TOOL,
            }

            role = role_mapping.get(type(msg), MessageRole.USER)

            # Handle ChatMessage with role attribute
            if isinstance(msg, ChatMessage):
                role_str_mapping = {
                    "system": MessageRole.SYSTEM,
                    "assistant": MessageRole.ASSISTANT,
                    "user": MessageRole.USER,
                    "function": MessageRole.TOOL,
                    "tool": MessageRole.TOOL,
                }
                role = role_str_mapping.get(msg.role, MessageRole.USER)

            # Create content
            content = self._convert_message_content(msg.content)

            native_messages.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=None,
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
            )

        return native_messages

    def _convert_message_content(
        self, content: Union[str, List[Union[str, Dict]]]
    ) -> List[MessageContent]:
        """Convert message content to MessageContent objects."""
        if isinstance(content, str):
            return [
                MessageContent(type=MessageContentType.TEXT, text=content, url=None)
            ]

        if isinstance(content, list):
            message_content = []
            for item in content:
                if isinstance(item, str):
                    message_content.append(
                        MessageContent(
                            type=MessageContentType.TEXT, text=item, url=None
                        )
                    )
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        message_content.append(
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=item.get("text", ""),
                                url=None,
                            )
                        )
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        url = (
                            image_url.get("url", "")
                            if isinstance(image_url, dict)
                            else str(image_url)
                        )
                        message_content.append(
                            MessageContent(
                                type=MessageContentType.IMAGE, text=None, url=url
                            )
                        )
            return message_content

        return []

    def merge_params_with_stop(self, stop: Optional[List[str]]) -> ModelProfile:
        """Merge stop sequences with existing parameters."""
        if stop:
            self.profile.parameters.stop = stop

        return self.profile

    async def collect_response(
        self,
        messages: List[Message],
        run_manager: Optional[
            Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]
        ] = None,
    ) -> T:
        """Collect complete response from run method."""
        chunks = []

        async for chunk in self.run(messages):
            chunk_text = str(chunk)

            # Handle callback manager
            if run_manager:
                if hasattr(run_manager, "on_llm_new_token"):
                    if asyncio.iscoroutinefunction(run_manager.on_llm_new_token):
                        await run_manager.on_llm_new_token(chunk_text)
                    else:
                        run_manager.on_llm_new_token(chunk_text)

            chunks.append(chunk)

        # For string responses, concatenate
        if chunks and isinstance(chunks[0], str):
            return "".join(chunks)  # type: ignore

        # For other types, return last chunk or combine appropriately
        return chunks[-1] if chunks else None  # type: ignore


class PipelineChatModel(BaseChatModel):
    """
    LangChain BaseChatModel wrapper for pipelines.
    """

    def __init__(self, pipeline: BasePipelineCore):
        super().__init__()
        self.pipeline = pipeline

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a chat result from a list of messages."""

        # Convert LangChain messages to our format
        converted_messages = self.pipeline.convert_langchain_messages(messages)

        # Create parameters with stop sequences
        self.pipeline.merge_params_with_stop(stop)

        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        full_response = loop.run_until_complete(
            self.pipeline.collect_response(converted_messages, run_manager)
        )

        # Create AI message from response
        ai_message = AIMessage(content=str(full_response))

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"model_name": self.pipeline.model.name},
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat model on the given messages."""

        # Convert messages and create parameters
        converted_messages = self.pipeline.convert_langchain_messages(messages)
        self.pipeline.merge_params_with_stop(stop)

        # Create async generator and convert to sync iterator
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _async_stream():
            async for chunk in self.pipeline.run(converted_messages):
                chunk_text = str(chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk_text)

                ai_chunk = AIMessageChunk(content=chunk_text)
                yield ChatGenerationChunk(message=ai_chunk)

        # Convert async generator to sync iterator
        async_gen = _async_stream()
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Async implementation of _generate."""
        converted_messages = self.pipeline.convert_langchain_messages(messages)
        self.pipeline.merge_params_with_stop(stop)

        full_response = await self.pipeline.collect_response(
            converted_messages, run_manager
        )

        ai_message = AIMessage(content=str(full_response))
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"model_name": self.pipeline.model.name},
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        """Async stream implementation."""
        converted_messages = self.pipeline.convert_langchain_messages(messages)
        self.pipeline.merge_params_with_stop(stop)

        async for chunk in self.pipeline.run(converted_messages):
            chunk_text = str(chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk_text)

            ai_chunk = AIMessageChunk(content=chunk_text)
            yield ChatGenerationChunk(message=ai_chunk)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for logging purposes."""
        return f"chat-pipeline-{self.pipeline.model.name}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.pipeline.model.name,
            "model_id": self.pipeline.model.id,
            "model_task": (
                self.pipeline.model.task.value
                if self.pipeline.model.task
                else "unknown"
            ),
        }


class PipelineLLM(LLM):
    """
    LangChain LLM wrapper for pipelines.
    """

    def __init__(self, pipeline: BasePipelineCore):
        super().__init__()
        self.pipeline = pipeline

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Process a single text prompt and return a response."""

        messages = [self.pipeline.create_user_message(prompt)]
        self.pipeline.merge_params_with_stop(stop)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            self.pipeline.collect_response(messages, run_manager)
        )

        return str(response)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt."""

        messages = [self.pipeline.create_user_message(prompt)]
        self.pipeline.merge_params_with_stop(stop)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _async_stream():
            async for chunk in self.pipeline.run(messages):
                chunk_text = str(chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk_text)
                yield GenerationChunk(text=chunk_text)

        # Convert async generator to sync iterator
        async_gen = _async_stream()
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> LLMResult:
        """Async implementation for multiple prompts."""
        generations = []

        for prompt in prompts:
            messages = [self.pipeline.create_user_message(prompt)]
            self.pipeline.merge_params_with_stop(stop)

            response = await self.pipeline.collect_response(messages, run_manager)
            generations.append([Generation(text=str(response))])

        return LLMResult(generations=generations)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> AsyncGenerator[GenerationChunk, None]:
        """Async stream for single prompt."""
        messages = [self.pipeline.create_user_message(prompt)]
        self.pipeline.merge_params_with_stop(stop)

        async for chunk in self.pipeline.run(messages):
            chunk_text = str(chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk_text)
            yield GenerationChunk(text=chunk_text)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for logging purposes."""
        return f"text-pipeline-{self.pipeline.model.name}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.pipeline.model.name,
            "model_id": self.pipeline.model.id,
            "model_task": (
                self.pipeline.model.task.value
                if self.pipeline.model.task
                else "unknown"
            ),
        }


class BasePipelineDual(BasePipelineCore[T]):
    """
    Base pipeline class with dual LangChain compatibility.

    This class provides both LLM and BaseChatModel interfaces through composition
    rather than multiple inheritance to avoid type conflicts.

    Usage:
        # Text generation model
        class MyTextModel(BasePipelineDual[str]):
            async def run(self, messages: List[Message], params: Optional[ModelParameters] = None) -> AsyncIterator[str]:
                yield "Hello world!"

        # Use as ChatModel
        pipeline = MyTextModel(model_def, params)
        chat_model = pipeline.as_chat_model()

        # Use as LLM
        llm = pipeline.as_llm()
    """

    def as_chat_model(self) -> PipelineChatModel:
        """Get a LangChain BaseChatModel interface for this pipeline."""
        return PipelineChatModel(self)

    def as_llm(self) -> PipelineLLM:
        """Get a LangChain LLM interface for this pipeline."""
        return PipelineLLM(self)


# Convenience base classes for common use cases


class TextPipeline(BasePipelineDual[str]):
    """Base class for text generation pipelines."""


class EmbeddingPipeline(BasePipelineDual[List[List[float]]]):
    """Base class for embedding pipelines."""
