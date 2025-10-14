"""
Simplified pipeline execution with metadata tracking and streaming support.
Designed for the new simplified architecture where orchestration is handled by Composer.
"""

import logging
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, List, AsyncIterator, Union, Type
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from langchain_core.tools import BaseTool

from models import (
    ChatResponse,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    ModelProvider,
)
from utils.message import extract_message_text
from utils.response import create_streaming_chunk

from .base import SimplePipelineCore, SimpleEmbeddingPipeline


# Type aliases for better readability
MessageInput = Union[str, Message, List[Union[str, Message]], List[Message], List[str]]
GrammarInput = Union[str, Path, Type[BaseModel], None]


class PipelineExecutionMetadata:
    """Tracks execution metadata for pipeline runs."""

    def __init__(
        self, pipeline: SimplePipelineCore, execution_id: Optional[str] = None
    ):
        self.execution_id = execution_id or str(uuid.uuid4())[:8]
        self.pipeline_name = type(pipeline).__name__
        self.model_name = (
            getattr(pipeline.model, "name", "unknown")
            if hasattr(pipeline, "model")
            else "unknown"
        )
        self.model_id = (
            getattr(pipeline.model, "id", "unknown")
            if hasattr(pipeline, "model")
            else "unknown"
        )
        self.provider = (
            getattr(pipeline.model, "provider", ModelProvider.OTHER)
            if hasattr(pipeline, "model")
            else ModelProvider.OTHER
        )
        self.is_cached = self._determine_if_cached(pipeline)
        self.expected_return_type = getattr(pipeline, "expected_return_type", None)
        self.start_time = datetime.now(timezone.utc)
        self.token_count = 0

    def _determine_if_cached(self, pipeline: SimplePipelineCore) -> bool:
        """Determine if this pipeline instance is from cache."""
        # Local providers use caching
        if hasattr(pipeline, "model") and hasattr(pipeline.model, "provider"):
            return pipeline.model.provider in {
                ModelProvider.LLAMA_CPP,
                ModelProvider.STABLE_DIFFUSION_CPP,
            }
        return False

    def log_start(self, logger: logging.Logger) -> None:
        """Log pipeline execution start with metadata."""
        cache_status = "cached" if self.is_cached else "transient"
        logger.info(
            f"[{self.execution_id}] Starting {self.pipeline_name} execution",
            extra={
                "execution_id": self.execution_id,
                "pipeline": self.pipeline_name,
                "model": self.model_name,
                "model_id": self.model_id,
                "provider": (
                    self.provider.value
                    if hasattr(self.provider, "value")
                    else str(self.provider)
                ),
                "cache_status": cache_status,
                "return_type": (
                    self.expected_return_type.__name__
                    if self.expected_return_type
                    else None
                ),
            },
        )

    def log_completion(self, logger: logging.Logger, success: bool = True) -> None:
        """Log pipeline execution completion."""
        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        status = "completed" if success else "failed"
        logger.info(
            f"[{self.execution_id}] Pipeline {status} in {duration:.2f}s",
            extra={
                "execution_id": self.execution_id,
                "pipeline": self.pipeline_name,
                "duration_seconds": duration,
                "token_count": self.token_count,
                "status": status,
            },
        )


def _normalize_message_input(
    input_data: MessageInput, role: MessageRole = MessageRole.USER
) -> List[Message]:
    """
    Normalize various input types to a List[Message].

    Args:
        input_data: Can be str, Message, List[str | Message]

    Returns:
        List[Message]: Normalized message list
    """
    if isinstance(input_data, str):
        # Single string -> single Message
        return [
            Message(
                role=role,
                content=[MessageContent(type=MessageContentType.TEXT, text=input_data)],
            )
        ]
    elif isinstance(input_data, Message):
        # Single Message -> list with one Message
        return [input_data]
    elif isinstance(input_data, list):
        if not input_data:
            return []

        # Coerce each item in the list to a Message object
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append(
                    Message(
                        role=role,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=item)
                        ],
                    )
                )
            elif isinstance(item, Message):
                messages.append(item)
            else:
                # Convert other types to string, then to Message
                messages.append(
                    Message(
                        role=role,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=str(item))
                        ],
                    )
                )
        return messages


async def stream_pipeline(
    messages: MessageInput,
    pipeline: SimplePipelineCore,
    tools: Optional[List[BaseTool]] = None,
    grammar: Optional[GrammarInput] = None,
) -> AsyncIterator[ChatResponse]:
    """
    Stream pipeline execution with metadata tracking.
    Simplified implementation - complex orchestration handled by Composer.

    Args:
        messages: Input messages in various formats
        pipeline: Pipeline instance to execute
        tools: Optional tools for the pipeline
        grammar: Optional grammar constraint
    """
    logger = logging.getLogger(__name__)
    metadata = PipelineExecutionMetadata(pipeline)
    metadata.log_start(logger)

    try:
        # Normalize input to List[Message]
        normalized_messages = _normalize_message_input(messages)

        if not normalized_messages:
            yield create_streaming_chunk("No messages provided", done=True)
            metadata.log_completion(logger, success=False)
            return

        # Check if pipeline supports streaming
        if hasattr(pipeline, "stream"):
            try:
                async for chunk in pipeline.stream(
                    normalized_messages, tools=tools, grammar=grammar
                ):
                    if isinstance(chunk, ChatResponse):
                        # Track tokens if available
                        if chunk.message:
                            text = extract_message_text(chunk.message)
                            if text:
                                metadata.token_count += len(text.split())
                        yield chunk
                    else:
                        # Convert non-ChatResponse to streaming chunk
                        yield create_streaming_chunk(str(chunk))

                metadata.log_completion(logger, success=True)
                return

            except Exception as e:
                logger.error(f"Error in pipeline streaming: {e}")
                yield create_streaming_chunk(f"Streaming error: {str(e)}", done=True)
                metadata.log_completion(logger, success=False)
                return

        # Fallback to invoke for pipelines that don't support streaming
        try:
            result = await pipeline.invoke(
                normalized_messages, tools=tools, grammar=grammar
            )

            if isinstance(result, ChatResponse):
                # Track token count
                if result.message:
                    text = extract_message_text(result.message)
                    if text:
                        metadata.token_count = len(text.split())
                yield result
            else:
                yield create_streaming_chunk(str(result), done=True)

            metadata.log_completion(logger, success=True)

        except Exception as e:
            logger.error(f"Error in pipeline invoke: {e}")
            yield create_streaming_chunk(f"Pipeline error: {str(e)}", done=True)
            metadata.log_completion(logger, success=False)

    except Exception as e:
        logger.error(f"Pipeline streaming error: {e}", exc_info=True)
        yield create_streaming_chunk(f"Pipeline error: {str(e)}", done=True)
        metadata.log_completion(logger, success=False)


async def run_pipeline(
    messages: MessageInput,
    pipeline: SimplePipelineCore,
    tools: Optional[List[BaseTool]] = None,
    grammar: Optional[GrammarInput] = None,
) -> ChatResponse:
    """
    Get a complete response from the pipeline with metadata tracking.
    Simplified implementation for the new architecture.

    Args:
        messages: Input messages in various formats
        pipeline: Pipeline instance to execute
        tools: Optional tools for the pipeline
        grammar: Optional grammar constraint
    """
    logger = logging.getLogger(__name__)
    metadata = PipelineExecutionMetadata(pipeline)
    metadata.log_start(logger)

    try:
        # Normalize input to List[Message]
        normalized_messages = _normalize_message_input(messages)

        if not normalized_messages:
            metadata.log_completion(logger, success=False)
            return ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text="No messages provided"
                        )
                    ],
                ),
                created_at=datetime.now(timezone.utc),
                finish_reason="error",
            )

        # Direct pipeline invocation
        result = await pipeline.invoke(
            normalized_messages, tools=tools, grammar=grammar
        )

        # Handle different return types based on pipeline
        if isinstance(result, ChatResponse):
            # Track token count
            if result.message:
                text = extract_message_text(result.message)
                if text:
                    metadata.token_count = len(text.split())
            metadata.log_completion(logger, success=True)
            return result

        elif isinstance(result, str):
            # Convert string result to ChatResponse
            metadata.token_count = len(result.split())
            metadata.log_completion(logger, success=True)
            return ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[MessageContent(type=MessageContentType.TEXT, text=result)],
                ),
                created_at=datetime.now(timezone.utc),
                finish_reason="stop",
            )
        else:
            # Handle other return types (embeddings, etc.)
            metadata.log_completion(logger, success=True)
            return ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=str(result))
                    ],
                ),
                created_at=datetime.now(timezone.utc),
                finish_reason="stop",
            )

    except Exception as e:
        logger.error(f"Error in run_pipeline: {e}")
        metadata.log_completion(logger, success=False)
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"Error processing request: {str(e)}",
                    )
                ],
            ),
            created_at=datetime.now(timezone.utc),
            finish_reason="error",
        )


async def embed_pipeline(
    messages: MessageInput,
    pipeline: SimpleEmbeddingPipeline,
    grammar: Optional[GrammarInput] = None,
) -> List[List[float]]:
    """
    Get embeddings from the pipeline with metadata tracking.
    Simplified interface for embedding operations.

    Args:
        messages: Input messages in various formats
        pipeline: Embedding pipeline instance
        grammar: Optional grammar constraint (typically not used for embeddings)
    """
    logger = logging.getLogger(__name__)
    metadata = PipelineExecutionMetadata(pipeline)
    metadata.log_start(logger)

    try:
        # Normalize input to List[Message]
        normalized_messages = _normalize_message_input(messages)

        if not normalized_messages:
            logger.warning("No messages provided to embed_pipeline")
            metadata.log_completion(logger, success=False)
            return []

        # Direct pipeline invocation
        result = await pipeline.invoke(normalized_messages, grammar=grammar)

        # Validate result format
        if isinstance(result, list) and (
            not result or all(isinstance(item, list) for item in result)
        ):
            metadata.log_completion(logger, success=True)
            return result
        else:
            logger.warning(f"Unexpected embedding result type: {type(result)}")
            metadata.log_completion(logger, success=False)
            return []

    except Exception as e:
        logger.error(f"Error in embed_pipeline: {e}")
        metadata.log_completion(logger, success=False)
        return []
