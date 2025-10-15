"""
Simplified pipeline execution with metadata tracking and streaming support.
Designed for the new simplified architecture where orchestration is handled by Composer.
"""

import logging
from typing import Optional, List, AsyncIterator, Union, Type
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from langchain_core.tools import BaseTool

from models import (
    ChatResponse,
    LangChainMessage,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)
from utils.response import create_error_response
from utils.logging import llmmllogger

from .base import SimplePipelineCore, SimpleEmbeddingPipeline


# Type aliases for better readability
MessageInput = Union[str, Message, List[Union[str, Message]], List[Message], List[str]]
GrammarInput = Union[str, Path, Type[BaseModel], None]


async def stream_pipeline(
    messages: List[LangChainMessage],
    pipeline: SimplePipelineCore,
    tools: Optional[List[BaseTool]] = None,
    grammar: Optional[GrammarInput] = None,
    metadata: Optional[dict] = None,
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
    logger = llmmllogger.logger.bind(module=__name__)

    try:
        async for chunk in pipeline.stream(messages, tools=tools, grammar=grammar):
            yield chunk

        return

    except Exception as e:
        logger.error(f"Error in pipeline streaming: {e}")
        yield create_error_response(f"Streaming error: {str(e)}", done=True)
        return


async def run_pipeline(
    messages: List[LangChainMessage],
    pipeline: SimplePipelineCore,
    tools: Optional[List[BaseTool]] = None,
    grammar: Optional[GrammarInput] = None,
    metadata: Optional[dict] = None,
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
    logger = llmmllogger.logger.bind(module=__name__)

    try:
        # Direct pipeline invocation
        result = await pipeline.invoke(
            messages,
            tools=tools,
            grammar=grammar,
            metadata=metadata,
        )

        # Handle different return types based on pipeline
        if isinstance(result, ChatResponse):
            return result

        elif isinstance(result, str):
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
