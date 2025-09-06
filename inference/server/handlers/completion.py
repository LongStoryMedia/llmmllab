"""
Enhanced chat completion handler with proper LangGraph integration and resource management.
"""

import asyncio
from typing import AsyncIterator, List, Optional
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks

from langchain_core.tools import BaseTool

from models import (
    ChatResponse,
    Message,
    MessageRole,
    MessageContentType,
    MessageContent,
)
from utils.serialization import serialize_to_json
from utils.message import extract_message_text
from utils.response import create_error_chunk, create_streaming_chunk

from runner.pipeline_factory import pipeline_factory
from runner.pipelines.run import stream_pipeline

from server.config import logger
from server.db import storage
from server.tools.integration import (
    get_tools,
)
from ..services.context import ConversationContext


class ResourceContext:
    """Manage resources with cleanup support."""

    def __init__(self):
        self._resources = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        for name, res in list(self._resources.items()):
            try:
                if hasattr(res, "cleanup"):
                    fn = getattr(res, "cleanup")
                    if asyncio.iscoroutinefunction(fn):
                        await fn()
                    else:
                        fn()
                elif hasattr(res, "close"):
                    fn = getattr(res, "close")
                    if asyncio.iscoroutinefunction(fn):
                        await fn()
                    else:
                        fn()
            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}")
        self._resources.clear()

    def add(self, name: str, resource) -> None:
        self._resources[name] = resource


class CompletionHandler:
    """Chat completion orchestrator."""

    @asynccontextmanager
    async def _resources(self) -> AsyncIterator[ResourceContext]:
        async with ResourceContext() as rc:
            yield rc

    async def handle_completion(
        self,
        conversation_ctx: ConversationContext,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> AsyncIterator[str]:
        """
        Handle chat completion requests.
        """
        async with self._resources() as rc:
            try:

                # 2) Acquire tools, streaming status updates
                tools: List[BaseTool] = []
                try:
                    async for item in get_tools(conversation_ctx):
                        if isinstance(item, str):
                            yield serialize_to_json(
                                create_streaming_chunk(item, done=False)
                            )
                        else:
                            tools = item
                            break
                except Exception as e:
                    logger.warning(f"Tool acquisition failed: {e}")
                    tools = []

                # 3) Resolve model profile and pipeline
                mp = await storage.get_service(
                    storage.model_profile
                ).get_model_profile_by_id(
                    conversation_ctx.user_config.model_profiles.primary_profile_id,
                    conversation_ctx.user_config.user_id,
                )
                assert mp
                full_response = ""

                with pipeline_factory.pipeline(mp, ChatResponse) as pipeline:
                    rc.add("pipeline", pipeline)

                    # 4) Stream execution
                    logger.info(
                        f"Starting pipeline execution with {len(conversation_ctx.messages)} messages and {len(tools)} tools"
                    )
                    chunk_count = 0
                    async for chunk in stream_pipeline(
                        conversation_ctx.messages, pipeline, tools
                    ):
                        chunk_count += 1
                        # Extract text content for accumulation
                        chunk_text = (
                            extract_message_text(chunk.message) if chunk.message else ""
                        )
                        full_response += chunk_text

                        # For streaming, only send serialized ChatResponse with text content
                        # Status messages and tool info come through as special chunks
                        yield serialize_to_json(chunk)

                    logger.info(
                        f"Pipeline execution completed with {chunk_count} chunks, response length: {len(full_response)}"
                    )

                # Store response in background
                if background_tasks and full_response.strip():
                    assistant_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=full_response
                            )
                        ],
                        conversation_id=conversation_ctx.conversation.id,
                    )
                    background_tasks.add_task(
                        conversation_ctx.add_message, assistant_message
                    )

            except Exception as e:
                logger.error(f"Completion error: {e}", exc_info=True)
                yield serialize_to_json(create_error_chunk(str(e)))


# Global handler instance
chat_handler = CompletionHandler()


async def agent_chat_completion(
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterator[str]:
    """Entry point used by FastAPI route to stream chat completions."""
    async for chunk in chat_handler.handle_completion(
        conversation_ctx, background_tasks
    ):
        yield chunk
