"""
Enhanced chat completion handler with proper LangGraph integration and resource management.
"""

import asyncio
import time
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
    StandardToolProvider,
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

    def __init__(self):
        self._model_profile_cache = {}
        self._cache_ttl = 300  # 5 minutes

    @asynccontextmanager
    async def _resources(self) -> AsyncIterator[ResourceContext]:
        async with ResourceContext() as rc:
            yield rc

    async def _get_model_profile(self, conversation_ctx: ConversationContext):
        """Get model profile with caching."""
        user_id = conversation_ctx.user_config.user_id
        primary_id = conversation_ctx.user_config.model_profiles.primary_profile_id

        # Try cache first
        cache_key = f"{user_id}:{primary_id}"
        if cache_key in self._model_profile_cache:
            cached_mp, timestamp = self._model_profile_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_mp

        # Get primary profile
        mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
            primary_id, user_id
        )

        if mp:
            self._model_profile_cache[cache_key] = (mp, time.time())
            return mp

        raise ValueError(f"Model profile {primary_id} not found for user {user_id}")

    def _should_skip_dynamic_tools(self, conversation_ctx: ConversationContext) -> bool:
        """Determine if we can skip expensive dynamic tool generation."""
        if not conversation_ctx.current_user_message:
            return True

        message_text = extract_message_text(conversation_ctx.current_user_message)
        if not message_text:
            return True

        # Skip for simple conversational queries
        simple_patterns = [
            "hello",
            "hi",
            "how are you",
            "what is",
            "tell me about",
            "explain",
            "describe",
            "who is",
            "where is",
            "when is",
        ]

        message_lower = message_text.lower()
        if any(pattern in message_lower for pattern in simple_patterns):
            if len(message_text.split()) < 10:  # Short simple queries
                return True

        return False

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

                # 2) Acquire tools efficiently (non-blocking)
                tools: List[BaseTool] = []
                try:
                    # Skip expensive dynamic tool generation for simple queries
                    if self._should_skip_dynamic_tools(conversation_ctx):
                        tools = StandardToolProvider.get_standard_tools(
                            conversation_ctx
                        )
                        yield serialize_to_json(
                            create_streaming_chunk("Using standard tools", done=False)
                        )
                    else:
                        # Use much longer timeout for tool acquisition in research/testing
                        tool_timeout = 120.0  # Reduced to 2 minutes for faster timeout
                        try:
                            tool_gen = get_tools(conversation_ctx)
                            start_time = asyncio.get_event_loop().time()
                            chunk_count = 0  # Track chunks to detect stuck generation

                            async for item in tool_gen:
                                # Check timeout manually since asyncio.wait_for doesn't work with async generators
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time

                                if elapsed > tool_timeout:
                                    raise asyncio.TimeoutError(
                                        f"Tool acquisition timed out after {tool_timeout}s"
                                    )

                                # Additional check: if we've been streaming for a while without completing
                                chunk_count += 1
                                if (
                                    chunk_count > 100 and elapsed > 60.0
                                ):  # Many chunks after 1 minute
                                    raise asyncio.TimeoutError(
                                        f"Tool generation appears stuck after {elapsed:.1f}s ({chunk_count} chunks)"
                                    )

                                if isinstance(item, str):
                                    yield serialize_to_json(
                                        create_streaming_chunk(item, done=False)
                                    )
                                else:
                                    tools = item
                                    break
                        except asyncio.TimeoutError as e:
                            logger.warning(
                                f"Tool acquisition timed out: {e}, using standard tools"
                            )

                            # Force cleanup after tool timeout to free memory
                            try:
                                evicted = pipeline_factory.force_memory_cleanup()
                                logger.info(
                                    f"Force cleanup after tool timeout evicted {evicted} pipelines"
                                )
                            except Exception as cleanup_error:
                                logger.warning(
                                    f"Post-timeout cleanup failed: {cleanup_error}"
                                )

                            tools = StandardToolProvider.get_standard_tools(
                                conversation_ctx
                            )
                            yield serialize_to_json(
                                create_streaming_chunk(
                                    "Tool generation timed out, using standard tools",
                                    done=False,
                                )
                            )
                except Exception as e:
                    logger.warning(
                        f"Tool acquisition failed: {e}, using standard tools"
                    )
                    tools = StandardToolProvider.get_standard_tools(conversation_ctx)
                    yield serialize_to_json(
                        create_streaming_chunk(
                            "Tool generation failed, using standard tools", done=False
                        )
                    )

                # 3) Resolve model profile
                mp = await self._get_model_profile(conversation_ctx)
                full_response = ""

                try:
                    # Use HIGH priority for main chat completion pipelines
                    from runner.pipeline_factory import PipelinePriority

                    # Get user's circuit breaker config from conversation context
                    user_circuit_breaker = conversation_ctx.user_config.circuit_breaker

                    with pipeline_factory.pipeline(
                        mp, ChatResponse, PipelinePriority.HIGH, user_circuit_breaker
                    ) as pipeline:
                        rc.add("pipeline", pipeline)

                        # 4) Stream execution - use enriched messages with summaries
                        enriched_messages = conversation_ctx.get_enriched_messages(
                            mp, tools
                        )
                        logger.info(
                            f"Starting pipeline execution with {len(enriched_messages)} messages and {len(tools)} tools"
                        )

                        # Ensure we have at least the current user message
                        messages_to_process = enriched_messages
                        if (
                            not messages_to_process
                            and conversation_ctx.current_user_message
                        ):
                            # If no enriched messages but we have current user message, create minimal list
                            messages_to_process = (
                                conversation_ctx.get_enriched_messages(mp, tools)
                            )
                            if not messages_to_process:
                                messages_to_process = [
                                    conversation_ctx.current_user_message
                                ]

                        if not messages_to_process:
                            logger.error("No messages to process")
                            yield serialize_to_json(
                                create_error_chunk("No messages to process")
                            )
                            return

                        chunk_count = 0
                        async for chunk in stream_pipeline(
                            messages_to_process, pipeline, tools
                        ):
                            chunk_count += 1
                            # Extract text content for accumulation
                            chunk_text = (
                                extract_message_text(chunk.message)
                                if chunk.message
                                else ""
                            )
                            full_response += chunk_text

                            # For streaming, only send serialized ChatResponse with text content
                            # Status messages and tool info come through as special chunks
                            yield serialize_to_json(chunk)

                        logger.info(
                            f"Pipeline execution completed with {chunk_count} chunks, response length: {len(full_response)}"
                        )
                except Exception as pipeline_error:
                    logger.error(f"Pipeline execution failed: {pipeline_error}")

                    # Only force cleanup if this appears to be a memory-related error
                    from utils.hardware_manager import is_memory_related_error

                    if is_memory_related_error(pipeline_error):
                        logger.info(
                            "Memory-related pipeline failure detected, performing cleanup"
                        )
                        try:
                            evicted = pipeline_factory.force_memory_cleanup()
                            logger.info(
                                f"Force cleanup after memory-related failure evicted {evicted} pipelines"
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Post-failure cleanup failed: {cleanup_error}"
                            )
                    else:
                        logger.debug(
                            "Non-memory pipeline failure detected, skipping aggressive cleanup"
                        )

                    # Try to provide a helpful error message to the user
                    yield serialize_to_json(
                        create_error_chunk(
                            f"Model execution failed: {str(pipeline_error)}. "
                            "This may be due to model compatibility issues or resource constraints."
                        )
                    )
                    return

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
                # Provide specific error message based on the error type
                if "Model profile" in str(e) and "not found" in str(e):
                    error_msg = "Model configuration not found. Please check your model settings."
                elif "Tool" in str(e) and "failed" in str(e):
                    error_msg = "Tool processing failed, but continuing with basic functionality."
                else:
                    error_msg = f"Processing error: {str(e)}"

                yield serialize_to_json(create_error_chunk(error_msg))


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
