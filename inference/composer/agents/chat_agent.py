"""
Chat Agent for LLM chat model operations.
Provides core business logic for chat completions, streaming, and tool integration.
"""

from typing import List, cast, Optional, Dict, Any
from datetime import datetime, timezone

from langchain.tools import BaseTool

from runner import PipelineFactory
from models import (
    ChatResponse,
    LangChainMessage,
    ModelProfile,
    PipelinePriority,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    CircuitBreakerConfig,
    NodeMetadata,
)
from utils.message import extract_message_text
from composer.utils.conversion import (
    message_to_langchain_message,
    convert_langchain_messages_to_messages,
)
from .base_agent import BaseAgent


class ChatAgent(BaseAgent[ChatResponse]):
    """
    Chat Agent for LLM chat model operations with streaming and tool support.

    Provides core business logic for chat completions, handling both streaming
    and non-streaming execution, tool integration, and response processing.
    Supports model profile configuration and circuit breaker integration.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile: ModelProfile,
        node_metadata: NodeMetadata,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
    ):
        """
        Initialize chat agent with dependency injection.

        Args:
            pipeline_factory: Factory for creating chat pipelines
            profile: Model profile for chat operations
            node_metadata: Node execution metadata for tracking
            priority: Pipeline execution priority
            stream: Whether to enable streaming responses by default
        """
        super().__init__(pipeline_factory, profile, node_metadata)
        self.priority = priority

    async def chat_completion(
        self,
        messages: List[LangChainMessage],
        tools: Optional[List[BaseTool]] = None,
        stream: Optional[bool] = None,
    ) -> ChatResponse:
        """
        Execute chat completion with optional streaming and tool support.

        Args:
            messages: Context messages for the chat completion
            user_id: User identifier
            tools: Optional tools available for the chat completion
            circuit_breaker: Optional circuit breaker configuration
            stream: Override default streaming behavior

        Returns:
            ChatResponse with the completion result
        """
        # Use provided stream setting or default
        should_stream = stream if stream is not None else self.stream

        if should_stream:
            # For streaming, we need to accumulate the response
            return await self._execute_streaming_completion_with_metadata(
                convert_langchain_messages_to_messages(messages),
                tools,
            )
        else:
            # For non-streaming, use the base class method directly
            return await self.run(
                messages=convert_langchain_messages_to_messages(messages),
                tools=tools,
                priority=self.priority,
            )

    async def _execute_streaming_completion_with_metadata(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
    ) -> ChatResponse:
        """Execute streaming chat completion using BaseAgent methods with metadata."""
        # Accumulate streaming response
        final_content = ""
        tool_calls = []
        chunk_count = 0

        try:
            async for chunk in self.stream(
                messages=messages,
                tools=tools,
                priority=self.priority,
            ):
                # Skip metadata boundary chunks
                if chunk.channels and chunk.channels.get("stream_metadata", {}).get(
                    "is_boundary"
                ):
                    continue

                chunk_count += 1

                # Accumulate content
                if chunk.message and chunk.message.content:
                    content_text = extract_message_text(chunk.message)
                    if content_text:
                        final_content += content_text

                # Collect tool calls
                if chunk.message and chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            self.logger.info(
                "Streaming completion with metadata finished",
                total_chunks=chunk_count,
                content_length=len(final_content),
                tool_calls_count=len(tool_calls),
            )

            # Create final response from accumulated content
            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=(
                    [MessageContent(type=MessageContentType.TEXT, text=final_content)]
                    if final_content
                    else []
                ),
                tool_calls=tool_calls if tool_calls else None,
            )

            return ChatResponse(
                done=True,
                message=final_message,
                finish_reason="stop",
                created_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            self._handle_node_error(
                "streaming_completion_with_metadata",
                e,
                message_count=len(messages),
            )
            return ChatResponse(done=True, message=None, finish_reason="error")

    async def stream_chat_completion(
        self,
        messages: List[LangChainMessage],
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Stream chat completion with metadata injection.

        This method is designed for LangGraph integration where you want to
        stream responses with node metadata for better observability.

        Args:
            messages: Context messages for the chat completion
            user_id: User identifier
            tools: Optional tools available for the chat completion
            circuit_breaker: Optional circuit breaker configuration

        Yields:
            ChatResponse: Streaming chunks with injected node metadata
        """
        async for chunk in self.stream(
            messages=convert_langchain_messages_to_messages(messages),
            tools=tools,
            priority=self.priority,
        ):
            yield chunk

    async def chat_completion_with_conversion(
        self,
        messages: List[LangChainMessage],
        tools: Optional[List[BaseTool]] = None,
        stream: Optional[bool] = None,
    ) -> LangChainMessage:
        """
        Execute chat completion and convert response to LangChainMessage.

        Convenience method that combines chat completion and message conversion.

        Args:
            messages: Context messages for the chat completion
            user_id: User identifier
            tools: Optional tools available for the chat completion
            circuit_breaker: Optional circuit breaker configuration
            stream: Override default streaming behavior

        Returns:
            LangChainMessage ready for workflow integration
        """
        try:
            # Execute chat completion
            response = await self.chat_completion(
                messages=messages,
                tools=tools,
                stream=stream,
            )

            # Convert to LangChain message
            return (
                message_to_langchain_message(response.message)
                if response.message
                else LangChainMessage(
                    type="ai",
                    content="",
                )
            )

        except Exception as e:
            self._handle_node_error(
                "chat_completion_with_conversion",
                e,
                message_count=len(messages),
            )
            return LangChainMessage(
                type="ai",
                content="Error during chat completion with conversion",
            )

    def extract_tool_calls(
        self, message: LangChainMessage
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tool calls from a LangChain message.

        Args:
            message: LangChain message to extract tool calls from

        Returns:
            List of tool call dictionaries or None if no tool calls
        """
        try:
            tool_calls = getattr(message, "tool_calls", None)

            if tool_calls:
                self.logger.debug(
                    "Extracted tool calls",
                    tool_calls_count=len(tool_calls),
                    tool_calls_preview=str(tool_calls)[:200],
                )

            return tool_calls

        except Exception as e:
            self.logger.error(
                "Failed to extract tool calls",
                error=str(e),
                message_type=getattr(message, "type", "unknown"),
            )
            return None

    def has_tool_calls(self, message: LangChainMessage) -> bool:
        """
        Check if a LangChain message has tool calls.

        Args:
            message: LangChain message to check

        Returns:
            True if message has tool calls, False otherwise
        """
        tool_calls = self.extract_tool_calls(message)
        return bool(tool_calls)
