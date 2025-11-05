"""
Chat Agent for LLM chat model operations.
Provides core business logic for chat completions, streaming, and tool integration.
"""

from typing import List, Optional
from datetime import datetime, timezone

from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from runner import PipelineFactory
from models import (
    ChatResponse,
    ModelProfile,
    PipelinePriority,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    NodeMetadata,
    ToolCall,
)
from utils.tool_call_extraction import (
    extract_tool_calls_from_langchain_message,
    has_tool_calls_in_langchain_message,
    extract_tool_calls_from_streaming_chunks,
    extract_tool_calls_from_message_content,
    create_tool_call_message_content,
)
from utils.message_conversion import lc_messages_to_messages
from utils import create_error_response
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
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        stream: bool = True,
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

        if stream:
            # For streaming, we need to accumulate the response
            return await self._execute_streaming_completion_with_metadata(
                messages,
                tools,
            )
        # For non-streaming, use the base class method directly
        return await self.run(
            messages=messages,
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
        chunks = []
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
                chunks.append(chunk)

                # Accumulate text content
                if chunk.message and chunk.message.content:
                    for content in chunk.message.content:
                        if content.type in (
                            MessageContentType.THINKING,
                            MessageContentType.TEXT,
                        ):
                            if hasattr(content, "text") and content.text:
                                final_content += content.text

            # Extract tool calls using shared utility
            tool_execution_results = extract_tool_calls_from_streaming_chunks(chunks)

            self.logger.info(
                "Streaming completion with metadata finished",
                total_chunks=chunk_count,
                content_length=len(final_content),
                tool_calls_count=len(tool_execution_results),
            )

            # Create final response from accumulated content
            content_items = []
            if final_content:
                content_items.append(
                    MessageContent(type=MessageContentType.TEXT, text=final_content)
                )

            # Add tool calls as content items using shared utility
            for tool_call in tool_execution_results:
                tool_call_content = create_tool_call_message_content(tool_call)
                if tool_call_content:
                    content_items.append(tool_call_content)

            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=content_items,
            )

            # Update final_message with tool_calls if present
            if tool_execution_results:
                final_message.tool_calls = tool_execution_results

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
        messages: List[BaseMessage],
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
            messages=lc_messages_to_messages(messages),
            tools=tools,
            priority=self.priority,
        ):
            yield chunk

    async def chat_completion_with_conversion(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        stream: bool = True,
    ) -> Message:
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

            if response.message is None:
                err_response = create_error_response(
                    "No message returned from chat completion"
                )
                assert err_response.message is not None
                return err_response.message

            return response.message

        except Exception as e:
            self._handle_node_error(
                "chat_completion_with_conversion",
                e,
                message_count=len(messages),
            )

            err_response = create_error_response(
                "Error during chat completion with conversion"
            )
            assert err_response.message is not None
            return err_response.message

    def extract_tool_call_requests(self, message: BaseMessage) -> List[ToolCall]:
        """
        Extract tool call requests from a LangChain message using shared utilities.

        Args:
            message: LangChain message to extract tool calls from

        Returns:
            List of validated ToolCall objects
        """
        return extract_tool_calls_from_langchain_message(message)

    def has_tool_call_requests(self, message: BaseMessage) -> bool:
        """
        Check if a LangChain message has tool call requests using shared utilities.

        Args:
            message: LangChain message to check

        Returns:
            True if message has tool call requests, False otherwise
        """
        return has_tool_calls_in_langchain_message(message)
