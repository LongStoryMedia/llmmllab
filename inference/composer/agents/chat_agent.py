"""
Chat Agent for LLM chat model operations.
Provides core business logic for chat completions, streaming, and tool integration.
"""

from typing import List, cast, Optional, Dict, Any

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
from composer.utils.conversion import message_to_langchain_message
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
        stream: bool = False,
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
        self.stream = stream

    async def execute_pipeline(self, stream: bool = False, **kwargs) -> ChatResponse:
        """
        Execute chat pipeline with the provided parameters.
        
        This is the standard interface for pipeline execution required by BaseAgent.
        
        Args:
            stream: Whether to stream the response
            **kwargs: Pipeline execution parameters, expected to include:
                - messages: List of LangChainMessage objects
                - user_id: User identifier
                - tools: Optional list of BaseTool objects
                - circuit_breaker: Optional CircuitBreakerConfig
        
        Returns:
            ChatResponse: The completion result
        """
        messages = kwargs.get('messages', [])
        user_id = kwargs.get('user_id', '')
        tools = kwargs.get('tools')
        circuit_breaker = kwargs.get('circuit_breaker')
        
        return await self.chat_completion(
            messages=messages,
            user_id=user_id,
            tools=tools,
            circuit_breaker=circuit_breaker,
            stream=stream
        )

    async def chat_completion(
        self,
        messages: List[LangChainMessage],
        user_id: str,
        tools: Optional[List[BaseTool]] = None,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
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

        try:
            self._log_operation_start(
                "chat_completion",
                user_id=user_id,
                message_count=len(messages),
                has_tools=bool(tools),
                streaming=should_stream,
                model=self.profile.model_name if self.profile else "unknown",
            )

            self.update_execution_context(
                operation="chat_completion",
                message_count=len(messages),
                tool_count=len(tools) if tools else 0,
                streaming=should_stream,
            )

            # Execute pipeline based on streaming configuration
            with self.pipeline_factory.pipeline(
                self.profile, ChatResponse, self.priority, circuit_breaker
            ) as pipe:
                
                if should_stream:
                    response = await self._execute_streaming_completion(
                        messages, pipe, tools, user_id
                    )
                else:
                    response = await self._execute_completion(
                        messages, pipe, tools, user_id
                    )

                self._log_operation_success(
                    "chat_completion",
                    user_id=user_id,
                    has_response=bool(response),
                    has_message=bool(response.message if response else False),
                    tool_calls_count=len(response.message.tool_calls) if response and response.message and response.message.tool_calls else 0,
                )

                return response

        except Exception as e:
            self._handle_node_error(
                "chat_completion",
                e,
                user_id=user_id,
                message_count=len(messages),
                has_tools=bool(tools),
            )

    async def _execute_streaming_completion(
        self,
        messages: List[LangChainMessage],
        pipeline: Any,
        tools: Optional[List[BaseTool]],
        user_id: str,
    ) -> ChatResponse:
        """Execute streaming chat completion."""
        # Lazy imports to avoid circular dependency
        from runner import stream_pipeline  # pylint: disable=import-outside-toplevel

        self.logger.info(
            "Starting streaming chat completion",
            user_id=user_id,
            pipeline_type=type(pipeline).__name__,
        )

        # Accumulate streaming response
        final_content = ""
        tool_calls = []
        chunk_count = 0

        async for chunk in stream_pipeline(
            messages,
            pipeline,
            cast(List[BaseTool], tools) if tools else None,
        ):
            chunk_count += 1

            self.logger.debug(
                "Received streaming chunk",
                user_id=user_id,
                chunk_num=chunk_count,
                has_message=bool(chunk.message),
                chunk_done=chunk.done,
            )

            if chunk.message:
                # Append new text content
                final_content += extract_message_text(chunk.message)

                # Collect tool calls
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

                if chunk.done:
                    break

        self.logger.info(
            "Streaming completion finished",
            user_id=user_id,
            total_chunks=chunk_count,
            content_length=len(final_content),
            tool_calls_count=len(tool_calls),
        )

        # Create final response from accumulated content
        response = ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, 
                        text=final_content
                    )
                ] if final_content else [],
                tool_calls=tool_calls if tool_calls else None,
            ),
            finish_reason="stop",
        )

        return response

    async def _execute_completion(
        self,
        messages: List[LangChainMessage],
        pipeline: Any,
        tools: Optional[List[BaseTool]],
        user_id: str,
    ) -> ChatResponse:
        """Execute non-streaming chat completion."""
        # Lazy imports to avoid circular dependency
        from runner import run_pipeline  # pylint: disable=import-outside-toplevel

        self.logger.info(
            "Starting non-streaming chat completion",
            user_id=user_id,
            pipeline_type=type(pipeline).__name__,
        )

        response = await run_pipeline(
            messages,
            pipeline,
            cast(List[BaseTool], tools) if tools else None,
        )

        self.logger.info(
            "Non-streaming completion finished",
            user_id=user_id,
            has_response=bool(response),
            tool_calls_count=len(response.message.tool_calls) if response and response.message and response.message.tool_calls else 0,
        )

        return response

    def convert_to_langchain_message(
        self, 
        response: ChatResponse,
        user_id: str,
    ) -> LangChainMessage:
        """
        Convert ChatResponse to LangChainMessage for workflow integration.

        Args:
            response: ChatResponse from pipeline execution
            user_id: User identifier for logging

        Returns:
            LangChainMessage compatible with workflow state
        """
        try:
            self._log_operation_start(
                "message_conversion",
                user_id=user_id,
                has_response=bool(response),
                has_message=bool(response.message if response else False),
            )

            if response and response.message:
                # Convert using existing utility
                langchain_message = message_to_langchain_message(response.message)

                self.logger.debug(
                    "Message conversion details",
                    user_id=user_id,
                    original_tool_calls=bool(response.message.tool_calls),
                    converted_tool_calls=bool(getattr(langchain_message, "tool_calls", None)),
                )

                self._log_operation_success(
                    "message_conversion",
                    user_id=user_id,
                    has_tool_calls=bool(getattr(langchain_message, "tool_calls", None)),
                )

                return langchain_message
            else:
                # Fallback message
                fallback_message = LangChainMessage(
                    type="ai",
                    content="No response generated from chat completion",
                )

                self.logger.warning(
                    "Using fallback message for empty response",
                    user_id=user_id,
                )

                return fallback_message

        except Exception as e:
            self._handle_node_error(
                "message_conversion",
                e,
                user_id=user_id,
                has_response=bool(response),
            )

    async def chat_completion_with_conversion(
        self,
        messages: List[LangChainMessage],
        user_id: str,
        tools: Optional[List[BaseTool]] = None,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
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
                user_id=user_id,
                tools=tools,
                circuit_breaker=circuit_breaker,
                stream=stream,
            )

            # Convert to LangChain message
            return self.convert_to_langchain_message(response, user_id)

        except Exception as e:
            self._handle_node_error(
                "chat_completion_with_conversion",
                e,
                user_id=user_id,
                message_count=len(messages),
            )

    def extract_tool_calls(self, message: LangChainMessage) -> Optional[List[Dict[str, Any]]]:
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