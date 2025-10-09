"""
Simplified OpenAI GPT OSS pipeline implementation - pure LLM interface.
Just calls LLM directly with messages, configuration, and hardware management.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

try:
    from langchain_community.chat_models.llamacpp import ChatLlamaCpp
except ImportError:
    ChatLlamaCpp = None
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from runner.pipelines.base import GrammarInput
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class OpenAIGptOssPipeline(BaseLlamaCppPipeline):
    """
    Simplified OpenAI GPT OSS pipeline - pure LLM interface.

    Just calls LLM directly with messages, configuration, and hardware management.
    No orchestration, no graphs - exactly what composer needs.
    """

    # Override to allow ChatResponse return type
    allowed_return_types = (ChatResponse,)
    default_return_type = ChatResponse

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
    ):
        super().__init__(model, profile)
        self._logger = logging.getLogger(self.__class__.__name__)

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt with tool descriptions if tools provided."""
        base_prompt = self.profile.system_prompt or "You are a helpful AI assistant."

        if not tools:
            return base_prompt

        # Create simple tool descriptions
        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- {tool.name}: {tool.description}"
            tool_descriptions.append(tool_desc)

        tools_section = "Available tools:\n" + "\n".join(tool_descriptions)

        return f"{base_prompt}\n\n{tools_section}"

    def _format_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Format messages using GPT OSS standard format."""
        formatted_parts = []

        # Add system prompt
        system_prompt = self._create_system_prompt(tools)
        formatted_parts.append(f"<|start|>system<|message|>{system_prompt}<|end|>")

        # Add conversation messages
        for msg in messages:
            content_text = ""
            for content in msg.content:
                if content.type == MessageContentType.TEXT and content.text:
                    content_text += content.text

            if msg.role == MessageRole.USER:
                formatted_parts.append(f"<|start|>user<|message|>{content_text}<|end|>")
            elif msg.role == MessageRole.ASSISTANT:
                formatted_parts.append(
                    f"<|start|>assistant<|message|>{content_text}<|end|>"
                )
            # Skip system messages as we handle them above

        # Add assistant start for completion
        formatted_parts.append("<|start|>assistant")

        return "\n".join(formatted_parts)

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """
            tools: Available tools (descriptions added to prompt)
            grammar: Grammar constraints (applied to LLM)

        Returns:
            ChatResponse: Response from LLM
        """
        _ = grammar, kwargs  # Suppress unused warnings
        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm(tools)

        try:
            # Format conversation
            formatted_prompt = self._format_messages(messages, tools)

            # Invoke LLM directly
            if self.llm is None:
                raise RuntimeError("LLM not initialized")
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])

            # Extract content
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=str(content))
                ],
            )

            return ChatResponse(done=True, message=result_message)

        except Exception as e:
            self._logger.error(f"LLM invocation failed: {e}")
            error_msg = f"Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            return ChatResponse(done=True, message=error_message)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> AsyncIterator[ChatResponse]:
        """
        Stream responses from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools (descriptions added to prompt)
            grammar: Grammar constraints (applied to LLM)

        Yields:
            ChatResponse: Streaming chunks from LLM
        """
        _ = grammar, kwargs  # Suppress unused warnings
        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm(tools)

        try:
            # Format conversation
            formatted_prompt = self._format_messages(messages, tools)

            # Stream from LLM
            if self.llm is None:
                raise RuntimeError("LLM not initialized")
            async for chunk in self.llm.astream(
                [HumanMessage(content=formatted_prompt)]
            ):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=str(chunk.content)
                            )
                        ],
                    )
                    yield ChatResponse(done=False, message=chunk_message)

            # Final chunk to indicate completion
            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="")],
            )
            yield ChatResponse(done=True, message=final_message)

        except Exception as e:
            self._logger.error(f"LLM streaming failed: {e}")
            error_msg = f"Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            yield ChatResponse(done=True, message=error_message)
