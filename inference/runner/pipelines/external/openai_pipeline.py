"""
OpenAI API Pipeline - External model interface.
Provides simple interface to OpenAI models via API.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import SecretStr

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from runner.pipelines.base import SimpleChatPipeline, GrammarInput


class OpenAIPipeline(SimpleChatPipeline):
    """
    Simple OpenAI API pipeline - direct model calls without orchestration.

    Features:
    - Direct OpenAI API initialization
    - Clean message conversion to OpenAI format
    - Streaming support
    - Tool calling integration
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.llm: Optional[ChatOpenAI] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def _initialize_llm(self, tools: Optional[List[BaseTool]] = None) -> ChatOpenAI:
        """Initialize OpenAI LLM with API configuration."""
        if self.llm is not None:
            return self.llm

        try:
            # Get OpenAI configuration from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            # Extract model name - use profile model_name or model name
            model_name = self.profile.model_name or self.model.name

            # Use common OpenAI model mappings
            openai_model_name = self._map_to_openai_model(model_name)

            # Create OpenAI LLM with profile parameters
            self.llm = ChatOpenAI(
                model=openai_model_name,
                api_key=SecretStr(api_key),
                temperature=self.profile.parameters.temperature or 0.7,
                max_completion_tokens=self.profile.parameters.max_tokens or 4096,
                model_kwargs={
                    "top_p": self.profile.parameters.top_p or 1.0,
                },
                streaming=True,
                timeout=60.0,
            )

            self._logger.info(f"OpenAI LLM initialized: {openai_model_name}")
            return self.llm

        except Exception as e:
            self._logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise

    def _map_to_openai_model(self, model_name: str) -> str:
        """Map internal model names to OpenAI API model names."""
        model_name_lower = model_name.lower()

        # Common mappings
        if "gpt-4" in model_name_lower:
            if "turbo" in model_name_lower:
                return "gpt-4-turbo"
            elif "mini" in model_name_lower:
                return "gpt-4o-mini"
            else:
                return "gpt-4o"
        elif "gpt-3.5" in model_name_lower or "gpt3.5" in model_name_lower:
            return "gpt-3.5-turbo"
        elif "gpt-4o" in model_name_lower:
            return "gpt-4o"
        else:
            # Default to GPT-4o if no specific mapping
            return model_name

    def _convert_messages(self, messages: List[Message]) -> List:
        """Convert internal messages to OpenAI format."""
        openai_messages = []

        # Add system message if present in profile
        if self.profile.system_prompt:
            openai_messages.append(SystemMessage(content=self.profile.system_prompt))

        # Convert conversation messages
        for msg in messages:
            content_text = ""
            for content in msg.content:
                if content.type == MessageContentType.TEXT and content.text:
                    content_text += content.text

            if msg.role == MessageRole.USER:
                openai_messages.append(HumanMessage(content=content_text))
            elif msg.role == MessageRole.ASSISTANT:
                # Skip empty assistant messages for cleaner conversation
                if content_text.strip():
                    openai_messages.append(HumanMessage(content=content_text))

        return openai_messages

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """Invoke OpenAI LLM directly."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm(tools)

        try:
            # Convert messages
            openai_messages = self._convert_messages(messages)

            # Bind tools if provided
            llm = self.llm
            if tools:
                llm = self.llm.bind_tools(tools)

            # Invoke LLM
            if llm is None:
                raise RuntimeError("LLM not initialized")
            response = await llm.ainvoke(openai_messages)

            # Extract content
            content = str(response.content) if response.content else ""

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=content)],
            )

            return ChatResponse(done=True, message=result_message)

        except Exception as e:
            self._logger.error(f"OpenAI LLM invocation failed: {e}")
            error_msg = f"OpenAI API Error: {str(e)}"
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
        """Stream responses from OpenAI LLM."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm(tools)

        try:
            # Convert messages
            openai_messages = self._convert_messages(messages)

            # Bind tools if provided
            llm = self.llm
            if tools:
                llm = self.llm.bind_tools(tools)

            # Stream from LLM
            if llm is None:
                raise RuntimeError("LLM not initialized")

            async for chunk in llm.astream(openai_messages):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_text = str(chunk.content)

                    chunk_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=chunk_text
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
            self._logger.error(f"OpenAI LLM streaming failed: {e}")
            error_msg = f"OpenAI API Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            yield ChatResponse(done=True, message=error_message)
