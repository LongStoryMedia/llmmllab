"""
Anthropic API Pipeline - External model interface.
Provides simple interface to Anthropic models via API.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator

try:
    from langchain_anthropic import ChatAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ChatAnthropic = None
    ANTHROPIC_AVAILABLE = False
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


class AnthropicPipeline(SimpleChatPipeline):
    """
    Simple Anthropic API pipeline - direct model calls without orchestration.

    Features:
    - Direct Anthropic API initialization
    - Clean message conversion to Anthropic format
    - Streaming support
    - Tool calling integration
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.llm: Optional[ChatAnthropic] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def _initialize_llm(self, tools: Optional[List[BaseTool]] = None) -> ChatAnthropic:
        """Initialize Anthropic LLM with API configuration."""
        if self.llm is not None:
            return self.llm

        if not ANTHROPIC_AVAILABLE or ChatAnthropic is None:
            raise ImportError(
                "langchain_anthropic is not installed. Install it with: pip install langchain-anthropic"
            )

        try:
            # Get Anthropic configuration from environment
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")

            # Extract model name - use profile model_name or model name
            model_name = self.profile.model_name or self.model.name

            # Use common Anthropic model mappings
            anthropic_model_name = self._map_to_anthropic_model(model_name)

            # Create Anthropic LLM with profile parameters
            self.llm = ChatAnthropic(
                model=anthropic_model_name,
                api_key=SecretStr(api_key),
                temperature=self.profile.parameters.temperature or 0.7,
                max_tokens=self.profile.parameters.max_tokens or 4096,
                top_p=self.profile.parameters.top_p or 1.0,
                streaming=True,
                timeout=60.0,
            )

            self._logger.info(f"Anthropic LLM initialized: {anthropic_model_name}")
            return self.llm

        except Exception as e:
            self._logger.error(f"Failed to initialize Anthropic LLM: {e}")
            raise

    def _map_to_anthropic_model(self, model_name: str) -> str:
        """Map internal model names to Anthropic API model names."""
        model_name_lower = model_name.lower()

        # Common mappings
        if "claude-3" in model_name_lower:
            if "opus" in model_name_lower:
                return "claude-3-opus-20240229"
            elif "sonnet" in model_name_lower:
                return "claude-3-sonnet-20240229"
            elif "haiku" in model_name_lower:
                return "claude-3-haiku-20240307"
        elif "claude-3.5" in model_name_lower or "claude3.5" in model_name_lower:
            if "sonnet" in model_name_lower:
                return "claude-3-5-sonnet-20241022"
        elif "claude" in model_name_lower:
            # Default to Claude 3.5 Sonnet if no specific mapping
            return "claude-3-5-sonnet-20241022"

        # Return as-is if it looks like a valid Anthropic model name
        return model_name

    def _convert_messages(self, messages: List[Message]) -> List:
        """Convert internal messages to Anthropic format."""
        anthropic_messages = []

        # Add system message if present in profile
        if self.profile.system_prompt:
            anthropic_messages.append(SystemMessage(content=self.profile.system_prompt))

        # Convert conversation messages
        for msg in messages:
            content_text = ""
            for content in msg.content:
                if content.type == MessageContentType.TEXT and content.text:
                    content_text += content.text

            if msg.role == MessageRole.USER:
                anthropic_messages.append(HumanMessage(content=content_text))
            elif msg.role == MessageRole.ASSISTANT:
                # Skip empty assistant messages for cleaner conversation
                if content_text.strip():
                    anthropic_messages.append(AIMessage(content=content_text))

        return anthropic_messages

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """Invoke Anthropic LLM directly."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm(tools)

        try:
            # Convert messages
            anthropic_messages = self._convert_messages(messages)

            # Bind tools if provided
            llm = self.llm
            if tools and self.llm:
                llm = self.llm.bind_tools(tools)

            # Invoke LLM
            if llm is None:
                raise RuntimeError("LLM not initialized")
            response = await llm.ainvoke(anthropic_messages)

            # Extract content
            content = str(response.content) if response.content else ""

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=content)],
            )

            return ChatResponse(done=True, message=result_message)

        except Exception as e:
            self._logger.error(f"Anthropic LLM invocation failed: {e}")
            error_msg = f"Anthropic API Error: {str(e)}"
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
        """Stream responses from Anthropic LLM."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm(tools)

        try:
            # Convert messages
            anthropic_messages = self._convert_messages(messages)

            # Bind tools if provided
            llm = self.llm
            if tools and self.llm:
                llm = self.llm.bind_tools(tools)

            # Stream from LLM
            if llm is None:
                raise RuntimeError("LLM not initialized")

            async for chunk in llm.astream(anthropic_messages):
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
            self._logger.error(f"Anthropic LLM streaming failed: {e}")
            error_msg = f"Anthropic API Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            yield ChatResponse(done=True, message=error_message)
