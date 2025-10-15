"""
Simplified OpenAI GPT OSS pipeline implementation - pure LLM interface.
Just calls LLM directly with messages, configuration, and hardware management.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator, Dict, Any, ClassVar

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, BaseMessage
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
    allowed_return_types: ClassVar = (ChatResponse,)
    default_return_type: ClassVar = ChatResponse

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        **kwargs,
    ):
        super().__init__(model, profile, **kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "openai-gpt-oss-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update({
            "model_type": "openai-gpt-oss",
            "chat_format": "openai-gpt",
        })
        return base_params

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

    def _format_messages_for_llama(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Override message formatting for OpenAI GPT OSS format."""
        from langchain_core.messages import BaseMessage
        
        formatted_messages = []
        
        # Add system message if we have tools or system prompt
        system_prompt = self.profile_config.system_prompt or "You are a helpful AI assistant."
        formatted_messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation messages
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                
                if hasattr(msg, 'type'):
                    if msg.type == "human":
                        formatted_messages.append({"role": "user", "content": content})
                    elif msg.type == "ai":
                        formatted_messages.append({"role": "assistant", "content": content})
                
        return formatted_messages
