"""
Simplified Llama Chat Summary pipeline - pure LLM interface, no orchestration.
Replaced 641 lines of complex LangGraph orchestration with direct LLM calls.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator, Dict, Any

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
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
from utils.message import extract_message_text


class LlamaChatSummPipe(BaseLlamaCppPipeline):
    """
    Simplified Llama Chat Summary pipeline - direct LLM calls for summarization.

    Features:
    - Direct LlamaCpp initialization for summarization models
    - Clean prompt formatting optimized for summary generation
    - Automatic text preprocessing for better summaries
    - Hardware optimization for Llama 3.2 3B models
    """

    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(model, profile, **kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-chat-summary-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update({
            "model_type": "llama-chat-summary",
            "task": "summarization",
        })
        return base_params

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better summarization."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove very short lines that are likely formatting artifacts
        lines = text.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 10:  # Keep lines with substance
                filtered_lines.append(stripped)

        return "\n".join(filtered_lines) if filtered_lines else text

    def _create_summary_prompt(self, text: str, summary_type: str = "concise") -> str:
        """Create an optimized prompt for summarization."""
        preprocessed_text = self._preprocess_text(text)

        # Determine summary style based on text length
        word_count = len(preprocessed_text.split())

        if word_count < 200:
            instruction = "Provide a brief 2-3 sentence summary of the following text:"
        elif word_count < 1000:
            instruction = "Provide a concise summary in 3-5 sentences highlighting the key points:"
        else:
            instruction = "Provide a comprehensive summary in 5-8 sentences covering the main topics and important details:"

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert text summarizer. Your task is to create clear, accurate, and informative summaries that capture the essence of the original content.

Guidelines:
- Focus on main ideas and key information
- Use clear, concise language
- Maintain objectivity and accuracy
- Preserve important context and details

<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

{preprocessed_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt for Llama Chat Summary."""
        base_prompt = "You are an expert summarization assistant. Provide clear, accurate, and concise summaries."

        if not tools:
            return base_prompt

        # Add tool descriptions if provided
        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- {tool.name}: {tool.description}"
            tool_descriptions.append(tool_desc)

        tools_section = "Available tools:\n" + "\n".join(tool_descriptions)
        return f"{base_prompt}\n\n{tools_section}"

    def _format_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Format messages for Llama Chat Summary format."""
        # Extract text to summarize
        texts_to_summarize = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue  # Skip system messages for summarization

            content_text = ""
            for content in msg.content:
                if content.type == MessageContentType.TEXT and content.text:
                    content_text += content.text

            if content_text:
                texts_to_summarize.append(content_text)

        # Combine all texts to summarize
        combined_text = "\n\n".join(texts_to_summarize)

        # Create summary prompt
        return self._create_summary_prompt(combined_text)

    def _format_messages_for_llama(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Override message formatting for Llama Chat Summary with summary-specific prompts."""
        from langchain_core.messages import HumanMessage
        
        # Extract text content from messages for summarization
        texts_to_summarize = []
        
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                texts_to_summarize.append(str(msg.content))
        
        # Combine all texts and create summary prompt
        combined_text = "\n\n".join(texts_to_summarize)
        summary_prompt = self._create_summary_prompt(combined_text)
        
        # Return formatted for llama-cpp-python
        return [{"role": "user", "content": summary_prompt}]
