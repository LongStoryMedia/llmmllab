"""
Simplified Llama Chat Summary pipeline - pure LLM interface, no orchestration.
Replaced 641 lines of complex LangGraph orchestration with direct LLM calls.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.messages import HumanMessage
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

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self._logger = logging.getLogger(self.__class__.__name__)

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

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """Invoke the Llama Chat Summary LLM directly."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Format conversation
            formatted_prompt = self._format_messages(messages, tools)

            # Invoke LLM directly
            if self.llm is None:
                raise RuntimeError("LLM not initialized")
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])

            # Extract content
            content = str(response.content) if response.content else ""

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=content)],
            )

            return ChatResponse(done=True, message=result_message)

        except Exception as e:
            self._logger.error(f"Llama Chat Summary invocation failed: {e}")
            error_msg = f"Summarization error: {str(e)}"
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
        """Stream responses from Llama Chat Summary LLM."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

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
            self._logger.error(f"Llama Chat Summary streaming failed: {e}")
            error_msg = f"Summarization error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            yield ChatResponse(done=True, message=error_message)
