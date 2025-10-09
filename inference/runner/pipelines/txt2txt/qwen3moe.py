"""
Simplified Qwen3 MoE pipeline - pure LLM interface, no orchestration.
Replaced 1020 lines of LangGraph complexity with direct LLM calls.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

# (Removed unused langchain imports from simplified runner pipeline)

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


class Qwen3Moe(BaseLlamaCppPipeline):
    """
    Simplified Qwen3 MoE pipeline - direct LLM calls with <think> tag processing.

    Features:
    - Direct LlamaCpp initialization
    - Clean message formatting with Qwen chat format
    - Hardware optimization for MoE models
    - Simple <think> tag extraction
    """

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
        """Create system prompt for Qwen with tool descriptions."""
        base_prompt = (
            self.profile.system_prompt or "You are Qwen, a helpful AI assistant."
        )

        if not tools:
            return base_prompt

        # Create tool descriptions for Qwen format
        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- {tool.name}: {tool.description}"
            tool_descriptions.append(tool_desc)

        tools_section = "Available tools:\n" + "\n".join(tool_descriptions)

        return f"{base_prompt}\n\n{tools_section}"

    async def _format_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Format messages using Qwen chat format."""
        formatted_parts = []

        # Add system prompt
        system_prompt = await self._create_system_prompt(tools)
        formatted_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # Add conversation messages
        for msg in messages:
            content_text = ""
            for content in msg.content:
                if content.type == MessageContentType.TEXT and content.text:
                    content_text += content.text

            if msg.role == MessageRole.USER:
                formatted_parts.append(f"<|im_start|>user\n{content_text}<|im_end|>")
            elif msg.role == MessageRole.ASSISTANT:
                formatted_parts.append(
                    f"<|im_start|>assistant\n{content_text}<|im_end|>"
                )

        # Add assistant start for completion
        formatted_parts.append("<|im_start|>assistant\n")

        return "\n".join(formatted_parts)

    def _extract_response_content(self, raw_response: str) -> str:
        """Extract response content and handle <think> tags."""
        # Remove <think>...</think> blocks for cleaner output
        import re

        # Remove think tags and their content
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        cleaned = cleaned.strip()

        return cleaned or raw_response  # Fallback to original if nothing left

    def _parse_tool_calls(self, content: str) -> List[dict]:
        """Parse tool calls from XML format."""
        import json
        import re
        
        tool_calls = []
        
        # Look for <tool_call> XML tags - handle multiline JSON
        tool_call_pattern = r"<tool_call>\s*(\{[^<]*?\})\s*</tool_call>"
        matches = re.findall(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)
        
        self._logger.debug(f"Parsing tool calls from content: {content[:500]}...")
        self._logger.debug(f"Found {len(matches)} potential tool call matches")

        for i, match in enumerate(matches):
            try:
                # Parse the JSON content
                tool_data = json.loads(match)

                if "name" in tool_data:
                    formatted_call = {
                        "name": tool_data["name"],
                        "args": tool_data.get("arguments", {}),
                        "id": f"call_{i}_{tool_data['name']}",
                        "type": "tool_call",
                    }
                    tool_calls.append(formatted_call)
                    self._logger.debug(f"Parsed XML tool call: {formatted_call}")
                else:
                    self._logger.warning(
                        f"Tool call missing 'name' field: {match[:100]}..."
                    )

            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(
                    f"Failed to parse XML tool call from: {match[:100]}... Error: {e}"
                )
                continue

        self._logger.debug(f"Returning {len(tool_calls)} parsed tool calls")
        return tool_calls

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """Invoke the Qwen LLM directly."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Format conversation
            formatted_prompt = await self._format_messages(messages, tools)

            # Invoke LLM directly
            if self.llm is None:
                raise RuntimeError("LLM not initialized")
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])

            # Extract and clean content
            raw_content = str(response.content) if response.content else ""
            cleaned_content = self._extract_response_content(raw_content)
            
            # Parse tool calls from raw content
            self._logger.info(f"QWEN3MOE INVOKE: tools param = {len(tools) if tools else 'None'}")
            self._logger.info(f"QWEN3MOE INVOKE: raw_content preview = {raw_content[:200]}...")
            tool_calls = self._parse_tool_calls(raw_content) if tools else None
            self._logger.info(f"QWEN3MOE INVOKE: parsed tool_calls = {len(tool_calls) if tool_calls else 'None'}")

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=cleaned_content)
                ],
                tool_calls=tool_calls,
            )
            return ChatResponse(done=True, message=result_message)

        except Exception as e:
            self._logger.error(f"Qwen LLM invocation failed: {e}")
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
        """Stream responses from Qwen LLM."""
        _ = grammar, kwargs  # Suppress unused warnings
        
        self._logger.info(f"QWEN3MOE STREAM START: tools param = {len(tools) if tools else 'None'}")

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Format conversation
            formatted_prompt = await self._format_messages(messages, tools)

            # Stream from LLM
            if self.llm is None:
                raise RuntimeError("LLM not initialized")

            accumulated_content = ""
            async for chunk in self.llm.astream(
                [HumanMessage(content=formatted_prompt)]
            ):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_text = str(chunk.content)
                    accumulated_content += chunk_text

                    # For streaming, we send raw chunks and clean at the end
                    chunk_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=chunk_text
                            )
                        ],
                    )
                    yield ChatResponse(done=False, message=chunk_message)

            # Final chunk to indicate completion with tool calls
            cleaned_content = self._extract_response_content(accumulated_content)
            self._logger.info(f"QWEN3MOE STREAM: tools param = {len(tools) if tools else 'None'}")
            self._logger.info(f"QWEN3MOE STREAM: accumulated_content preview = {accumulated_content[:200]}...")
            tool_calls = self._parse_tool_calls(accumulated_content) if tools else None
            self._logger.info(f"QWEN3MOE STREAM: parsed tool_calls = {len(tool_calls) if tool_calls else 'None'}")
            
            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=cleaned_content)],
                tool_calls=tool_calls,
            )
            
            # Debug: Log final message tool calls
            self._logger.info(
                f"QWEN3MOE STREAM: Final message created with tool_calls={len(final_message.tool_calls) if final_message.tool_calls else 0}"
            )
            
            yield ChatResponse(done=True, message=final_message)

        except Exception as e:
            self._logger.error(f"Qwen LLM streaming failed: {e}")
            error_msg = f"Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            yield ChatResponse(done=True, message=error_message)
