"""
Complete LangChain integration that preserves existing Pydantic models and streaming interface
"""

import asyncio
import json
import logging
from datetime import datetime as dt
from typing import Any, AsyncIterable, Dict, List, Optional

from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from models.chat_req import ChatReq
from models.chat_response import ChatResponse
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from server.context.conversation import ConversationContext, SummaryContext
from server.utils.chat.message import extract_message_text, to_lc_message
from server.config import logger
from server.context.memory import MemoryContext


# ============================================================================
# LangChain Tools for RAG Components
# ============================================================================


class MemoryRetrievalTool(BaseTool):
    """Tool for retrieving conversation memories using embeddings"""

    name: str = "memory_retrieval"
    description: str = "Retrieve relevant memories based on query embeddings"
    conversation_ctx: ConversationContext

    def __init__(self, conversation_ctx: ConversationContext):
        super().__init__(conversation_ctx=conversation_ctx)

    async def _arun(
        self, embeddings: List[List[float]], run_manager=None, **kwargs
    ) -> str:
        """Async implementation for memory retrieval"""
        try:
            assert (
                self.conversation_ctx.memory_context
            ), "Memory context must be initialized"
            # Retrieve memories
            memories = await self.conversation_ctx.memory_context.retrieve_memories(
                embeddings
            )

            if memories:
                return f"Retrieved memories: {json.dumps(memories)}"
            return "No relevant memories found"
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return f"Memory retrieval failed: {str(e)}"

    def _run(self, embeddings: List[List[float]], **kwargs) -> str:
        """Sync fallback - not recommended for production"""
        return asyncio.run(self._arun(embeddings, **kwargs))


class WebSearchTool(BaseTool):
    """Tool for web search functionality"""

    name: str = "web_search"
    description: str = "Perform a web search and retrieve relevant results"
    conversation_ctx: ConversationContext

    def __init__(self, conversation_ctx: ConversationContext):
        super().__init__(conversation_ctx=conversation_ctx)

    async def _arun(self, query: Message, run_manager=None, **kwargs) -> str:
        """Async implementation for web search"""
        try:
            # Perform search
            results = await self.conversation_ctx.search_context.search(
                query, kwargs.get("conversation_id", 0)
            )

            if results:
                return f"Web search results: {json.dumps(results)}"
            return "No relevant web results found"
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"

    def _run(self, query: Message, **kwargs) -> str:
        """Sync fallback"""
        return asyncio.run(self._arun(query, **kwargs))


class SummarizationTool(BaseTool):
    """Tool for conversation summarization"""

    name: str = "summarization"
    description: str = "Summarize the conversation context"
    conversation_ctx: ConversationContext

    def __init__(self, conversation_ctx: ConversationContext):
        super().__init__(conversation_ctx=conversation_ctx)

    async def _arun(self, messages: List, run_manager=None, **kwargs) -> str:
        """Async implementation for summarization"""
        try:
            # Perform summarization
            await self.conversation_ctx.summary_context.summarize(messages)

            return "No summary generated"
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"Summarization failed: {str(e)}"

    def _run(self, messages: str, **kwargs) -> str:
        """Sync fallback"""
        return "Summarization requires async execution"
