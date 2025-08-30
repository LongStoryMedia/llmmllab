"""
LangChain tool wrappers for RAG components compatible with latest BaseTool API.
"""

import asyncio
import json
from typing import List

from langchain_core.tools import BaseTool

from models.message import Message
from server.services.context import ConversationContext
from server.config import logger


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

    async def _arun(self, *args, **kwargs) -> str:
        """Async implementation for memory retrieval"""
        try:
            tool_input = args[0] if args else kwargs.get("tool_input")
            embeddings: List[List[float]] = (
                tool_input if isinstance(tool_input, list) else []
            )
            if not embeddings:
                return "No embeddings provided"

            memories = await self.conversation_ctx.memory_context.retrieve_memories(
                embeddings
            )

            if memories:
                return f"Retrieved memories: {json.dumps(memories)}"
            return "No relevant memories found"
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return f"Memory retrieval failed: {str(e)}"

    def _run(self, *args, **kwargs) -> str:
        """Sync fallback - not recommended for production"""
        return asyncio.run(self._arun(*args, **kwargs))


class WebSearchTool(BaseTool):
    """Tool for web search functionality"""

    name: str = "web_search"
    description: str = "Perform a web search and retrieve relevant results"
    conversation_ctx: ConversationContext

    def __init__(self, conversation_ctx: ConversationContext):
        super().__init__(conversation_ctx=conversation_ctx)

    async def _arun(self, *args, **kwargs) -> str:
        """Async implementation for web search"""
        try:
            tool_input = args[0] if args else kwargs.get("tool_input")
            query: Message = tool_input if isinstance(tool_input, Message) else None  # type: ignore
            # Perform search
            results = await self.conversation_ctx.search_context.search(
                query, getattr(self.conversation_ctx.conversation, "id", 0)
            )

            if results:
                return f"Web search results: {json.dumps(results)}"
            return "No relevant web results found"
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"

    def _run(self, *args, **kwargs) -> str:
        """Sync fallback"""
        return asyncio.run(self._arun(*args, **kwargs))


class SummarizationTool(BaseTool):
    """Tool for conversation summarization"""

    name: str = "summarization"
    description: str = "Summarize the conversation context"
    conversation_ctx: ConversationContext

    def __init__(self, conversation_ctx: ConversationContext):
        super().__init__(conversation_ctx=conversation_ctx)

    async def _arun(self, *args, **kwargs) -> str:
        """Async implementation for summarization"""
        try:
            # Perform summarization
            tool_input = args[0] if args else kwargs.get("tool_input")
            messages = tool_input if isinstance(tool_input, list) else []
            await self.conversation_ctx.summary_context.summarize(messages)

            return "No summary generated"
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"Summarization failed: {str(e)}"

    def _run(self, *args, **kwargs) -> str:
        """Sync fallback"""
        return "Summarization requires async execution"
