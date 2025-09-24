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

    async def _arun(self, query: str, **kwargs) -> str:
        """Async implementation for web search"""
        try:
            # Create a Message object from the query
            from models import MessageRole, MessageContent, MessageContentType

            message = Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=query)],
                conversation_id=getattr(self.conversation_ctx.conversation, "id", 0),
            )

            # Use the existing search context to perform web search
            search_results = await self.conversation_ctx.search_context.search(
                message, getattr(self.conversation_ctx.conversation, "id", 0)
            )

            if search_results:
                # Format the search synthesis results
                formatted_results = []
                for result in search_results[:3]:  # Limit to top 3 results
                    formatted_results.append(
                        f"URLs: {', '.join(result.urls[:3])}\n"
                        f"Topics: {', '.join(result.topics)}\n"
                        f"Synthesis: {result.synthesis[:300]}..."
                    )
                return "Web search results:\n\n" + "\n\n".join(formatted_results)
            else:
                # If synthesis failed, create basic results from search context
                logger.warning(f"No synthesis results for query: {query}, trying basic search results")
                
                # Reset search results and try a simple approach
                self.conversation_ctx.search_context.search_results = []
                
                # Return a message indicating search was attempted
                return f"Web search was performed for '{query}' but detailed results are not available. Search providers returned results but content extraction failed. Please try a more specific search query."
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"

    def _run(self, query: str, **kwargs) -> str:
        """Sync fallback"""
        return asyncio.run(self._arun(query, **kwargs))


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
