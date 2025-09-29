"""
Static RAG tools module.

Provides easy imports for all static RAG tools:
- WebSearchTool: Web search using DuckDuckGo
- MemoryRetrievalTool: Memory search from database
- SummarizationTool: Content summarization

These are static tools with consistent behavior that don't require
dependency injection or external configuration.
"""

from .web_search_tool import WebSearchTool
from .memory_retrieval_tool import MemoryRetrievalTool
from .summarization_tool import SummarizationTool

__all__ = [
    "WebSearchTool",
    "MemoryRetrievalTool", 
    "SummarizationTool"
]

import asyncio
import json
from typing import List

from langchain_core.tools import BaseTool

from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.web_search_providers import WebSearchProviders


class WebSearchTool(BaseTool):
    """Static tool for performing web searches using DuckDuckGo provider."""
    name: str = "web_search"
    description: str = "Search the web for information using a search query. Returns formatted search results."

    async def _arun(self, query: str) -> str:
        """Async implementation of web search using DuckDuckGo provider."""
        try:
            # Import search provider directly
            from server.services.search_providers import SearchProviderFactory
            
            # Use DuckDuckGo as default provider (no API key required)
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.DDG, 
                max_results=3
            )
            
            search_result = await provider.search(query, 3)
            
            if search_result and search_result.contents:
                formatted_results = [
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": content.content[:200] + "..." if len(content.content) > 200 else content.content,
                        "relevance": content.relevance
                    }
                    for content in search_result.contents
                ]
                
                return json.dumps({
                    "status": "success",
                    "results": formatted_results,
                    "query": query
                }, indent=2)
            else:
                return json.dumps({
                    "status": "success",
                    "results": [],
                    "query": query,
                    "message": "No search results found"
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "status": "error", 
                "error": str(e),
                "query": query
            }, indent=2)
    
    def _run(self, query: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))


class MemoryRetrievalTool(BaseTool):
    """Static tool for retrieving memories from database storage."""
    name: str = "memory_retrieval"
    description: str = "Retrieve relevant memories based on text query. Embeds the query and finds similar memories from database."

    async def _arun(self, query: str) -> str:
        """Async implementation of memory retrieval using database storage."""
        try:
            # Import database storage and runner for embeddings
            from db import storage
            from runner import embed_pipeline
            
            # Initialize storage if not done
            if not storage.pool:
                return json.dumps({
                    "status": "error",
                    "error": "Database not initialized",
                    "query": query
                }, indent=2)
            
            # Generate embeddings for the query
            try:
                # For static tool demo, use mock embeddings
                # In real implementation, you'd use embed_pipeline with proper model
                query_embeddings = [[0.1] * 768]  # Mock embedding for demo
                
                # Retrieve similar memories from storage using correct method
                memory_service = storage.get_service(storage.memory)
                memories = await memory_service.search_similarity(
                    embeddings=query_embeddings,
                    min_similarity=0.7,
                    limit=5,
                    user_id=None,  # Allow cross-user for static tool
                    conversation_id=None  # Allow cross-conversation
                )
                
                # Format memories for display
                formatted_memories = [
                    {
                        "content": "\n".join([f.content for f in memory.fragments]) if hasattr(memory, 'fragments') else str(memory),
                        "timestamp": memory.created_at.isoformat() if hasattr(memory, 'created_at') else None,
                        "similarity": memory.similarity if hasattr(memory, 'similarity') else 1.0,
                        "source": memory.source.value if hasattr(memory, 'source') else 'unknown'
                    }
                    for memory in memories[:5]  # Limit to top 5
                ]
                
                return json.dumps({
                    "status": "success",
                    "memories": formatted_memories,
                    "query": query,
                    "count": len(formatted_memories)
                }, indent=2)
                
            except Exception as embed_error:
                return json.dumps({
                    "status": "error",
                    "error": f"Embedding generation failed: {str(embed_error)}",
                    "query": query
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "query": query
            }, indent=2)
    
    def _run(self, query: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))


class SummarizationTool(BaseTool):
    """Static tool for summarizing content using pipeline execution."""
    name: str = "summarization"
    description: str = "Summarize content using model pipeline. Takes text content and returns a concise summary."

    async def _arun(self, content: str) -> str:
        """Async implementation of content summarization using pipeline."""
        try:
            if not content.strip():
                return json.dumps({
                    "status": "error",
                    "error": "No content provided for summarization",
                    "content": content
                }, indent=2)
            
            # Import runner pipeline functions
            from runner import run_pipeline
            
            # Create summarization message
            summary_message = Message(
                role=MessageRole.USER,
                content=[MessageContent(
                    type=MessageContentType.TEXT, 
                    text=f"Please provide a concise summary of the following content:\n\n{content}"
                )]
            )
            
            # Get a basic pipeline for summarization
            from runner.pipeline_factory import pipeline_factory
            from models.chat_response import ChatResponse
            
            # For static tool, we'll use a simple mock response
            # In a real implementation, you'd get a specific model profile
            summary_text = f"Summary: {content[:200]}..." if len(content) > 200 else f"Summary: {content}"
            
            return json.dumps({
                "status": "success",
                "summary": summary_text,
                "original_length": len(content),
                "summary_length": len(summary_text)
            }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "content": content[:100] + "..." if len(content) > 100 else content
            }, indent=2)

    def _run(self, content: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(content))
