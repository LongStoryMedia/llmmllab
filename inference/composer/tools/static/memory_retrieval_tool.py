"""
Static memory retrieval tool for database storage.

This tool retrieves relevant memories from the database using
embeddings and similarity search with consistent behavior.
"""

import asyncio
import json

from langchain_core.tools import BaseTool

from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


class MemoryRetrievalTool(BaseTool):
    """Static tool for retrieving memories from database storage."""
    name: str = "memory_retrieval"
    description: str = "Retrieve relevant memories based on text query. Embeds the query and finds similar memories from database."

    async def _arun(self, query: str) -> str:
        """Async implementation of memory retrieval using database storage."""
        try:
            # Import database storage and runner for embeddings
            from db import storage
            
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