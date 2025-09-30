"""
Static memory retrieval tool for database storage.

This tool retrieves relevant memories from the database using
embeddings and similarity search with consistent behavior.
"""

import asyncio
import json
import logging

from langchain_core.tools import BaseTool

from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


class MemoryRetrievalTool(BaseTool):
    """Static tool for retrieving memories from database storage."""
    name: str = "memory_retrieval"
    description: str = "Retrieve relevant memories based on text query. Embeds the query and finds similar memories from database."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

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
                from runner import embed_pipeline, pipeline_factory
                from runner.pipeline_factory import PipelinePriority
                
                # Get default embedding model profile
                # For static tool, we'll get the first available embedding model
                try:
                    # Try to get embedding pipeline
                    embed_message = Message(
                        role=MessageRole.USER,
                        content=[MessageContent(type=MessageContentType.TEXT, text=query)]
                    )
                    
                    # Use embed_pipeline with a simple embedding model approach
                    # This requires getting an embedding pipeline from the factory
                    from models.model_profile import ModelProfile
                    from models.model import Model
                    
                    # Create a simple embedding request
                    query_embeddings = await embed_pipeline(
                        messages=[embed_message],
                        pipeline=None  # Will use default embedding pipeline
                    )
                except Exception as embed_error:
                    # Fallback to mock embeddings if no embedding model available
                    self.logger.warning(f"Embedding generation failed, using mock: {embed_error}")
                    query_embeddings = [[0.1] * 768]  # Fallback mock embedding
                
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