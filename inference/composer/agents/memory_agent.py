"""
Memory Agent for semantic memory storage and retrieval.
Provides core business logic for memory operations and similarity search.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from models import Memory
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class MemoryAgent:
    """
    Memory Agent for semantic memory operations with embedding-based retrieval.
    
    Provides core business logic for storing, retrieving, and managing conversational memories
    using embedding-based similarity search. Integrates with the database memory storage layer.
    """

    def __init__(self):
        """Initialize memory agent."""
        self.logger = composer_logger.logger.bind(component="MemoryAgent")

    async def store_memories(
        self,
        user_id: str,
        conversation_id: int,
        messages: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Store conversation messages as memories with their embeddings.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            messages: List of message data to store
            embeddings: Corresponding embeddings for each message
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            from db import storage  # pylint: disable=import-outside-toplevel
            
            self.logger.info(
                "Storing memories",
                user_id=user_id,
                conversation_id=conversation_id,
                message_count=len(messages),
                embedding_count=len(embeddings)
            )

            # Validate inputs
            if len(messages) != len(embeddings):
                raise NodeExecutionError("Message count must match embedding count")

            # Store each message-embedding pair
            for message_data, embedding in zip(messages, embeddings):
                role = message_data.get('role', 'user')
                source_id = message_data.get('id', conversation_id)  # Use message ID or conversation ID
                
                await storage.get_service(storage.memory).store_memory(
                    user_id=user_id,
                    source="message",  # Source type
                    role=role,
                    source_id=source_id,
                    embeddings=[embedding]  # Store as list
                )

            self.logger.info(
                "Successfully stored memories",
                user_id=user_id,
                conversation_id=conversation_id,
                stored_count=len(messages)
            )
            return True

        except Exception as e:
            self.logger.error(
                "Memory storage failed",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e)
            )
            raise NodeExecutionError(f"Memory storage failed: {e}") from e

    async def search_memories(
        self,
        embeddings: List[List[float]],
        user_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        min_similarity: float = 0.7,
        limit: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Memory]:
        """
        Search for semantically similar memories using embedding vectors.
        
        Args:
            embeddings: Query embedding vectors
            user_id: Optional user ID filter
            conversation_id: Optional conversation ID filter
            min_similarity: Minimum similarity threshold
            limit: Maximum number of results
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of similar memories
        """
        try:
            from db import storage  # pylint: disable=import-outside-toplevel
            
            self.logger.info(
                "Searching memories",
                user_id=user_id,
                conversation_id=conversation_id,
                query_embeddings=len(embeddings),
                min_similarity=min_similarity,
                limit=limit
            )

            # Perform similarity search
            memories = await storage.get_service(storage.memory).search_similarity(
                embeddings=embeddings,
                min_similarity=min_similarity,
                limit=limit,
                user_id=user_id,
                conversation_id=conversation_id,
                start_date=start_date,
                end_date=end_date
            )

            self.logger.info(
                "Memory search completed",
                user_id=user_id,
                results_count=len(memories),
                min_similarity=min_similarity
            )

            return memories

        except Exception as e:
            self.logger.error(
                "Memory search failed",
                user_id=user_id,
                error=str(e),
                min_similarity=min_similarity
            )
            raise NodeExecutionError(f"Memory search failed: {e}") from e

    async def get_memory_context(
        self,
        query_embeddings: List[List[float]],
        user_id: str,
        conversation_id: Optional[int] = None,
        max_memories: int = 5
    ) -> str:
        """
        Get formatted memory context for augmenting LLM prompts.
        
        Args:
            query_embeddings: Embeddings of the current query
            user_id: User identifier
            conversation_id: Optional conversation context
            max_memories: Maximum number of memories to include
            
        Returns:
            Formatted memory context string
        """
        try:
            # Get user configuration for memory settings
            from db import storage  # pylint: disable=import-outside-toplevel
            
            user_config = await storage.get_service(storage.user_config).get_user_config(user_id)
            
            # Use user's memory configuration
            similarity_threshold = user_config.memory.similarity_threshold if hasattr(user_config, 'memory') else 0.7
            cross_conversation = user_config.memory.enable_cross_conversation if hasattr(user_config, 'memory') else False
            
            # Search for relevant memories
            memories = await self.search_memories(
                embeddings=query_embeddings,
                user_id=user_id,
                conversation_id=None if cross_conversation else conversation_id,
                min_similarity=similarity_threshold,
                limit=max_memories
            )

            if not memories:
                return ""

            # Format memories as context
            context_parts = []
            for memory in memories:
                # Format memory fragments
                fragment_texts = []
                for fragment in memory.fragments:
                    role_name = fragment.role.value if hasattr(fragment.role, 'value') else str(fragment.role)
                    fragment_texts.append(f"{role_name}: {fragment.content}")
                
                # Add memory with timestamp
                memory_text = "\n".join(fragment_texts)
                context_parts.append(f"({memory.created_at}) {memory_text}")

            context = "\n\n".join(context_parts)
            
            self.logger.info(
                "Generated memory context",
                user_id=user_id,
                memory_count=len(memories),
                context_length=len(context)
            )

            return context

        except Exception as e:
            self.logger.error(
                "Memory context generation failed",
                user_id=user_id,
                error=str(e)
            )
            # Return empty context on failure - don't break the workflow
            return ""

    async def delete_memories(
        self,
        user_id: str,
        conversation_id: Optional[int] = None,
        memory_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Delete memories by user, conversation, or specific memory IDs.
        
        Args:
            user_id: User identifier
            conversation_id: Optional conversation to delete memories from
            memory_ids: Optional specific memory IDs to delete
            
        Returns:
            True if deletion successful
        """
        try:
            from db import storage  # pylint: disable=import-outside-toplevel
            
            if memory_ids:
                # Delete specific memories
                for memory_id in memory_ids:
                    await storage.get_service(storage.memory).delete_memory(memory_id, user_id)
                self.logger.info(f"Deleted {len(memory_ids)} specific memories", user_id=user_id)
            else:
                # Delete all user memories
                await storage.get_service(storage.memory).delete_all_user_memories(user_id)
                self.logger.info("Deleted all user memories", user_id=user_id)

            return True

        except Exception as e:
            self.logger.error(
                "Memory deletion failed",
                user_id=user_id,
                error=str(e)
            )
            raise NodeExecutionError(f"Memory deletion failed: {e}") from e

    def format_memories_for_display(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Format memories for user-friendly display.
        
        Args:
            memories: List of memories to format
            
        Returns:
            Formatted memory data for display
        """
        formatted_memories = []
        
        for memory in memories:
            formatted_memory = {
                "id": memory.source_id,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "similarity": memory.similarity,
                "source": memory.source.value if hasattr(memory.source, 'value') else str(memory.source),
                "conversation_id": memory.conversation_id,
                "fragments": []
            }
            
            for fragment in memory.fragments:
                formatted_fragment = {
                    "id": fragment.id,
                    "role": fragment.role.value if hasattr(fragment.role, 'value') else str(fragment.role),
                    "content": fragment.content[:200] + "..." if len(fragment.content) > 200 else fragment.content
                }
                formatted_memory["fragments"].append(formatted_fragment)
            
            formatted_memories.append(formatted_memory)
        
        return {
            "memories": formatted_memories,
            "total_count": len(memories),
            "summary": f"Retrieved {len(memories)} relevant memories"
        }