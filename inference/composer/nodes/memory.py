"""
Memory Node for LangGraph workflows.
Wraps MemoryAgent to provide memory operations within workflows.
"""

from composer.agents.memory_agent import MemoryAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class MemoryNode:
    """
    LangGraph node wrapper for memory operations.
    
    Provides memory storage and retrieval capabilities within LangGraph workflows,
    including semantic search and context augmentation.
    """

    def __init__(self):
        """Initialize memory node."""
        self.agent = MemoryAgent()
        self.logger = composer_logger.logger.bind(component="MemoryNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Perform memory retrieval and augment state with relevant memories.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with memory context
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for memory operations")

            # Get query embeddings from state
            query_embedding = state.execution_metadata.get("query_embedding")
            if not query_embedding:
                self.logger.info("No query embedding found, skipping memory retrieval")
                return state

            self.logger.info(
                "Performing memory retrieval",
                user_id=user_id,
                has_query_embedding=bool(query_embedding)
            )

            # Get conversation ID if available
            conversation_id = state.execution_metadata.get("conversation_id")

            # Search for relevant memories
            memory_context = await self.agent.get_memory_context(
                query_embeddings=[query_embedding],
                user_id=user_id,
                conversation_id=conversation_id,
                max_memories=5
            )

            # Store memory context in state
            if memory_context:
                state.execution_metadata["memory_context"] = memory_context
                state.execution_metadata["has_memory_context"] = True
                
                self.logger.info(
                    "Retrieved memory context",
                    user_id=user_id,
                    context_length=len(memory_context)
                )
            else:
                state.execution_metadata["has_memory_context"] = False
                self.logger.info("No relevant memories found", user_id=user_id)

            return state

        except Exception as e:
            self.logger.error(
                "Memory node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Memory retrieval failed: {str(e)}")
            state.execution_metadata["has_memory_context"] = False
            return state

    async def store_memories(self, state: WorkflowState) -> WorkflowState:
        """
        Store current conversation messages as memories.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for memory storage")

            # Get embeddings and texts from state
            embeddings = state.execution_metadata.get("embeddings", [])
            texts = state.execution_metadata.get("embedding_texts", [])
            
            if not embeddings or not texts:
                self.logger.info("No embeddings or texts found for memory storage")
                return state

            # Get conversation ID
            conversation_id = state.execution_metadata.get("conversation_id", 1)

            # Prepare message data for storage
            messages = []
            for i, text in enumerate(texts):
                # Determine role from message type or position
                role = "user" if i % 2 == 0 else "assistant"  # Simple alternating pattern
                messages.append({
                    "id": conversation_id + i,
                    "role": role,
                    "content": text
                })

            self.logger.info(
                "Storing memories",
                user_id=user_id,
                message_count=len(messages),
                conversation_id=conversation_id
            )

            # Store memories using the agent
            success = await self.agent.store_memories(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=messages,
                embeddings=embeddings
            )

            state.execution_metadata["memories_stored"] = success
            
            if success:
                self.logger.info("Successfully stored memories", user_id=user_id)
            else:
                self.logger.warning("Memory storage failed", user_id=user_id)

            return state

        except Exception as e:
            self.logger.error(
                "Memory storage failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Memory storage failed: {str(e)}")
            state.execution_metadata["memories_stored"] = False
            return state