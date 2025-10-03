"""
Embedding Node for LangGraph workflows.
Wraps EmbeddingAgent to provide embedding generation capabilities within workflows.
"""

from typing import Optional

from composer.agents.embedding_agent import EmbeddingAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class EmbeddingNode:
    """
    LangGraph node wrapper for embedding generation.
    
    Provides embedding generation capabilities within LangGraph workflows,
    supporting both single and batch text embedding operations.
    """

    def __init__(self, pipeline_factory, model_name: Optional[str] = None):
        """
        Initialize embedding node.
        
        Args:
            pipeline_factory: Factory for creating embedding pipelines
            model_name: Optional default model name for embeddings
        """
        self.agent = EmbeddingAgent(pipeline_factory)
        self.model_name = model_name
        self.logger = composer_logger.logger.bind(component="EmbeddingNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Generate embeddings for messages in workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with embeddings
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for embedding generation")

            # Extract texts from messages
            texts = []
            for message in state.messages:
                if hasattr(message, 'content') and message.content:
                    texts.append(str(message.content))

            if not texts:
                self.logger.info("No texts found for embedding generation")
                return state

            self.logger.info(
                "Generating embeddings for workflow",
                user_id=user_id,
                text_count=len(texts)
            )

            # Generate embeddings using the agent
            embeddings = await self.agent.generate_embeddings(
                texts=texts,
                user_id=user_id,
                model_name=self.model_name
            )

            # Store embeddings in state metadata
            state.execution_metadata["embeddings"] = embeddings
            state.execution_metadata["embedding_texts"] = texts
            state.execution_metadata["embedding_model"] = self.model_name or "default"

            self.logger.info(
                "Successfully generated embeddings",
                user_id=user_id,
                embedding_count=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0
            )

            return state

        except Exception as e:
            self.logger.error(
                "Embedding node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Embedding generation failed: {str(e)}")
            return state

    async def generate_query_embeddings(self, state: WorkflowState) -> WorkflowState:
        """
        Generate embeddings specifically for the latest user query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with query embeddings
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for query embedding generation")

            # Get latest user message
            latest_user_message = None
            for message in reversed(state.messages):
                if hasattr(message, 'type') and message.type == "human":
                    latest_user_message = message
                    break

            if not latest_user_message or not latest_user_message.content:
                self.logger.info("No user query found for embedding generation")
                return state

            query_text = str(latest_user_message.content)

            self.logger.info(
                "Generating query embedding",
                user_id=user_id,
                query_length=len(query_text)
            )

            # Generate single embedding for query
            query_embedding = await self.agent.generate_single_embedding(
                text=query_text,
                user_id=user_id,
                model_name=self.model_name
            )

            # Store query embedding in state metadata
            state.execution_metadata["query_embedding"] = query_embedding
            state.execution_metadata["query_text"] = query_text

            self.logger.info(
                "Successfully generated query embedding",
                user_id=user_id,
                embedding_dimensions=len(query_embedding)
            )

            return state

        except Exception as e:
            self.logger.error(
                "Query embedding generation failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Query embedding generation failed: {str(e)}")
            return state


class MemoryNode:
    """
    LangGraph node wrapper for memory operations.
    
    Provides memory storage and retrieval capabilities within LangGraph workflows,
    including semantic search and context augmentation.
    """

    def __init__(self):
        """Initialize memory node."""
        from composer.agents.memory_agent import MemoryAgent
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