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