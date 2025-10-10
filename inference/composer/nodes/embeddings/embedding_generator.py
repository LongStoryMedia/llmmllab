"""
Embedding Generator Node for LangGraph workflows.
Generates embeddings from text using the embedding agent.
"""

from typing import Optional

from runner import PipelineFactory
from composer.agents.embedding_agent import EmbeddingAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from composer.utils.extraction import extract_content_from_langchain_message


class EmbeddingGeneratorNode:
    """
    Node for generating embeddings from text content in workflow state.

    Takes text inputs from workflow state and produces embeddings using
    the embedding agent, storing results back in state for other nodes.
    """

    def __init__(self, pipeline_factory: PipelineFactory, model_name: Optional[str] = None):
        """
        Initialize embedding generator node.

        Args:
            pipeline_factory: Factory for creating embedding pipelines
            model_name: Optional specific embedding model to use
        """
        self.agent = EmbeddingAgent(pipeline_factory)
        self.model_name = model_name
        self.logger = composer_logger.logger.bind(component="EmbeddingGeneratorNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Generate embeddings for all messages in workflow state.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with embeddings in execution_metadata
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for embedding generation")

            # Extract texts from messages
            texts = []
            for message in state.messages:
                if hasattr(message, "content") and message.content:
                    content = extract_content_from_langchain_message(message)
                    if content and content.strip():
                        texts.append(content)

            if not texts:
                self.logger.info("No texts found for embedding generation")
                return state

            self.logger.info(
                "Generating embeddings for messages",
                user_id=user_id,
                text_count=len(texts),
            )

            # Generate embeddings using the agent
            embeddings = await self.agent.generate_embeddings(
                texts=texts, user_id=user_id
            )

            self.logger.info(
                "Successfully generated embeddings",
                user_id=user_id,
                embedding_count=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0,
            )

            return state

        except Exception as e:
            self.logger.error(
                "Embedding generation failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Embedding generation failed: {str(e)}")
            return state

    async def generate_query_embedding(self, state: WorkflowState) -> WorkflowState:
        """
        Generate embedding specifically for the latest user query.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with query embedding in execution_metadata
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError(
                    "User ID required for query embedding generation"
                )

            # Get latest user message
            latest_user_message = None
            for message in reversed(state.messages):
                if hasattr(message, "type") and message.type in ("human", "user"):
                    latest_user_message = message
                    break

            if not latest_user_message or not latest_user_message.content:
                self.logger.info("No user query found for embedding generation")
                return state

            query_text = extract_content_from_langchain_message(latest_user_message)

            self.logger.info(
                "Generating query embedding",
                user_id=user_id,
                query_length=len(query_text),
            )

            # Generate single embedding for query
            query_embedding = await self.agent.generate_single_embedding(
                text=query_text, user_id=user_id
            )

            self.logger.info(
                "Successfully generated query embedding",
                user_id=user_id,
                embedding_dimensions=len(query_embedding),
            )

            return state

        except Exception as e:
            self.logger.error(
                "Query embedding generation failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Query embedding generation failed: {str(e)}")
            return state
