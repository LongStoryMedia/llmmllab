"""
Embedding Generator Node for LangGraph workflows.
Generates embeddings from text using the embedding agent.
"""

from typing import TYPE_CHECKING

from composer.graph.state import WorkflowState
from composer.core.errors import NodeExecutionError
from composer.utils.extraction import extract_content_from_langchain_message
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from composer.agents.embedding_agent import EmbeddingAgent


class EmbeddingGeneratorNode:
    """
    Node for generating embeddings from text content in workflow state.

    Takes text inputs from workflow state and produces embeddings using
    the embedding agent, storing results back in state for other nodes.
    """

    def __init__(
        self,
        embedding_agent: "EmbeddingAgent",
        model_name: str,
    ):
        """
        Initialize embedding generator node.

        Args:
            embedding_agent: Required EmbeddingAgent instance
            model_name: Required specific embedding model to use
        """
        self.agent = embedding_agent
        self.model_name = model_name
        self.logger = llmmllogger.logger.bind(component="EmbeddingGeneratorNode")

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
            embeddings = await self.agent.generate_embeddings(texts=texts)

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
