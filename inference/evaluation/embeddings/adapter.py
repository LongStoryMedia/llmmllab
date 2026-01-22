"""
Adapter for the Nomic Embed Text model to capture and return embeddings.

This adapter uses the runner package to access embedding models through PipelineFactory.
"""

import datetime
import logging
from typing import List, Optional, Union

from numpy import cov

from runner.pipeline_factory import pipeline_factory
from models import Message, MessageContent, MessageContentType, ChatReq
from models.message_role import MessageRole


class EmbeddingModelAdapter:
    """Adapter class to handle embedding operations using embedding models via the pipeline factory."""

    def __init__(self):
        """Initialize the embedding model adapter."""
        self.logger = logging.getLogger(__name__)

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model_path: str,
        is_query: Optional[bool] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts using the runner.

        Args:
            texts: The text or list of texts to embed
            model_path: Path or ID of the embedding model
            is_query: Whether the text is a query (True), document (False), or auto-detect (None)
            matryoshka_dim: Optional dimension for Matryoshka embedding truncation (256, 512, or 768)

        Returns:
            A list of embeddings for each input text
        """
        # Standardize input to list format
        if isinstance(texts, str):
            texts = [texts]

        self.logger.info(
            f"Generating embeddings for {len(texts)} texts using model: {model_path}"
        )

        # Process embeddings for all texts
        all_embeddings = []

        try:
            for text in texts:
                # Apply appropriate prefix based on is_query setting
                has_prefix = text.startswith("search_document: ") or text.startswith(
                    "search_query: "
                )

                if not has_prefix:
                    if is_query is True:
                        text = f"search_query: {text}"
                    elif is_query is False:
                        text = f"search_document: {text}"
                    else:
                        # Auto-detect based on text length
                        prefix = (
                            "search_query: " if len(text) < 100 else "search_document: "
                        )
                        text = f"{prefix}{text}"

                # Create message with all required fields
                message = Message(
                    role=MessageRole.USER,
                    content=[MessageContent(type=MessageContentType.TEXT, text=text)],
                    id=None,  # Optional field
                    created_at=datetime.datetime.now(
                        tz=datetime.timezone.utc
                    ),  # Set current timestamp with timezone
                )

                # Create request with all required fields
                req = ChatReq(
                    messages=[message],
                    conversation_id=999,
                    stream=True,  # Required field
                )

                # Execute the request using PipelineFactory
                try:
                    # Get pipeline for the model
                    pipeline, _ = pipeline_factory.get_pipeline(model_path)

                    # Generate embeddings using the pipeline
                    responses = list(pipeline.run(req))

                    # Extract embedding from response
                    embedding = self._extract_embedding_from_response(responses)
                    if embedding:
                        all_embeddings.append(embedding)
                    else:
                        self.logger.error(
                            f"Could not extract embedding for text: {text[:50]}..."
                        )
                        # Return empty embedding as fallback
                        dim = matryoshka_dim or 768
                        all_embeddings.append([0.0] * dim)

                except Exception as e:
                    self.logger.error(f"Error generating embedding: {str(e)}")
                    # Return empty embedding as fallback
                    dim = matryoshka_dim or 768
                    all_embeddings.append([0.0] * dim)

            return all_embeddings

        except Exception as e:
            self.logger.error(f"Error in generate_embeddings: {str(e)}")
            # Return empty embeddings as fallback
            dim = matryoshka_dim or 768
            return [[0.0] * dim] * len(texts)

    def _extract_embedding_from_response(self, responses):
        """
        Extract embedding from model responses.

        Args:
            responses: List of responses from the model

        Returns:
            Embedding vector as list of floats or None if not found
        """
        # Extract embedding from the context field of ChatResponse
        for response in responses:
            if hasattr(response, "context") and isinstance(response.context, list):
                return response.context

        self.logger.warning("Could not find embedding in pipeline response")
        return None
