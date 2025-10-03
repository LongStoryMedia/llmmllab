"""
Embedding Agent for generating semantic embeddings from text input.
Provides core business logic for embedding generation and vector operations.
"""

from typing import List, Optional, Dict, Any

from models import EmbeddingReq, EmbeddingResponse, ModelProfileType
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class EmbeddingAgent:
    """
    Embedding Agent for text-to-vector conversion with model profile integration.
    
    Provides core business logic for embedding generation using configured embedding models.
    Supports both single text and batch text embedding generation.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize embedding agent.
        
        Args:
            pipeline_factory: Factory for creating embedding pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger.bind(component="EmbeddingAgent")

    async def generate_embeddings(
        self, 
        texts: List[str], 
        user_id: str,
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts using configured embedding model.
        
        Args:
            texts: List of text strings to generate embeddings for
            user_id: User identifier for model profile retrieval
            model_name: Optional specific model name (overrides profile)
            
        Returns:
            List of embedding vectors (one per input text)
        """
        try:
            self.logger.info(
                "Generating embeddings",
                user_id=user_id,
                text_count=len(texts),
                model_name=model_name
            )

            # Get model profile for embeddings
            try:
                from utils.model_profile import get_model_profile
                
                # Use embedding model profile or fallback to provided model
                if not model_name:
                    model_profile = await get_model_profile(user_id, ModelProfileType.EMBEDDING)
                    model_name = model_profile.model_name if model_profile else "nomic-embed-text"
                    
            except Exception as e:
                self.logger.warning(f"Could not get model profile, using default: {e}")
                model_name = model_name or "nomic-embed-text"

            # Create embedding request
            embedding_req = EmbeddingReq(
                model=model_name,
                input=texts,
                truncate=True  # Ensure we don't exceed model context limits
            )

            # Get embedding pipeline and generate embeddings
            if self.pipeline_factory:
                pipeline = await self.pipeline_factory.get_pipeline(
                    model_name, EmbeddingResponse, streaming=False
                )
                
                # Execute embedding generation
                response = await pipeline.invoke({"request": embedding_req})
                
                if hasattr(response, 'embeddings') and response.embeddings:
                    embeddings = response.embeddings
                    self.logger.info(
                        "Successfully generated embeddings",
                        user_id=user_id,
                        embedding_count=len(embeddings),
                        embedding_dimensions=len(embeddings[0]) if embeddings else 0
                    )
                    return embeddings
                else:
                    raise NodeExecutionError("No embeddings returned from pipeline")
            else:
                raise NodeExecutionError("Pipeline factory not available for embedding generation")

        except Exception as e:
            self.logger.error(
                "Embedding generation failed",
                user_id=user_id,
                error=str(e),
                text_count=len(texts)
            )
            raise NodeExecutionError(f"Embedding generation failed: {e}") from e

    async def generate_single_embedding(
        self, 
        text: str, 
        user_id: str,
        model_name: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for single text input.
        
        Args:
            text: Text string to generate embedding for
            user_id: User identifier for model profile retrieval
            model_name: Optional specific model name
            
        Returns:
            Single embedding vector
        """
        embeddings = await self.generate_embeddings([text], user_id, model_name)
        return embeddings[0] if embeddings else []

    async def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate that embeddings are properly formatted.
        
        Args:
            embeddings: List of embedding vectors to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not embeddings or not isinstance(embeddings, list):
            return False
            
        if not all(isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb) for emb in embeddings):
            return False
            
        # Check that all embeddings have the same dimension
        if len(set(len(emb) for emb in embeddings)) > 1:
            return False
            
        return True