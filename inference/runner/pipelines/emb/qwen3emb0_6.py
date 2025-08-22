"""
Embedding pipeline for Qwen3-Embedding-0.6B model.

This pipeline implements the Qwen3-Embedding-0.6B model for generating
text embeddings. It's designed for semantic search, RAG, and other
text similarity applications.

Requirements:
- Supports multiple languages
- 0.6B parameters for efficient inference
- Context length up to 8192 tokens
- 1024-dimensional embeddings
- Optimized for semantic similarity tasks

For more details see: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
"""

import datetime
import logging
import os
import numpy as np
import torch
from typing import Any, Dict, List, Generator, Optional, Union
from llama_cpp import Llama  # type: ignore # pylint: disable=E0401

from models import (
    Model,
    Message,
    ChatResponse,
    ModelParameters,
    MessageContent,
    MessageContentType,
    ChatReq,
)
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from ..base_pipeline import BasePipeline


class Qwen3EmbeddingPipe(BasePipeline):
    """
    Pipeline for running text embeddings with Qwen3-Embedding-0.6B model.

    This pipeline supports the Qwen/Qwen3-Embedding-0.6B model in GGUF format.

    Key features of the model:
    - Multilingual text embedding model
    - 0.6B parameters for efficient inference
    - Context length up to 8192 tokens
    - 1024-dimensional embeddings
    - Optimized for semantic similarity and retrieval tasks
    - No special prefixes required (unlike some other embedding models)
    """

    # Class-level attributes
    model: Any = None

    def __init__(self, model_definition: Model):
        """Initialize the Qwen3 Embedding pipeline."""
        # Call base class initialization first
        super().__init__(model_definition)

        # Set up logger
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing Qwen3EmbeddingPipe")
        self._logger.info(f"Model definition: {self.model_def.json()}")

        # Ensure model details for GGUF are provided
        if not (self.model_def.details and self.model_def.model):
            raise ValueError(
                "Model definition for Qwen3EmbeddingPipe must include model path details."
            )

        # Log model info for debugging
        self._logger.info(f"Model ID: {self.model_def.id}")

        # Get the GGUF file path
        gguf = (
            model_definition.details.gguf_file
            if hasattr(model_definition.details, "gguf_file")
            and model_definition.details.gguf_file
            else model_definition.model
        )

        # Check file size
        file_size = os.path.getsize(gguf)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf}"
            )

        # Log the file path we're actually using
        self._logger.info(
            f"Using GGUF file path: {gguf} (size: {file_size/1_000_000:.2f} MB)"
        )

        # Load the GGUF model using llama-cpp-python for embedding
        try:
            self.model = Llama(
                model_path=gguf,
                n_ctx=8192,  # Maximum context length for Qwen3-Embedding
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=4,
                use_mlock=True,
                embedding=True,  # Enable embedding mode
                verbose=True,
                n_batch=512,
                offload_kqv=True,
            )

            self._logger.info(
                f"Qwen3 Embedding model '{self.model_def.name}' loaded successfully."
            )
        except Exception as e:
            self._logger.error(
                f"Error initializing {self.__class__.__name__}: {str(e)}"
            )
            raise

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process input messages to generate embeddings for text.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            Generator[ChatResponse, Any, None]: Yields ChatResponse objects with embeddings in context.
        """
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        load_time = 0.0

        try:
            # Extract text from messages in the request
            inputs = []
            for message in req.messages:
                if not message.content:
                    continue
                for content in message.content:
                    if (
                        hasattr(content, "type")
                        and hasattr(content, "text")
                        and content.text
                    ):
                        # Qwen3-Embedding doesn't require special prefixes
                        # Use text as-is for maximum flexibility
                        inputs.append(content.text)

            if not inputs:
                self._logger.warning("No text inputs found in messages")
                inputs = [""]  # Add empty input to avoid errors

            self._logger.info(f"Running embedding model with {len(inputs)} inputs")

            embeddings = []

            # Check if custom embedding dimension is specified in parameters
            # Qwen3-Embedding-0.6B outputs 1024-dimensional embeddings by default
            target_dim = None
            if req.options and hasattr(req.options, "num_ctx") and req.options.num_ctx:
                # Use num_ctx as a way to specify embedding dimension if needed
                # This is a bit of a hack, but provides flexibility
                if req.options.num_ctx in [256, 512, 768, 1024]:
                    target_dim = req.options.num_ctx

            for text_input in inputs:
                if self.model and hasattr(self.model, "embed"):
                    # Generate embedding for each input text
                    embedding_result = self.model.embed(text_input)
                    self._logger.debug(f"Raw embedding type: {type(embedding_result)}")

                    # Extract embeddings array from model output
                    embedding = embedding_result
                    self._logger.debug(f"Using raw embedding: {type(embedding)}")

                    # Convert to numpy array for processing
                    embedding_array = np.asarray(embedding)

                    # Apply dimension truncation if specified
                    if target_dim and target_dim < len(embedding_array):
                        self._logger.info(
                            f"Truncating embedding from {len(embedding_array)} to {target_dim} dimensions"
                        )
                        embedding_array = embedding_array[:target_dim]

                    # Normalize embedding for better similarity computation
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    else:
                        self._logger.warning(
                            "Zero-norm embedding detected, skipping normalization"
                        )

                    embeddings.append(embedding_array.tolist())
                else:
                    self._logger.error(
                        "Model not properly initialized or doesn't support embedding"
                    )
                    embedding_dim = target_dim if target_dim else 1024
                    embeddings.append([0.0] * embedding_dim)

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Log embedding information
            self._logger.debug(
                f"Final embeddings type: {type(embeddings)}, length: {len(embeddings) if embeddings else 0}"
            )
            if embeddings and len(embeddings) > 0:
                self._logger.info(
                    f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions each"
                )

            # Create a ChatResponse object to return
            response = ChatResponse(
                message=Message(
                    id=None,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Generated {len(embeddings)} embeddings with {len(embeddings[0]) if embeddings else 0} dimensions each",
                            url=None,
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    created_at=start_time,
                ),
                done=True,
                finish_reason="stop",
                context=embeddings,  # Return embeddings in context
                total_duration=total_duration,
                load_duration=load_time,
                prompt_eval_count=len(inputs),
                prompt_eval_duration=0,
                eval_count=len(inputs),
                eval_duration=total_duration,
                created_at=start_time,
                model=str(self.model_def.id),
            )
            yield response

        except (RuntimeError, ValueError) as e:
            self._logger.error(f"Error running Qwen3 Embedding model: {str(e)}")
            error_response = ChatResponse(
                message=Message(
                    id=None,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error generating embeddings: {str(e)}",
                            url=None,
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    created_at=start_time,
                ),
                done=True,
                finish_reason="error",
                total_duration=(
                    datetime.datetime.now(tz=datetime.timezone.utc) - start_time
                ).total_seconds()
                * 1000,
                load_duration=load_time,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=0,
                created_at=start_time,
                model=str(self.model_def.id),
            )
            yield error_response
            raise

    async def emb(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: The text or list of texts to embed
            normalize: Whether to normalize the embeddings (default: True)
            truncate_dim: Optional dimension to truncate embeddings to (256, 512, 768, 1024)

        Returns:
            A list of embeddings for each input text
        """
        # Standardize input to list format
        if isinstance(texts, str):
            texts = [texts]

        try:
            all_embeddings = []

            for text in texts:
                # Create message with required fields
                message = Message(
                    role=MessageRole.USER,
                    content=[MessageContent(type=MessageContentType.TEXT, text=text)],
                    id=None,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                )

                # Create request with optional dimension specification
                params = None
                if truncate_dim:
                    params = ModelParameters(num_ctx=truncate_dim)

                req = ChatReq(
                    messages=[message],
                    conversation_id=999,
                    stream=True,
                    options=params,
                )

                # Execute the request
                try:
                    responses = list(self.run(req))

                    # Extract embedding from response
                    embedding = self._extract_embedding_from_response(responses)
                    if embedding and len(embedding) > 0:
                        # Apply normalization if requested
                        if normalize and isinstance(embedding[0], (list, np.ndarray)):
                            # embedding is a list of embeddings, take the first one
                            emb_array = np.asarray(embedding[0])
                            norm = np.linalg.norm(emb_array)
                            if norm > 0:
                                embedding[0] = (emb_array / norm).tolist()
                        all_embeddings.extend(embedding)
                    else:
                        # Return empty embedding as fallback
                        dim = truncate_dim or 1024
                        all_embeddings.append([0.0] * dim)

                except Exception as e:
                    self._logger.error(f"Error processing text embedding: {e}")
                    # Return empty embedding as fallback
                    dim = truncate_dim or 1024
                    all_embeddings.append([0.0] * dim)

            return all_embeddings

        except Exception as e:
            self._logger.error(f"Error in emb method: {e}")
            # Return empty embeddings as fallback
            dim = truncate_dim or 1024
            return [[0.0] * dim] * len(texts)

    async def similarity(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[tuple[int, float, str]]:
        """
        Compute similarity between a query and multiple documents.

        Args:
            query: The query text
            documents: List of document texts
            top_k: Optional limit on number of results to return

        Returns:
            List of tuples: (original_index, similarity_score, document_text)
            Sorted by similarity score (descending)
        """
        try:
            # Generate embeddings for query and documents
            all_texts = [query] + documents
            embeddings = await self.emb(all_texts, normalize=True)

            if len(embeddings) < len(all_texts):
                self._logger.error("Not enough embeddings generated")
                return []

            # Extract query embedding and document embeddings
            query_embedding = np.array(embeddings[0])
            doc_embeddings = [np.array(emb) for emb in embeddings[1:]]

            # Compute cosine similarities
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                try:
                    # Cosine similarity = dot product of normalized vectors
                    similarity = np.dot(query_embedding, doc_embedding)
                    similarities.append((i, float(similarity), documents[i]))
                except Exception as e:
                    self._logger.warning(
                        f"Error computing similarity for document {i}: {e}"
                    )
                    similarities.append((i, 0.0, documents[i]))

            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit if specified
            if top_k and len(similarities) > top_k:
                similarities = similarities[:top_k]

            return similarities

        except Exception as e:
            self._logger.error(f"Error computing similarities: {e}")
            # Return documents with neutral scores as fallback
            return [
                (i, 0.0, doc)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]

    def batch_embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches for efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            normalize: Whether to normalize the embeddings
            truncate_dim: Optional dimension to truncate embeddings to

        Returns:
            List of embeddings for each input text
        """
        all_embeddings = []

        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                self._logger.info(
                    f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts"
                )

                # Create batch message with all texts
                batch_content = []
                for text in batch_texts:
                    batch_content.append(
                        MessageContent(type=MessageContentType.TEXT, text=text)
                    )

                message = Message(
                    role=MessageRole.USER,
                    content=batch_content,
                    id=None,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                )

                # Create request with optional dimension specification
                params = None
                if truncate_dim:
                    params = ModelParameters(num_ctx=truncate_dim)

                req = ChatReq(
                    messages=[message],
                    conversation_id=999,
                    stream=True,
                    options=params,
                )

                # Execute the batch request
                try:
                    responses = list(self.run(req))
                    batch_embeddings = self._extract_embedding_from_response(responses)

                    if batch_embeddings:
                        # Apply normalization if requested
                        if normalize:
                            normalized_embeddings = []
                            for embedding in batch_embeddings:
                                emb_array = np.asarray(embedding)
                                norm = np.linalg.norm(emb_array)
                                if norm > 0:
                                    normalized_embeddings.append(
                                        (emb_array / norm).tolist()
                                    )
                                else:
                                    normalized_embeddings.append(embedding)
                            all_embeddings.extend(normalized_embeddings)
                        else:
                            all_embeddings.extend(batch_embeddings)
                    else:
                        # Add fallback embeddings for failed batch
                        dim = truncate_dim or 1024
                        all_embeddings.extend([[0.0] * dim] * len(batch_texts))

                except Exception as e:
                    self._logger.error(f"Error processing batch: {e}")
                    # Add fallback embeddings for failed batch
                    dim = truncate_dim or 1024
                    all_embeddings.extend([[0.0] * dim] * len(batch_texts))

            return all_embeddings

        except Exception as e:
            self._logger.error(f"Error in batch_embed: {e}")
            # Return fallback embeddings
            dim = truncate_dim or 1024
            return [[0.0] * dim] * len(texts)

    def __del__(self) -> None:
        """Clean up resources used by the Qwen3EmbeddingPipe."""
        try:
            self._logger.info(
                f"Qwen3EmbeddingPipe for {self.model_def.name if hasattr(self, 'model_def') else 'unknown'}: Cleanup initiated"
            )
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AttributeError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error cleaning up Qwen3EmbeddingPipe resources: {str(e)}")
