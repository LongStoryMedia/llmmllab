"""
Embedding model pipeline for Nomic Embed Text v2 model.

This pipeline implements the Nomic Embed Text v2 Mixture of Experts (MoE) model for generating
text embeddings. It properly applies the required task instruction prefixes as per the model's
usage guidelines.

Requirements:
- Input must include a task instruction prefix:
  - "search_query: " for queries/questions
  - "search_document: " for documents/content
- Maximum input length is 512 tokens
- The pipeline automatically determines the appropriate prefix based on text length
- Embeddings are normalized by default

For more details see: https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
"""

import datetime
import logging
import os
from typing import Any, List, Generator, Optional, Union, cast
import torch
from llama_cpp import Llama

from models import (
    Model,
    Message,
    ChatResponse,
    MessageContent,
    MessageContentType,
    ChatReq,
    MessageRole,
)
from ..base_pipeline import BasePipeline


class NomicEmbedTextPipe(BasePipeline):
    """
    Pipeline for running text embeddings with Nomic Embed Text v2 model.

    This pipeline supports the nomic-ai/nomic-embed-text-v2-moe model in GGUF format.

    Key features of the model:
    - Multilingual MoE (Mixture of Experts) text embedding model
    - Supports ~100 languages with 768-dimensional embeddings (truncatable to 256)
    - Uses task-specific prefixes: "search_query: " for queries and "search_document: " for documents
    - Maximum sequence length is 512 tokens
    - Model size: 305M parameters (475M total, 305M active during inference)
    - Trained on 1.6B multilingual pairs
    """

    # Class-level attributes
    model: Optional[Llama] = None

    def __init__(self, model_definition: Model):
        """Initialize the Nomic Embed Text pipeline."""
        super().__init__(model_definition)

        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing NomicEmbedTextPipe")

        # Validate model definition
        if not (self.model_def.details and self.model_def.model):
            raise ValueError(
                "Model definition for NomicEmbedTextPipe must include model path details."
            )

        # Get the GGUF file path
        gguf_path = self._get_gguf_path()

        # Validate file
        self._validate_gguf_file(gguf_path)

        # Initialize the model
        self._initialize_model(gguf_path)

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        return (
            self.model_def.details.gguf_file
            if hasattr(self.model_def.details, "gguf_file")
            and self.model_def.details.gguf_file
            else self.model_def.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate the GGUF file exists and has reasonable size."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf_path}"
            )

        self._logger.info(
            f"Using GGUF file: {gguf_path} (size: {file_size/1_000_000:.2f} MB)"
        )

    def _initialize_model(self, gguf_path: str) -> None:
        """Initialize the Llama model for embedding."""
        try:
            self.model = Llama(
                model_path=gguf_path,
                n_ctx=512,  # 512 is the maximum sequence length for this model
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=4,
                use_mlock=True,
                embedding=True,  # Enable embedding mode
                verbose=False,  # Reduce verbosity
            )
            self._logger.info(
                f"Nomic Embed Text model '{self.model_def.name}' loaded successfully."
            )
        except Exception as e:
            self._logger.error(
                f"Error initializing {self.__class__.__name__}: {str(e)}"
            )
            raise

    def _add_task_prefix(self, text: str, is_query: Optional[bool] = None) -> str:
        """Add appropriate task prefix to text based on content or explicit flag."""
        # Check if text already has a prefix
        if text.startswith(("search_document: ", "search_query: ")):
            return text

        # Determine prefix based on is_query flag or text length heuristic
        if is_query is True:
            prefix = "search_query: "
        elif is_query is False:
            prefix = "search_document: "
        else:
            # Auto-detect: shorter texts are likely queries, longer ones are documents
            prefix = "search_query: " if len(text) < 100 else "search_document: "

        return f"{prefix}{text}"

    def _extract_texts_from_messages(self, messages: List[Message]) -> List[str]:
        """Extract text content from messages."""
        texts = []
        for message in messages:
            if not message.content:
                continue
            for content in message.content:
                if (
                    hasattr(content, "type")
                    and hasattr(content, "text")
                    and content.text
                ):
                    texts.append(content.text)
        return texts

    def _generate_embeddings(
        self, texts: List[str], normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Llama's built-in embed method.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings

        Returns:
            List of embeddings, one per text
        """
        if not self.model:
            raise RuntimeError("Model not initialized")

        if not texts:
            return []

        try:
            # Use Llama's built-in embed method which handles batching, tokenization, and truncation
            embeddings = self.model.embed(
                input=texts,
                normalize=normalize,  # Let Llama handle normalization
                truncate=True,  # Automatically truncate to model's max context
            )

            # Convert the embeddings to the expected List[List[float]] format
            result = []

            # Handle tuple return format (embeddings, token_count)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Handle single embedding (List[float])
            if embeddings and not isinstance(embeddings[0], list):
                result = [embeddings]
            else:
                result = embeddings

            self._logger.debug(
                f"Generated {len(result)} embeddings using Llama's embed method"
            )
            return cast(List[List[float]], result)

        except Exception as e:
            self._logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """Process input messages to generate embeddings for text."""
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            # Extract texts from messages
            raw_texts = self._extract_texts_from_messages(req.messages)

            if not raw_texts:
                self._logger.warning("No text inputs found in messages")
                raw_texts = [""]  # Add empty input to avoid errors

            # Add task prefixes to texts
            processed_texts = [self._add_task_prefix(text) for text in raw_texts]

            self._logger.info(
                f"Processing {len(processed_texts)} text inputs for embedding"
            )

            # Generate embeddings using Llama's built-in method
            embeddings = self._generate_embeddings(processed_texts, normalize=True)

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            self._logger.debug(
                f"Generated {len(embeddings)} embeddings with {len(embeddings[0]) if embeddings else 0} dimensions each"
            )

            # Create response
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
                finish_reason="success",
                context=embeddings,  # Store embeddings in context
                total_duration=total_duration,
                load_duration=0.0,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=0,
                created_at=start_time,
                model=str(self.model_def.id),
            )
            yield response

        except Exception as e:
            self._logger.error(f"Error running Nomic Embed Text model: {str(e)}")
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
                context=[],  # Empty context on error
                total_duration=(
                    datetime.datetime.now(tz=datetime.timezone.utc) - start_time
                ).total_seconds()
                * 1000,
                load_duration=0.0,
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
        is_query: Optional[bool] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts using Llama's built-in embed method.
        Always returns List[List[float]] where each embedding is 768 dimensions (or matryoshka_dim).

        Args:
            texts: The text or list of texts to embed
            is_query: Whether the text is a query (True), document (False), or auto-detect (None)
            matryoshka_dim: Optional dimension for Matryoshka embedding truncation (256, 512, or 768)

        Returns:
            A list of embeddings for each input text
        """
        # Standardize input to list format
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        try:
            # Add appropriate prefixes to all texts
            processed_texts = [self._add_task_prefix(text, is_query) for text in texts]

            # Generate embeddings using Llama's built-in method
            embeddings = self._generate_embeddings(processed_texts, normalize=True)

            # Apply Matryoshka truncation if specified
            if matryoshka_dim and matryoshka_dim in [256, 512, 768]:
                self._logger.debug(
                    f"Truncating embeddings to {matryoshka_dim} dimensions"
                )
                embeddings = [emb[:matryoshka_dim] for emb in embeddings]
            elif matryoshka_dim and matryoshka_dim not in [256, 512, 768]:
                self._logger.warning(
                    f"Invalid matryoshka_dim: {matryoshka_dim}. Using full 768 dimensions."
                )

            # Ensure all embeddings have consistent dimensions
            expected_dim = matryoshka_dim if matryoshka_dim in [256, 512, 768] else 768
            normalized_embeddings = []

            for embedding in embeddings:
                if len(embedding) > expected_dim:
                    embedding = embedding[:expected_dim]
                elif len(embedding) < expected_dim:
                    embedding.extend([0.0] * (expected_dim - len(embedding)))
                normalized_embeddings.append(embedding)

            self._logger.debug(
                f"Generated {len(normalized_embeddings)} embeddings from {len(texts)} input texts"
            )
            return normalized_embeddings

        except Exception as e:
            self._logger.error(f"Error in emb method: {str(e)}")
            # Return empty embeddings as fallback
            dim = matryoshka_dim if matryoshka_dim in [256, 512, 768] else 768
            return [[0.0] * dim for _ in texts]

    def _extract_embedding_from_response(
        self, responses
    ) -> Optional[List[List[float]]]:
        """
        Extract embeddings from model responses.
        Always returns List[List[float]] format.

        Args:
            responses: List of responses from the model

        Returns:
            List of embedding vectors or None if not found
        """
        # Extract embeddings from the context field of ChatResponse
        for response in responses:
            if hasattr(response, "context") and response.context:
                # Ensure we always return List[List[float]]
                if isinstance(response.context, list):
                    # Check if it's already List[List[float]]
                    if response.context and isinstance(response.context[0], list):
                        return response.context
                    # Convert List[float] to List[List[float]]
                    elif response.context and isinstance(
                        response.context[0], (int, float)
                    ):
                        return [response.context]
                return response.context

        return None
        """Clean up resources used by the NomicEmbedTextPipe."""
        try:
            if hasattr(self, "_logger"):
                self._logger.info(f"NomicEmbedTextPipe cleanup initiated")

            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error cleaning up NomicEmbedTextPipe resources: {str(e)}")
