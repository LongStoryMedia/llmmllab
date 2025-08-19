"""
Embedding model pipeline for Nomic Embed Text v2 model.

This pipeline implemen                # Log model info for debugging
        self._logger.info(f"Model ID: {self.model_def.id}")

        # Get the GGUF file path
        gguf = (e Nomic Embed Text v2 Mixture of Experts (MoE) model for generating
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
    model: Any = None
    tokenizer: Any = None
    encoder: Any = None

    def __init__(self, model_definition: Model):
        """Initialize the Nomic Embed Text pipeline."""
        # Call base class initialization first
        super().__init__(model_definition)

        # Set up logger
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing NomicEmbedTextPipe")
        self._logger.info(f"Model definition: {self.model_def.json()}")

        # Ensure model details for GGUF are provided
        if not (self.model_def.details and self.model_def.model):
            raise ValueError(
                "Model definition for NomicEmbedTextPipe must include model path details."
            )

        # Ensure model details for GGUF are provided
        if not (model_definition.details and model_definition.model):
            raise ValueError(
                "Model definition for NomicEmbedTextPipe must include model path details."
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
            # Using n_ctx=512 as per the model's specifications in the documentation
            # https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
            self.model = Llama(
                model_path=gguf,
                n_ctx=512,  # 512 is the maximum sequence length for this model
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=4,
                use_mlock=True,
                embedding=True,  # Enable embedding mode
            )

            self._logger.info(
                f"Nomic Embed Text model '{self.model_def.name}' loaded successfully."
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
            Generator[ChatResponse, Any, None]: Yields ChatResponse objects.
        """
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        load_time = 0.0  # No loading time measurement in this case

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
                        # Add required task instruction prefix according to Nomic Embed v2 requirements
                        # Using "search_document: " prefix as we're embedding content
                        # See: https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
                        if not content.text.startswith(
                            "search_document: "
                        ) and not content.text.startswith("search_query: "):
                            # Determine if this is more likely a query or a document
                            # For simplicity, treating shorter texts (< 100 chars) as queries, longer as documents
                            if len(content.text) < 100:
                                prefix = "search_query: "
                            else:
                                prefix = "search_document: "
                            processed_text = f"{prefix}{content.text}"
                        else:
                            # Already has a prefix
                            processed_text = content.text

                        inputs.append(processed_text)

            if not inputs:
                self._logger.warning("No text inputs found in messages")
                inputs = ["search_document: "]  # Add empty input to avoid errors

            self._logger.info(f"Running embedding model with {len(inputs)} inputs")

            embeddings = []
            # Check if matryoshka dimension is specified in parameters
            matryoshka_dim = None
            # if hasattr(req, "parameters") and req.parameters:
            #     if (
            #         hasattr(req.parameters, "matryoshka_dim")
            #         and req.parameters.matryoshka_dim
            #     ):
            #         try:
            #             matryoshka_dim = int(req.parameters.matryoshka_dim)
            #             # Valid Matryoshka dimensions are 256, 512, and 768 (full size)
            #             if matryoshka_dim not in [256, 512, 768]:
            #                 self.logger.warning(
            #                     f"Invalid matryoshka_dim: {matryoshka_dim}. Using full 768 dimensions."
            #                 )
            #                 matryoshka_dim = None
            #         except (ValueError, TypeError):
            #             self.logger.warning(
            #                 f"Invalid matryoshka_dim value: {req.parameters.matryoshka_dim}. Using full 768 dimensions."
            #             )

            for text_input in inputs:
                if self.model and hasattr(self.model, "embed"):
                    # Generate embedding for each input text
                    embedding_result = self.model.embed(text_input)
                    self._logger.info(f"Raw embedding type: {type(embedding_result)}")

                    # Extract embeddings array from model output
                    self._logger.debug(
                        f"Processing embedding result type={type(embedding_result)}"
                    )
                    # The model returns a direct list of numbers for embeddings
                    embedding = embedding_result
                    self._logger.debug(f"Using raw embedding list: {type(embedding)}")

                    # Convert to numpy array for processing, handle both list and numpy array inputs
                    embedding_array = np.asarray(embedding)

                    # Apply Matryoshka truncation if specified
                    if matryoshka_dim and matryoshka_dim < len(embedding_array):
                        self._logger.info(
                            f"Truncating embedding to {matryoshka_dim} dimensions"
                        )
                        embedding_array = embedding_array[:matryoshka_dim]

                    # Normalize embedding (required by the model)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm

                    embeddings.append(embedding_array.tolist())
                else:
                    self._logger.error(
                        "Model not properly initialized or doesn't support embedding"
                    )
                    embedding_dim = 768 if not matryoshka_dim else matryoshka_dim
                    embeddings.append(
                        [0.0] * embedding_dim
                    )  # Return empty embedding as fallback

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Log embedding information before creating response
            self._logger.debug(
                f"Final embeddings type: {type(embeddings)}, length: {len(embeddings) if embeddings else 0}"
            )
            if embeddings and len(embeddings) > 0:
                self._logger.debug(
                    f"First embedding type: {type(embeddings[0])}, length: {len(embeddings[0]) if isinstance(embeddings[0], (list, np.ndarray)) else 'N/A'}"
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
                finish_reason="success",
                context=embeddings,
                total_duration=total_duration,
                load_duration=load_time,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=0,
                created_at=start_time,
                model=str(self.model_def.id),
            )
            yield response

        except (RuntimeError, ValueError) as e:
            self._logger.error(f"Error running Nomic Embed Text model: {str(e)}")
            # Create error response
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
                # context argument removed
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
                    # Generate embeddings using the pipeline
                    responses = list(self.run(req))

                    # Extract embedding from response
                    embedding = self._extract_embedding_from_response(responses)
                    if embedding:
                        all_embeddings.append(embedding)
                    else:
                        # Return empty embedding as fallback
                        dim = matryoshka_dim or 768
                        all_embeddings.append([0.0] * dim)

                except Exception:
                    # Return empty embedding as fallback
                    dim = matryoshka_dim or 768
                    all_embeddings.append([0.0] * dim)

            return all_embeddings

        except Exception as e:
            # Return empty embeddings as fallback
            dim = matryoshka_dim or 768
            return [[0.0] * dim] * len(texts)

    def __del__(self) -> None:
        """
        Clean up resources used by the NomicEmbedTextPipe.
        """
        try:
            self._logger.info(
                f"NomicEmbedTextPipe for {self.model_def.name if hasattr(self, 'model_def') else 'unknown'}: Cleanup initiated"
            )
            if hasattr(self, "model") and self.model is not None:
                # llama-cpp-python models should have their resources cleaned up
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AttributeError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error cleaning up NomicEmbedTextPipe resources: {str(e)}")
