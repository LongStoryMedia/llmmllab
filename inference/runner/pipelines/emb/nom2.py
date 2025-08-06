"""
Embedding model pipeline for Nomic Embed Text v2 model.
"""

import datetime
import logging
import os
import numpy as np
import torch
from typing import Any, Dict, List, Generator, Optional
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
    """

    def __init__(self, model_definition: Model):
        """Initialize the Nomic Embed Text pipeline."""
        super().__init__()
        self.model_def = model_definition
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.encoder = None

        # Ensure model details for GGUF are provided
        if not (model_definition.details and model_definition.model):
            raise ValueError(
                "Model definition for NomicEmbedTextPipe must include model path details."
            )

        # Log model info for debugging
        self.logger.info(f"Model ID: {self.model_def.id}")

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
        self.logger.info(
            f"Using GGUF file path: {gguf} (size: {file_size/1_000_000:.2f} MB)"
        )

        # Load the GGUF model using llama-cpp-python for embedding
        try:
            self.model = Llama(
                model_path=gguf,
                n_ctx=512,  # Smaller context for embeddings
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=4,
                use_mlock=True,
                embedding=True,  # Enable embedding mode
            )

            self.logger.info(
                f"Nomic Embed Text model '{self.model_def.name}' loaded successfully."
            )
        except Exception as e:
            self.logger.error(f"Error initializing {self.__class__.__name__}: {str(e)}")
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
                        inputs.append(content.text)

            if not inputs:
                self.logger.warning("No text inputs found in messages")
                inputs = [""]  # Add empty input to avoid errors

            self.logger.info(f"Running embedding model with {len(inputs)} inputs")

            embeddings = []
            for text_input in inputs:
                if self.model and hasattr(self.model, "embed"):
                    # Generate embedding for each input text
                    embedding = self.model.embed(text_input)
                    embedding_array = np.array(embedding)
                    # Normalize if requested (default to True)
                    normalize = True
                    # Remove use of params.normalize (not in ModelParameters)
                    if normalize:
                        norm = np.linalg.norm(embedding_array)
                        if norm > 0:
                            embedding_array = embedding_array / norm
                    embeddings.append(embedding_array.tolist())
                else:
                    self.logger.error(
                        "Model not properly initialized or doesn't support embedding"
                    )
                    embeddings.append([0.0] * 256)  # Return empty embedding as fallback

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Create a ChatResponse object to return
            response = ChatResponse(
                message=Message(
                    id=None,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Generated {len(embeddings)} embeddings",
                            url=None,
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    created_at=start_time,
                    conversation_id=0,
                ),
                done=True,
                finish_reason="success",
                # context argument removed
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
            self.logger.error(f"Error running Nomic Embed Text model: {str(e)}")
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
                    conversation_id=0,
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

    def __del__(self) -> None:
        """
        Clean up resources used by the NomicEmbedTextPipe.
        """
        try:
            self.logger.info(
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
