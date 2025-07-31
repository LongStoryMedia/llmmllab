"""
Embedding model pipeline for Nomic Embed Text v2 model.
"""

import datetime
import logging
import os
import numpy as np
import torch
from typing import Any, Dict, List, Generator

from models import Model
from ..base_pipeline import BasePipeline


class NomicEmbedTextPipe(BasePipeline):
    """
    Pipeline for running text embeddings with Nomic Embed Text v2 model.

    This pipeline supports the nomic-ai/nomic-embed-text-v2-moe model in GGUF format.
    """

    def __init__(self, model_definition: Model):
        """Initialize the Nomic Embed Text pipeline."""
        self.model_def = model_definition
        self.logger = logging.getLogger(__name__)

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
            if model_definition.details.gguf_file
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

        try:
            # Import here to avoid issues
            from llama_cpp import Llama

            # Load the GGUF model using llama-cpp-python for embedding
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

    def __del__(self) -> None:
        """
        Clean up resources used by the NomicEmbedTextPipe.
        """
        try:
            self.logger.info(
                f"NomicEmbedTextPipe for {self.model_def.name}: Cleanup initiated"
            )
            if hasattr(self, "model"):
                # llama-cpp-python models should have their resources cleaned up
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(
                f"Error cleaning up NomicEmbedTextPipe resources: {str(e)}"
            )

    def run(self, req, load_time: float) -> Generator[Dict, Any, None]:
        """
        Run the Nomic Embed Text model to generate embeddings for text.
        """
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            # For embeddings, we assume req has 'inputs' which is a list of strings
            inputs = req.inputs if hasattr(req, "inputs") else [req.input]
            self.logger.info(f"Running embedding model with {len(inputs)} inputs")

            embeddings = []
            for text_input in inputs:
                # Generate embedding for each input text
                embedding = self.model.embed(text_input)

                # Convert to numpy array
                embedding_array = np.array(embedding)

                # Normalize if requested (default to True)
                normalize = True
                if (
                    hasattr(req, "options")
                    and hasattr(req.options, "normalize")
                    and req.options.normalize is not None
                ):
                    normalize = req.options.normalize

                if normalize:
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm

                embeddings.append(embedding_array.tolist())

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Create the response as a dict
            response = {
                "embeddings": embeddings,
                "model": self.model_def.id,
                "usage": {
                    "prompt_tokens": sum(
                        len(text.split()) for text in inputs
                    ),  # Rough estimate
                    "total_tokens": sum(len(text.split()) for text in inputs),
                },
                "created_at": end_time,
                "total_duration": total_duration,
                "load_duration": load_time,
            }

            yield response

        except Exception as e:
            self.logger.error(f"Error running {self.__class__.__name__}: {str(e)}")
            raise

    def __del__(self) -> None:
        """
        Clean up resources used by the NomicEmbedTextPipe.
        """
        try:
            self.logger.info(
                f"NomicEmbedTextPipe for {self.model_def.name}: Cleanup initiated"
            )
            if hasattr(self, "model"):
                # llama-cpp-python models should have their resources cleaned up
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(
                f"Error cleaning up NomicEmbedTextPipe resources: {str(e)}"
            )

    def run(self, req, load_time: float) -> Generator[Any, Any, None]:
        """
        Run the Nomic Embed Text model to generate embeddings for text.
        """
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            self.logger.info(f"Running embedding model with {len(req.inputs)} inputs")

            embeddings = []
            for text_input in req.inputs:
                # Generate embedding for each input text
                embedding = self.model.embed(text_input)

                # Convert to numpy array
                embedding_array = np.array(embedding)

                # Normalize if requested (default to True)
                if (
                    req.options
                    and req.options.normalize is not None
                    and req.options.normalize
                ):
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm

                embeddings.append(embedding_array.tolist())

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Create the response - using a dict instead of EmbeddingResponse to avoid import issues
            response = {
                "embeddings": embeddings,
                "model": self.model_def.id,
                "usage": {
                    "prompt_tokens": sum(
                        len(text.split()) for text in req.inputs
                    ),  # Rough estimate
                    "total_tokens": sum(len(text.split()) for text in req.inputs),
                },
                "created_at": end_time,
                "total_duration": total_duration,
                "load_duration": load_time,
            }

            yield response

        except Exception as e:
            self.logger.error(f"Error running Nomic Embed Text model: {str(e)}")
            raise

        try:
            import ctranslate2

            self.logger.info("Loading GGUF model with ctranslate2")

            # Verify the model file exists
            gguf_path = model.details.get("gguf_file")
            if not gguf_path or not os.path.exists(gguf_path):
                raise FileNotFoundError(f"GGUF file not found at: {gguf_path}")

            # Load the model
            self.encoder = ctranslate2.Encoder(
                gguf_path, device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Load tokenizer - using SentencePiece for Nomic embeddings
            from sentencepiece import SentencePieceProcessor

            tokenizer_path = os.path.join(os.path.dirname(gguf_path), "tokenizer.model")
            if os.path.exists(tokenizer_path):
                self.tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
            else:
                # Fallback to using transformers tokenizer
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    "nomic-ai/nomic-embed-text-v2-moe"
                )

            self.logger.info(f"Successfully initialized {self.__class__.__name__}")

        except Exception as e:
            self.logger.error(f"Error initializing {self.__class__.__name__}: {str(e)}")
            raise

    def encode_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts (List[str]): List of text strings to encode.
            **kwargs: Additional parameters for encoding.

        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_size).
        """
        batch_size = kwargs.get("batch_size", 8)
        normalize = kwargs.get("normalize", True)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            if hasattr(self.tokenizer, "encode_as_ids"):
                # Using SentencePiece
                tokens = [self.tokenizer.encode_as_ids(text) for text in batch_texts]
            else:
                # Using transformers tokenizer
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                tokens = encoded["input_ids"].tolist()

            # Get embeddings
            embeddings = self.encoder.forward_batch(tokens)
            embeddings = np.array(
                [emb[0] for emb in embeddings]
            )  # Use the [CLS] embedding or equivalent

            if normalize:
                # L2 normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the embedding pipeline on the provided input.

        Args:
            inputs (Dict[str, Any]): Dictionary containing:
                - 'text' or 'texts': String or list of strings to embed
                - Optional parameters for embedding

        Returns:
            Dict[str, Any]: Dictionary containing the embeddings under 'embeddings' key
        """
        try:
            # Handle both single text and list of texts
            if "text" in inputs:
                texts = [inputs["text"]]
            elif "texts" in inputs:
                texts = inputs["texts"]
            else:
                raise ValueError("Input must contain either 'text' or 'texts' key")

            # Extract additional parameters
            kwargs = {k: v for k, v in inputs.items() if k not in ["text", "texts"]}

            # Get embeddings
            embeddings = self.encode_texts(texts, **kwargs)

            # Return result
            single_input = "text" in inputs
            return {
                "embeddings": embeddings[0] if single_input else embeddings,
                "embedding_dim": embeddings.shape[-1],
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            return {"error": str(e), "success": False}

    def cleanup(self):
        """
        Clean up resources used by the pipeline.
        """
        try:
            # Clean up memory
            self.encoder = None
            self.tokenizer = None

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"{self.__class__.__name__} resources cleaned up")

        except Exception as e:
            self.logger.warning(
                f"Error during {self.__class__.__name__} cleanup: {str(e)}"
            )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the embedding pipeline with the given inputs.
        Implementation of the abstract method from BasePipeline.

        Args:
            inputs (Dict[str, Any]): The input data for the pipeline

        Returns:
            Dict[str, Any]: The output from the pipeline
        """
        return self(inputs)

    def __del__(self):
        """
        Clean up resources when the pipeline is deleted.
        Implementation of the abstract method from BasePipeline.
        """
        try:
            self.cleanup()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Error during {self.__class__.__name__} cleanup in __del__: {str(e)}"
            )
