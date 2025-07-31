"""
Embedding service implementation for the gRPC server.
"""

from inference.server.config import logger
from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys
import logging
import numpy as np
from typing import Dict, List

# Fix imports using absolute paths
try:
    from inference.protos import embedding_req_pb2
    from inference.protos import embedding_response_pb2
except ImportError:
    # Alternative import path
    from protos import embedding_req_pb2
    from protos import embedding_response_pb2


class EmbeddingService:
    """
    Service for handling embedding generation requests.
    """

    def __init__(self):
        self.logger = logger
        self.models = {}
        self.tokenizers = {}
        self.default_embedding_model = "intfloat/multilingual-e5-large"

    def _load_model_if_needed(self, model_name: str):
        """
        Load the embedding model if it's not already loaded.
        """
        if model_name in self.models and model_name in self.tokenizers:
            return self.models[model_name], self.tokenizers[model_name]

        self.logger.info(f"Loading embedding model: {model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")

        # Save model and tokenizer
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer

        return model, tokenizer

    def GetEmbedding(self, request: embedding_req_pb2.EmbeddingReq, context):
        """
        Generate embeddings for the provided text.
        """
        # Get the model name, using default if not provided
        model_name = request.model if request.model else self.default_embedding_model
        try:
            # Load the model if needed
            model, tokenizer = self._load_model_if_needed(model_name)

            # Process the text and generate embedding
            if model_name.startswith("intfloat/e5") or model_name.startswith(
                "intfloat/multilingual-e5"
            ):
                # Special handling for E5 models
                text = f"query: {request.input}"
            else:
                text = request.input

            # Tokenize the input
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            # For most models, CLS token embedding is used as sentence embedding
            if model_name.startswith("intfloat/e5") or model_name.startswith(
                "intfloat/multilingual-e5"
            ):
                # E5 models: mean pooling of last hidden state
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            else:
                # Default: use CLS token
                embeddings = outputs.last_hidden_state[:, 0]

            # Convert to float32 and flatten
            embedding_np = embeddings.cpu().numpy().flatten().astype(np.float32)

            # Create and return response
            return embedding_response_pb2.EmbeddingResponse(
                embeddings=embedding_np,
                model=model_name,
                total_duration=0,  # Placeholder for duration
                load_duration=0,  # Placeholder for load duration
                prompt_eval_count=1,  # Placeholder for prompt eval count
            )

        except Exception as e:
            self.logger.error(f"Error in get_embedding: {e}")
            return embedding_response_pb2.EmbeddingResponse(
                embeddings=[],
                model=model_name,
                total_duration=0,
                load_duration=0,
                prompt_eval_count=1,
            )
        finally:
            # Clean up resources if needed
            torch.cuda.empty_cache()

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling for E5 models.
        """
        # Expand attention mask to same dimensions as token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Apply mask and calculate mean
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
