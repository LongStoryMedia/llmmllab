"""
LlamaCppServerEmbeddings - Direct llama.cpp server embedding integration.

This replaces llama-cpp-python for embeddings using the llama.cpp server's
native /v1/embeddings endpoint with the new server manager architecture.
"""

from typing import List, Optional
import asyncio

import requests
from langchain_core.embeddings import Embeddings

from runner.models import Model, ModelProfile, UserConfig
from runner.utils.logging import llmmllogger
from runner.server_manager import LlamaCppServerManager


logger = llmmllogger.bind(component="LlamaCppServerEmbeddings")


class EmbedLlamaCppPipeline(Embeddings):
    """
    LangChain embeddings implementation using llama.cpp server.

    Uses the new LlamaCppServerManager for consistent server management.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        user_config: Optional[UserConfig] = None,
        metadata: Optional[dict] = {},
    ):
        """Initialize embeddings with persistent server."""
        self.model = model
        self.profile = profile
        self.user_config = user_config
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )

        # Use the new server manager architecture
        self.server_manager = LlamaCppServerManager(
            model=model,
            profile=profile,
            user_config=user_config,
            is_embedding=True,  # Enable embedding mode
        )

        # Start persistent server for embeddings
        self.started = self.server_manager.start()
        if not self.started:
            raise RuntimeError(
                f"Failed to start embedding server for model {model.name}"
            )

        self._logger.info(f"Embedding server ready for model {model.name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts."""
        if not texts:
            return []

        try:
            # Use the flexible endpoint system
            embeddings_url = self.server_manager.get_api_endpoint("/embeddings")

            response = requests.post(
                embeddings_url,
                json={
                    "input": texts,
                    "model": "local-model",
                    "encoding_format": "float",
                },
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]

            self._logger.debug(f"Generated embeddings for {len(texts)} documents")
            return embeddings

        except Exception as e:
            self._logger.error(f"Failed to generate document embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if not text:
            return []

        try:
            # Use the flexible endpoint system
            embeddings_url = self.server_manager.get_api_endpoint("/embeddings")

            response = requests.post(
                embeddings_url,
                json={
                    "input": [text],  # Wrap single text in list
                    "model": "local-model",
                    "encoding_format": "float",
                },
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            self._logger.debug(f"Generated embedding for query: {text[:50]}...")
            return embedding

        except Exception as e:
            self._logger.error(f"Failed to generate query embedding: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of document texts."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.embed_documents, texts
        )

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query text."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.embed_query, text
        )

    def close(self):
        """Clean up server resources."""
        try:
            if self.server_manager:
                self.server_manager.stop()
                self._logger.info("Embedding server stopped successfully")
        except Exception as e:
            self._logger.error(f"Error stopping embedding server: {e}")

    def shutdown(self):
        """Alias for close to match server manager interface."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


__all__ = ["EmbedLlamaCppPipeline"]
