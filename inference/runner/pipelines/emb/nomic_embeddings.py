"""
Embeddings wrapper for NomicEmbedTextPipe to implement LangChain Embeddings interface.
"""

import logging
from typing import List

from langchain_core.embeddings import Embeddings

from models import Model, ModelProfile, Message, MessageRole
from runner.pipelines.emb.nom2 import NomicEmbedTextPipe
from utils.message import extract_message_text


class NomicEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation wrapping NomicEmbedTextPipe.

    This wrapper provides the standard LangChain Embeddings interface
    while using the existing NomicEmbedTextPipe implementation.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Nomic embeddings wrapper."""
        self.pipeline = NomicEmbedTextPipe(model, profile)
        self._logger = logging.getLogger(self.__class__.__name__)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        import asyncio

        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        import asyncio

        return asyncio.run(self.aembed_query(text))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous embed search docs."""
        return await self.pipeline.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous embed query text."""
        return await self.pipeline.embed_query(text)

    def embed_messages(self, messages: List[Message]) -> List[List[float]]:
        """Embed messages using the pipeline interface."""
        import asyncio

        return asyncio.run(self.aembed_messages(messages))

    async def aembed_messages(self, messages: List[Message]) -> List[List[float]]:
        """Asynchronously embed messages using the pipeline interface."""
        return await self.pipeline.invoke(messages)
