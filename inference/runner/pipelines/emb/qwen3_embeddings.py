"""
Embeddings wrapper for Qwen3EmbeddingPipe to implement LangChain Embeddings interface.
"""

import logging
from typing import List

from langchain_core.embeddings import Embeddings

from models import Model, ModelProfile, Message
from runner.pipelines.emb.qwen3emb import Qwen3EmbeddingPipe


class Qwen3Embeddings(Embeddings):
    """
    LangChain Embeddings implementation wrapping Qwen3EmbeddingPipe.

    This wrapper provides the standard LangChain Embeddings interface
    while using the existing Qwen3EmbeddingPipe implementation.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Qwen3 embeddings wrapper."""
        self.pipeline = Qwen3EmbeddingPipe(model, profile)
        self._logger = logging.getLogger(self.__class__.__name__)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        import asyncio

        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        import asyncio

        # For single text, return the first (and only) embedding
        embeddings = asyncio.run(self.aembed_documents([text]))
        return embeddings[0] if embeddings else [0.0] * 1024

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous embed search docs."""
        # Convert texts to Messages for the pipeline
        messages = []
        from models import Message, MessageRole, MessageContent, MessageContentType

        for text in texts:
            message = Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=text)],
            )
            messages.append(message)

        # Use the pipeline's process_messages method
        return await self.pipeline.process_messages(messages)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous embed query text."""
        embeddings = await self.aembed_documents([text])
        return embeddings[0] if embeddings else [0.0] * 1024

    def embed_messages(self, messages: List[Message]) -> List[List[float]]:
        """Embed messages using the pipeline interface."""
        import asyncio

        return asyncio.run(self.aembed_messages(messages))

    async def aembed_messages(self, messages: List[Message]) -> List[List[float]]:
        """Asynchronously embed messages using the pipeline interface."""
        return await self.pipeline.process_messages(messages)
