"""
Document storage module for RAG pipelines.
Stores retrieved documents separately from conversation memories.
"""

import logging
from typing import List, Optional

import asyncpg
from langchain_core.documents import Document

from server.db.db_utils import typed_pool
from server.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        """Initialize DocumentStorage with database pool and query function."""
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def init_document_schema(self):
        """Initialize document storage schema if it doesn't exist."""
        logger.info("Initializing document storage schema...")
        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("document.init_document_schema"))
            logger.info("Created documents table")
            await conn.execute(self.get_query("document.create_document_indexes"))
            logger.info("Created document indexes")
            try:
                await conn.execute(
                    self.get_query("document.enable_documents_compression")
                )
                logger.info("Enabled documents compression")
            except asyncpg.PostgresError as e:
                logger.warning(
                    f"Failed to add documents compression due to database error: {e}"
                )
            except ValueError as e:
                logger.warning(
                    f"Failed to add documents compression due to value error: {e}"
                )
            except (TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Failed to add documents compression due to error: {e}")
            except Exception as e:
                # FIXME: This is a general catch-all exception handler that should be refined
                # in the future based on observed error patterns in production.
                logger.critical(
                    f"Critical error adding documents compression: {e}", exc_info=True
                )

            try:
                await conn.execute(
                    self.get_query("document.documents_retention_policy")
                )
                logger.info("Added documents retention policy")
            except asyncpg.PostgresError as e:
                logger.warning(
                    f"Failed to add documents retention policy due to database error: {e}"
                )
            except ValueError as e:
                logger.warning(
                    f"Failed to add documents retention policy due to value error: {e}"
                )
            except (TypeError, AttributeError, KeyError) as e:
                logger.warning(
                    f"Failed to add documents retention policy due to error: {e}"
                )
            except Exception as e:
                # FIXME: This is a general catch-all exception handler that should be refined
                # in the future based on observed error patterns in production.
                logger.critical(
                    f"Critical error adding documents retention policy: {e}",
                    exc_info=True,
                )
        logger.info("Document schema initialized successfully")

    async def store_document(
        self,
        user_id: str,
        content: str,
        source: str,
        conversation_id: int,
        embedding: List[float],
        metadata: Optional[dict] = None,
    ):
        """
        Store a document with its embedding.

        Args:
            user_id: User ID the document is associated with
            content: Document content
            source: Source of the document (web, arxiv, etc.)
            conversation_id: ID of the conversation that retrieved this document
            embedding: Vector embedding of the document content
            metadata: Additional metadata about the document
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                embedding_str = self.format_embedding_for_pgvector(embedding)
                metadata_str = str(metadata) if metadata else "{}"

                await conn.execute(
                    self.get_query("document.store_document"),
                    user_id,
                    conversation_id,
                    content,
                    source,
                    embedding_str,
                    metadata_str,
                )

    async def search_documents(
        self,
        embedding: List[float],
        min_similarity: float = 0.7,
        limit: int = 10,
        user_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Search for semantically similar documents.

        Args:
            embedding: The query embedding vector
            min_similarity: Minimum cosine similarity threshold
            limit: Maximum number of results to return
            user_id: Filter by user ID
            source_types: Filter by document source types

        Returns:
            List of Document objects with metadata
        """
        docs = []
        embedding_str = self.format_embedding_for_pgvector(embedding)

        params = [embedding_str, min_similarity, limit, user_id]
        source_types_param = source_types if source_types else None
        params.append(source_types_param)

        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(self.get_query("document.search_similar"), *params)

            for row in rows:
                metadata = {
                    "id": row["id"],
                    "source": row["source"],
                    "conversation_id": row["conversation_id"],
                    "created_at": row["created_at"].isoformat(),
                    "similarity": float(row["similarity"]),
                }

                # Add any additional metadata if available
                try:
                    if row["metadata"] and isinstance(row["metadata"], dict):
                        metadata.update(row["metadata"])
                except (ValueError, TypeError):
                    pass

                docs.append(Document(page_content=row["content"], metadata=metadata))

        return docs

    @staticmethod
    def format_embedding_for_pgvector(embedding: List[float]) -> str:
        """Format embedding for pgvector storage."""
        return "[" + ",".join(f"{val:f}" for val in embedding) + "]"


async def store_documents(
    documents: List[Document],
    user_id: str,
    conversation_id: int,
    embedding_service: EmbeddingService,
    document_storage: DocumentStorage,
) -> None:
    """
    Store retrieved documents for future retrieval.

    Args:
        documents: The documents to store
        user_id: The user ID
        conversation_id: The conversation ID
        embedding_service: The embedding service to use for embedding generation
        document_storage: The document storage instance
    """
    try:
        for doc in documents:
            # Get content from document
            content = doc.page_content
            if not content:
                continue

            # Generate embedding
            embedding = await embedding_service.create_embeddings(content)
            if not embedding or len(embedding) == 0:
                continue

            # Determine source
            source = "external"
            if "source_type" in doc.metadata:
                source = doc.metadata["source_type"]
            elif "source" in doc.metadata:
                source = doc.metadata["source"]

            # Store as document with metadata
            await document_storage.store_document(
                user_id=user_id,
                content=content,
                source=source,
                conversation_id=conversation_id,
                embedding=embedding[0],
                metadata=doc.metadata,
            )

        logger.info(f"Stored {len(documents)} documents for user {user_id}")
    except ValueError as e:
        logger.error(f"Value error storing documents: {e}")
    except TypeError as e:
        logger.error(f"Type error storing documents: {e}")
    except asyncpg.PostgresError as e:
        logger.error(f"Database error storing documents: {e}")
    except (AttributeError, KeyError, IndexError) as e:
        logger.error(f"Error accessing document attributes: {e}", exc_info=True)
    except Exception as e:
        # FIXME: This is a general catch-all exception handler that should be refined
        # in the future based on observed error patterns in production.
        logger.critical(f"Critical error storing documents: {e}", exc_info=True)
