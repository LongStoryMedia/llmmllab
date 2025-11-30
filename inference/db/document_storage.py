"""Document storage service for database operations."""

from typing import List, Optional
import asyncpg

from models.document import Document


class DocumentStorage:
    """Storage service for document operations."""

    def __init__(self, pool: asyncpg.Pool, get_query: callable):
        """Initialize with database connection pool and query getter."""
        self.pool = pool
        self.get_query = get_query

    async def store_document(
        self,
        conversation_id: int,
        user_id: str,
        filename: str,
        content_type: str,
        file_size: int,
        content: str,
        text_content: Optional[str] = None,
    ) -> Document:
        """Store a new document and return the created object."""
        query = self.get_query("document", "store_document")

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                query,
                conversation_id,
                user_id,
                filename,
                content_type,
                file_size,
                content,
                text_content,
            )

            return Document(
                id=result["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                filename=filename,
                content_type=content_type,
                file_size=file_size,
                content=content,
                text_content=text_content,
                created_at=result["created_at"],
                updated_at=result["created_at"],  # Same as created_at initially
            )

    async def get_document(self, document_id: int) -> Optional[Document]:
        """Get a document by ID."""
        query = self.get_query("document", "get_document")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, document_id)

            if not row:
                return None

            return Document(
                id=row["id"],
                conversation_id=row["conversation_id"],
                user_id=row["user_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                file_size=row["file_size"],
                content=row["content"],
                text_content=row["text_content"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def get_documents_for_conversation(
        self, conversation_id: int
    ) -> List[Document]:
        """Get all documents for a conversation."""
        query = self.get_query("document", "get_documents_by_conversation")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, conversation_id)

            return [
                Document(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    user_id=row["user_id"],
                    filename=row["filename"],
                    content_type=row["content_type"],
                    file_size=row["file_size"],
                    content=row["content"],
                    text_content=row["text_content"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]
