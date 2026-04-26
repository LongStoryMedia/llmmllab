"""Document storage service for database operations."""

from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from models.document import Document


class DocumentStorage:
    """Storage service for document operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Initialize with session factory."""
        self.session_factory = session_factory

    async def store_document(
        self,
        message_id: int,
        user_id: str,
        filename: str,
        content_type: str,
        file_size: int,
        content: str,
        text_content: Optional[str] = None,
    ) -> Document:
        """Store a new document and return the created object."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO documents(message_id, user_id, filename, content_type, file_size, content, text_content)
                    VALUES (:message_id, :user_id, :filename, :content_type, :file_size, :content, :text_content)
                    RETURNING id, created_at
                """),
                {
                    "message_id": message_id,
                    "user_id": user_id,
                    "filename": filename,
                    "content_type": content_type,
                    "file_size": file_size,
                    "content": content,
                    "text_content": text_content,
                },
            )
            await session.commit()

            row = result.mappings().first()
            return Document(
                id=row["id"],
                message_id=message_id,
                user_id=user_id,
                filename=filename,
                content_type=content_type,
                file_size=file_size,
                content=content,
                text_content=text_content,
                created_at=row["created_at"],
                updated_at=row["created_at"],  # Same as created_at initially
            )

    async def get_document(self, document_id: int) -> Optional[Document]:
        """Get a document by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        id,
                        conversation_id,
                        user_id,
                        filename,
                        content_type,
                        file_size,
                        content,
                        text_content,
                        created_at,
                        updated_at
                    FROM documents
                    WHERE id = :document_id
                """),
                {"document_id": document_id},
            )
            row = result.mappings().first()

            if not row:
                return None

            return Document(
                id=row["id"],
                message_id=row["message_id"],
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
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        d.id,
                        d.message_id,
                        d.user_id,
                        d.filename,
                        d.content_type,
                        d.file_size,
                        d.content,
                        d.text_content,
                        d.created_at,
                        d.updated_at
                    FROM
                        documents d
                        INNER JOIN messages m ON d.message_id = m.id
                    WHERE
                        m.conversation_id = :conversation_id
                    ORDER BY
                        d.created_at ASC
                """),
                {"conversation_id": conversation_id},
            )

            return [
                Document(
                    id=row["id"],
                    message_id=row["message_id"],
                    user_id=row["user_id"],
                    filename=row["filename"],
                    content_type=row["content_type"],
                    file_size=row["file_size"],
                    content=row["content"],
                    text_content=row["text_content"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in result.mappings()
            ]

    async def get_documents_for_message(self, message_id: int) -> List[Document]:
        """Get all documents for a specific message."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        id,
                        message_id,
                        user_id,
                        filename,
                        content_type,
                        file_size,
                        content,
                        text_content,
                        created_at,
                        updated_at
                    FROM documents
                    WHERE message_id = :message_id
                    ORDER BY created_at ASC
                """),
                {"message_id": message_id},
            )

            return [
                Document(
                    id=row["id"],
                    message_id=row["message_id"],
                    user_id=row["user_id"],
                    filename=row["filename"],
                    content_type=row["content_type"],
                    file_size=row["file_size"],
                    content=row["content"],
                    text_content=row["text_content"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in result.mappings()
            ]