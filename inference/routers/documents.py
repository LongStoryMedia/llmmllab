"""Document API endpoints."""

import base64
from typing import List
from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form,
    Request,
)
from fastapi.responses import Response

from models.document import Document
from utils.logging import llmmllogger
from utils.text_extraction import extract_text_content, get_file_metadata
from middleware.auth import get_user_id
from db import storage

logger = llmmllogger.bind(component="document_router")

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=Document)
async def upload_document(
    request: Request,
    conversation_id: int = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a document to a conversation.

    Args:
        request: FastAPI request object for auth
        conversation_id: ID of the conversation
        file: File to upload

    Returns:
        The created Document object
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.document:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Read file content
        content = await file.read()
        file_size = len(content)

        # Encode as base64 for storage
        content_base64 = base64.b64encode(content).decode("utf-8")

        # Extract text content for searchability
        filename = file.filename or "untitled"
        content_type = file.content_type or "application/octet-stream"
        text_content = extract_text_content(content_base64, content_type, filename)

        # Store the document
        document = await storage.document.store_document(
            message_id=conversation_id,
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            content=content_base64,
            text_content=text_content,
        )

        # Create memory embedding if we have text content and embedding service
        # TODO: Add embedding service integration when available
        if text_content and storage.memory:
            try:
                # Get file metadata for context
                metadata = get_file_metadata(filename, content_type, file_size)

                # Create embedding context
                embedding_text = f"File: {filename}\n"
                embedding_text += f"Type: {content_type}\n"
                if metadata["is_code"]:
                    embedding_text += "Category: Code file\n"
                elif metadata["is_image"]:
                    embedding_text += "Category: Image file\n"
                elif metadata["is_text"]:
                    embedding_text += "Category: Text document\n"
                embedding_text += f"Content:\n{text_content}"

                # TODO: Generate embedding when embedding service is available
                # embedding = await embedding_service.get_embedding(embedding_text)
                # if embedding:
                #     await storage.memory.store_memory(
                #         user_id=user_id,
                #         source=MemorySource.DOCUMENT,
                #         role="system",
                #         source_id=document.id,
                #         embeddings=[embedding],
                #     )

                logger.info(
                    f"Document {document.id} stored with text content extracted"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to process memory for document {document.id}: {e}"
                )
                # Don't fail the document storage if memory creation fails

        logger.info(
            f"Uploaded document {document.id} for conversation {conversation_id}"
        )
        return document

    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload file: {str(e)}"
        ) from e


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: int,
    request: Request,
):
    """Get a document by ID."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.document:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    document = await storage.document.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    request: Request,
):
    """Download a document."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.document:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    document = await storage.document.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Decode base64 content
        content_bytes = base64.b64decode(document.content)

        return Response(
            content=content_bytes,
            media_type=document.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{document.filename}"'
            },
        )
    except Exception as e:
        logger.error(f"Failed to decode document content: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to decode file content"
        ) from e


@router.get("/conversation/{conversation_id}", response_model=List[Document])
async def get_conversation_documents(
    conversation_id: int,
    request: Request,
):
    """Get all documents for a conversation."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized or not storage.document:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    documents = await storage.document.get_documents_for_conversation(conversation_id)
    return documents
