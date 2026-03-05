"""
Message transformation utilities.

Handles transformation of incoming messages, including converting file content
to document attachments and other message processing tasks.
"""

from typing import Optional
from datetime import datetime, timezone

from server.models import Message, MessageContent, MessageContentType, Document
from server.utils.data_uri_utils import (
    is_data_uri,
    extract_base64_from_data_uri,
    extract_mime_type_from_data_uri,
    get_decoded_data,
)
from server.utils.text_extraction import extract_text_content
from server.utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_transformation")


async def transform_file_content_to_documents(message: Message, user_id: str) -> Message:
    """
    Transform message content of type 'file' into Document objects in message.documents.
    
    This function processes incoming messages from the UI and converts any content items
    with type="file" into proper Document objects that can be stored in the database.
    Uses existing utility functions to avoid code duplication.
    
    Args:
        message: The message to transform
        user_id: ID of the user sending the message
        
    Returns:
        Transformed message with file content moved to documents array
    """
    if not message.content:
        return message
    
    # Initialize documents list if not present
    if message.documents is None:
        message.documents = []
    
    # Process content items and extract files
    new_content = []
    
    for content_item in message.content:
        if content_item.type == MessageContentType.FILE:
            # Extract file information using existing utilities
            document = await create_document_from_content(
                content_item=content_item,
                user_id=user_id
            )
            
            if document:
                # Add to documents array
                message.documents.append(document)
                
                # Replace the file content with a text reference
                text_ref = MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"[File: {document.filename}]"
                )
                new_content.append(text_ref)
            else:
                # Keep original content if document creation failed
                new_content.append(content_item)
            
        else:
            # Keep non-file content as-is
            new_content.append(content_item)
    
    # Update message content
    message.content = new_content
    
    return message


async def create_document_from_content(
    content_item: MessageContent, user_id: str
) -> Optional[Document]:
    """
    Create a Document object from a MessageContent item using existing file utilities.
    
    Args:
        content_item: MessageContent with type FILE
        user_id: ID of the user uploading the file
        
    Returns:
        Document object or None if creation fails
    """
    try:
        # Get filename from name field or generate default
        filename = content_item.name or "attachment"
        
        # Extract content and MIME type using existing utilities
        content_type = None
        base64_content = None
        
        # Try to extract from data URI first (preferred method)
        if content_item.url and is_data_uri(content_item.url):
            content_type = extract_mime_type_from_data_uri(content_item.url)
            base64_content = extract_base64_from_data_uri(content_item.url)
        
        # Fallback to format field for content type
        if not content_type:
            content_type = content_item.format or "application/octet-stream"
            
        # Fallback to text field for content
        if not base64_content:
            base64_content = content_item.text or ""
            
        # Calculate file size using utility
        try:
            decoded_data = get_decoded_data(content_item.url) if content_item.url and is_data_uri(content_item.url) else None
            if decoded_data:
                file_size = len(decoded_data)
            else:
                # Fallback calculation
                import base64
                file_size = len(base64.b64decode(base64_content))
        except Exception:
            # If base64 decode fails, use string length as approximation
            file_size = len(base64_content)
            
        # Extract text content for searchability using existing utility
        text_content = extract_text_content(base64_content, content_type, filename)
        
        # Create Document object
        document = Document(
            message_id=0,  # Temporary, will be set by message storage
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            content=base64_content,
            text_content=text_content,
            created_at=datetime.now(timezone.utc),
        )
        
        logger.info(f"Created document from content: {filename} ({content_type}, {file_size} bytes)")
        return document
        
    except Exception as e:
        logger.warning(f"Failed to create document from content: {e}")
        return None