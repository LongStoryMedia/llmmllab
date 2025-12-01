"""Text extraction utilities for file attachments."""

import base64
from typing import Optional, Dict, Any

from .file_extensions import ALL_TEXT_EXTENSIONS, get_file_extension, get_file_metadata as get_file_metadata_base


def extract_text_content(content: str, content_type: str, filename: str) -> Optional[str]:
    """
    Extract text content from file content for embedding and search purposes.
    
    Args:
        content: Base64 encoded file content or plain text
        content_type: MIME type of the file
        filename: Original filename
        
    Returns:
        Extracted text content or None if no text can be extracted
    """
    
    # Check if this is a text-based file (by MIME type or extension)
    is_text_by_mime = (
        content_type.startswith('text/') or
        content_type in ['application/json', 'application/xml', 'application/x-yaml', 'text/yaml']
    )
    is_text_by_extension = get_file_extension(filename) in ALL_TEXT_EXTENSIONS
    
    if is_text_by_mime or is_text_by_extension:
        try:
            # For text/plain, check if it's already decoded or needs base64 decoding
            if content_type == 'text/plain' and not _is_base64_encoded(content):
                # Already plain text, return as-is
                return content
            else:
                # Assume base64 encoded, decode it
                decoded_content = base64.b64decode(content).decode('utf-8')
                return decoded_content
        except Exception:
            return None
    
    # For binary files (images, PDFs, etc.), return filename for basic searchability
    return f"File: {filename}"


def _is_base64_encoded(content: str) -> bool:
    """Check if content appears to be base64 encoded."""
    try:
        # Try to decode as base64
        base64.b64decode(content, validate=True)
        # If it decodes without error, it's likely base64 encoded
        return True
    except Exception:
        return False


def get_file_metadata(filename: str, content_type: str, file_size: int) -> Dict[str, Any]:
    """
    Extract metadata from file information for embedding context.
    
    Args:
        filename: Original filename
        content_type: MIME type
        file_size: Size in bytes
        
    Returns:
        Dictionary with file metadata
    """
    # Use the centralized metadata function
    return get_file_metadata_base(filename, content_type, file_size)