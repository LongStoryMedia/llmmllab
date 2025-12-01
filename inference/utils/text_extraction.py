"""Text extraction utilities for file attachments."""

import base64
from typing import Optional, Dict, Any

from .file_extensions import (
    ALL_TEXT_EXTENSIONS,
    get_file_extension,
    get_file_metadata as get_file_metadata_base,
)


def extract_text_content(
    content: str, content_type: str, filename: str
) -> Optional[str]:
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
    is_text_by_mime = content_type.startswith("text/") or content_type in [
        "application/json",
        "application/xml",
        "application/x-yaml",
        "text/yaml",
    ]
    is_text_by_extension = get_file_extension(filename) in ALL_TEXT_EXTENSIONS

    if is_text_by_mime or is_text_by_extension:
        try:
            # For text/plain, try to decode as base64 first, fall back to plain text
            if content_type == "text/plain":
                decoded_content = _try_base64_decode(content)
                if decoded_content is not None:
                    return decoded_content
                else:
                    # Not base64, assume it's already plain text
                    return content
            else:
                # Assume base64 encoded for other text types
                decoded_content = base64.b64decode(content).decode("utf-8")
                return decoded_content
        except Exception:
            return None

    # For binary files (images, PDFs, etc.), return filename for basic searchability
    return f"File: {filename}"


def _try_base64_decode(content: str) -> Optional[str]:
    """Try to decode content as base64. Return decoded content or None if not base64."""
    try:
        # Try to decode as base64
        decoded_bytes = base64.b64decode(content, validate=True)
        decoded_text = decoded_bytes.decode("utf-8")
        return decoded_text
    except Exception:
        return None


def get_file_metadata(
    filename: str, content_type: str, file_size: int
) -> Dict[str, Any]:
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
