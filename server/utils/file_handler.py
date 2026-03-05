"""
Utility functions for handling file attachments in llama.cpp compatible format.
Handles base64 decoding, file storage, and URL generation for images and files.
"""

import os
import base64
import hashlib
import mimetypes
from typing import Optional, Tuple
from datetime import datetime
import uuid

from server.utils.logging import llmmllogger
from .file_extensions import ALL_TEXT_EXTENSIONS, get_file_extension as get_ext_from_filename

logger = llmmllogger.bind(component="file_handler")

# Supported image formats for llama.cpp
SUPPORTED_IMAGE_FORMATS = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}


def get_file_extension(mime_type: str) -> str:
    """Get appropriate file extension for MIME type."""
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext

    # Fallback for common types
    mime_to_ext = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/json": ".json",
    }

    return mime_to_ext.get(mime_type, ".bin")


def generate_safe_filename(
    original_name: Optional[str], mime_type: str, content: bytes
) -> str:
    """Generate a safe, unique filename for stored content."""
    # Create hash of content for uniqueness and deduplication
    content_hash = hashlib.md5(content).hexdigest()[:12]

    # Use timestamp for additional uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get appropriate extension
    ext = get_file_extension(mime_type)

    # Create safe filename
    if original_name:
        # Sanitize original name
        safe_name = "".join(c for c in original_name if c.isalnum() or c in "._-")
        safe_name = safe_name[:20]  # Limit length
        return f"{timestamp}_{content_hash}_{safe_name}{ext}"
    else:
        return f"{timestamp}_{content_hash}{ext}"


def decode_and_save_image(
    base64_data: str, mime_type: str, filename: Optional[str] = None
) -> Tuple[str, str]:
    """
    Decode base64 image data and save to static directory.

    Args:
        base64_data: Base64 encoded image data
        mime_type: MIME type of the image
        filename: Optional original filename

    Returns:
        Tuple of (local_path, public_url)
    """
    try:
        # Get IMAGE_DIR dynamically to support testing
        from server.config import IMAGE_DIR

        # Decode base64 data
        image_data = base64.b64decode(base64_data)

        # Generate safe filename
        safe_filename = generate_safe_filename(filename, mime_type, image_data)

        # Ensure images directory exists
        os.makedirs(IMAGE_DIR, exist_ok=True)

        # Full path for saving
        file_path = os.path.join(IMAGE_DIR, safe_filename)

        # Write image data
        with open(file_path, "wb") as f:
            f.write(image_data)

        # Generate public URL for accessing the image
        public_url = f"/static/images/view/{safe_filename}"

        logger.info(f"Saved image: {safe_filename} ({len(image_data)} bytes)")
        return file_path, public_url

    except Exception as e:
        logger.error(f"Failed to decode and save image: {e}")
        raise


def extract_text_from_file(
    base64_data: str, mime_type: str, filename: Optional[str] = None
) -> str:
    """
    Extract text content from non-image files for llama.cpp processing.

    Args:
        base64_data: Base64 encoded file data
        mime_type: MIME type of the file
        filename: Optional original filename

    Returns:
        Text representation of file content
    """
    try:
        # Decode base64 data
        file_data = base64.b64decode(base64_data)

        # Handle different file types
        if mime_type.startswith("text/"):
            # Text files - decode as UTF-8 and wrap in XML tags
            try:
                text_content = file_data.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_data.decode("latin-1", errors="replace")

            # Wrap content in XML tags with filename
            safe_filename = filename or "text-file.txt"
            return f'<file name="{safe_filename}">{text_content}</file>'

        elif mime_type == "application/json":
            # JSON files - format and wrap in XML tags
            try:
                import json

                json_content = json.dumps(
                    json.loads(file_data.decode("utf-8")), indent=2
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                json_content = file_data.decode("utf-8", errors="replace")

            # Wrap JSON content in XML tags
            safe_filename = filename or "data.json"
            return f'<file name="{safe_filename}">{json_content}</file>'

        elif mime_type in ("application/xml", "text/xml"):
            # XML files
            try:
                xml_content = file_data.decode("utf-8")
            except UnicodeDecodeError:
                xml_content = file_data.decode("latin-1", errors="replace")

            safe_filename = filename or "document.xml"
            return f'<file name="{safe_filename}">{xml_content}</file>'

        elif mime_type in ("application/javascript", "text/javascript"):
            # JavaScript files
            try:
                js_content = file_data.decode("utf-8")
            except UnicodeDecodeError:
                js_content = file_data.decode("latin-1", errors="replace")

            safe_filename = filename or "script.js"
            return f'<file name="{safe_filename}">{js_content}</file>'

        elif mime_type == "text/html":
            # HTML files
            try:
                html_content = file_data.decode("utf-8")
            except UnicodeDecodeError:
                html_content = file_data.decode("latin-1", errors="replace")

            safe_filename = filename or "document.html"
            return f'<file name="{safe_filename}">{html_content}</file>'

        elif mime_type == "text/css":
            # CSS files
            try:
                css_content = file_data.decode("utf-8")
            except UnicodeDecodeError:
                css_content = file_data.decode("latin-1", errors="replace")

            safe_filename = filename or "styles.css"
            return f'<file name="{safe_filename}">{css_content}</file>'

        elif mime_type == "application/pdf":
            # PDF files - extract text if possible
            return (
                f"[PDF Document: {filename or 'attachment'} ({len(file_data)} bytes)]"
            )

        elif mime_type.startswith("audio/"):
            # Audio files
            return f"[Audio File: {filename or 'attachment'} ({mime_type}, {len(file_data)} bytes)]"

        elif mime_type.startswith("video/"):
            # Video files
            return f"[Video File: {filename or 'attachment'} ({mime_type}, {len(file_data)} bytes)]"

        else:
            # Check for text-based files by extension if MIME type is generic
            if filename:
                ext = get_ext_from_filename(filename)

                if ext in ALL_TEXT_EXTENSIONS:
                    try:
                        text_content = file_data.decode("utf-8")
                        safe_filename = filename or f"code{ext}"
                        return f'<file name="{safe_filename}">{text_content}</file>'
                    except UnicodeDecodeError:
                        try:
                            text_content = file_data.decode("latin-1", errors="replace")
                            safe_filename = filename or f"code{ext}"
                            return f'<file name="{safe_filename}">{text_content}</file>'
                        except Exception:
                            pass  # Fall through to generic file handling

            # Generic files - not text-based
            return f"[File: {filename or 'attachment'} ({mime_type}, {len(file_data)} bytes)]"

    except Exception as e:
        logger.warning(f"Failed to extract text from file: {e}")
        return f"[File: {filename or 'attachment'} (unable to process)]"


def is_image_format(mime_type: str) -> bool:
    """Check if MIME type represents a supported image format."""
    return mime_type.lower() in SUPPORTED_IMAGE_FORMATS


def cleanup_old_files(max_age_hours: int = 24) -> None:
    """Clean up old temporary files."""
    try:
        from server.config import IMAGE_DIR
        import time

        cutoff_time = time.time() - (max_age_hours * 3600)

        for filename in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, filename)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                os.remove(file_path)
                logger.debug(f"Cleaned up old file: {filename}")

    except Exception as e:
        logger.warning(f"Failed to clean up old files: {e}")
