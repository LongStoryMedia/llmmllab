"""Text extraction utilities for file attachments."""

import base64
from typing import Optional, Dict, Any


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
    
    # Handle text files directly
    if content_type.startswith('text/'):
        try:
            # If it's base64 encoded, decode it
            if content_type != 'text/plain' or _is_base64_encoded(content):
                decoded_content = base64.b64decode(content).decode('utf-8')
                return decoded_content
            else:
                # It's already plain text
                return content
        except Exception:
            return None
    
    # Handle specific file types
    if content_type == 'application/json':
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        except Exception:
            return None
    
    if content_type in ['application/xml', 'application/x-yaml', 'text/yaml']:
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        except Exception:
            return None
            
    # For markdown, code files, etc.
    if (content_type.startswith('text/') or 
        filename.endswith(('.md', '.py', '.js', '.ts', '.html', '.css', '.sql', '.sh', '.yaml', '.yml', '.json', '.xml'))):
        try:
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
    file_extension = filename.split('.')[-1] if '.' in filename else ''
    
    return {
        'filename': filename,
        'extension': file_extension,
        'content_type': content_type,
        'file_size': file_size,
        'is_text': content_type.startswith('text/') or file_extension in ['md', 'py', 'js', 'ts', 'html', 'css', 'sql', 'sh', 'yaml', 'yml', 'json', 'xml'],
        'is_image': content_type.startswith('image/'),
        'is_code': file_extension in ['py', 'js', 'ts', 'html', 'css', 'sql', 'sh', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'php'],
    }