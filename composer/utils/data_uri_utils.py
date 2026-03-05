"""
Utilities for parsing and working with data URIs.
"""

import re
import base64
from typing import Optional, Tuple
from .logging import llmmllogger

logger = llmmllogger.bind(component="data_uri_utils")


def parse_data_uri(data_uri: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a data URI and extract its components.

    Args:
        data_uri: Data URI string in format "data:mime_type;base64,base64_data"

    Returns:
        Tuple of (mime_type, encoding, base64_data) if valid, None if invalid

    Example:
        >>> parse_data_uri("data:image/png;base64,iVBORw0KGgoA...")
        ('image/png', 'base64', 'iVBORw0KGgoA...')
    """
    if not data_uri or not isinstance(data_uri, str):
        return None

    # Match data URI format: data:mime_type;encoding,data
    pattern = r"^data:([^;]+);([^,]+),(.+)$"
    match = re.match(pattern, data_uri)

    if not match:
        logger.warning(f"Invalid data URI format: {data_uri[:50]}...")
        return None

    mime_type, encoding, data = match.groups()
    return mime_type, encoding, data


def extract_base64_from_data_uri(data_uri: str) -> Optional[str]:
    """
    Extract base64 encoded data from a data URI.

    Args:
        data_uri: Data URI string

    Returns:
        Base64 encoded string if valid, None if invalid or not base64 encoded
    """
    parsed = parse_data_uri(data_uri)
    if not parsed:
        return None

    mime_type, encoding, data = parsed

    if encoding.lower() != "base64":
        logger.warning(f"Data URI is not base64 encoded: {encoding}")
        return None

    return data


def extract_mime_type_from_data_uri(data_uri: str) -> Optional[str]:
    """
    Extract MIME type from a data URI.

    Args:
        data_uri: Data URI string

    Returns:
        MIME type string if valid, None if invalid
    """
    parsed = parse_data_uri(data_uri)
    if not parsed:
        return None

    mime_type, _, _ = parsed
    return mime_type


def create_data_uri(mime_type: str, base64_data: str, encoding: str = "base64") -> str:
    """
    Create a data URI from components.

    Args:
        mime_type: MIME type (e.g., "image/png")
        base64_data: Base64 encoded data
        encoding: Encoding type (default: "base64")

    Returns:
        Complete data URI string
    """
    return f"data:{mime_type};{encoding},{base64_data}"


def validate_data_uri(data_uri: str) -> bool:
    """
    Validate that a string is a properly formatted data URI.

    Args:
        data_uri: Data URI string to validate

    Returns:
        True if valid data URI, False otherwise
    """
    parsed = parse_data_uri(data_uri)
    if not parsed:
        return False

    mime_type, encoding, data = parsed

    # Check that data is valid base64 if encoding is base64
    if encoding.lower() == "base64":
        try:
            base64.b64decode(data, validate=True)
            return True
        except Exception as e:
            logger.warning(f"Invalid base64 data in URI: {e}")
            return False

    # For other encodings, just check that we have data
    return len(data) > 0


def get_decoded_data(data_uri: str) -> Optional[bytes]:
    """
    Get the decoded binary data from a data URI.

    Args:
        data_uri: Data URI string

    Returns:
        Decoded binary data if valid base64, None otherwise
    """
    base64_data = extract_base64_from_data_uri(data_uri)
    if not base64_data:
        return None

    try:
        return base64.b64decode(base64_data)
    except Exception as e:
        logger.error(f"Failed to decode base64 data: {e}")
        return None


def is_data_uri(uri: str) -> bool:
    """
    Check if a string is a data URI.

    Args:
        uri: String to check

    Returns:
        True if it's a data URI, False otherwise
    """
    if not uri or not isinstance(uri, str):
        return False
    return uri.startswith("data:")


def is_image_data_uri(data_uri: str) -> bool:
    """
    Check if a data URI contains image data.

    Args:
        data_uri: Data URI string

    Returns:
        True if it's an image data URI, False otherwise
    """
    mime_type = extract_mime_type_from_data_uri(data_uri)
    return mime_type is not None and mime_type.startswith("image/")


def is_audio_data_uri(data_uri: str) -> bool:
    """
    Check if a data URI contains audio data.

    Args:
        data_uri: Data URI string

    Returns:
        True if it's an audio data URI, False otherwise
    """
    mime_type = extract_mime_type_from_data_uri(data_uri)
    return mime_type is not None and mime_type.startswith("audio/")


def is_video_data_uri(data_uri: str) -> bool:
    """
    Check if a data URI contains video data.

    Args:
        data_uri: Data URI string

    Returns:
        True if it's a video data URI, False otherwise
    """
    mime_type = extract_mime_type_from_data_uri(data_uri)
    return mime_type is not None and mime_type.startswith("video/")
