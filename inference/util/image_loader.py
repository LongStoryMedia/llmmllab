import os
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


def load_image_with_headers(image_url_or_path, headers=None):
    """
    Load an image from a URL or file path, supporting custom headers for authentication.

    Args:
        image_url_or_path (str): URL or local path to the image
        headers (dict, optional): HTTP headers to include in the request

    Returns:
        PIL.Image.Image: The loaded image

    Raises:
        ValueError: If the image cannot be loaded
    """
    # Check if it's a URL or a local path
    parsed = urlparse(image_url_or_path)

    if parsed.scheme in ('http', 'https'):
        # It's a URL, use requests with headers if provided
        try:
            logger.info(f"Loading image from URL: {image_url_or_path}")
            response = requests.get(image_url_or_path, headers=headers, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Failed to load image from URL: {e}")
            raise ValueError(f"Failed to load image from URL: {e}")
    else:
        # It's a local path, use PIL directly
        try:
            if not os.path.exists(image_url_or_path):
                raise FileNotFoundError(f"Image file not found: {image_url_or_path}")
            logger.info(f"Loading image from local path: {image_url_or_path}")
            return Image.open(image_url_or_path)
        except Exception as e:
            logger.error(f"Failed to load image from local path: {e}")
            raise ValueError(f"Failed to load image from local path: {e}")
