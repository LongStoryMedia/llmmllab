"""
Web content reader tool for fetching and reading content from URLs.

This tool provides direct content reading from web URLs, complementing the web search tool.
It's designed to fetch and extract readable text content from web pages.
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from composer.utils.logging import llmmllogger


logger = llmmllogger.logger.bind(component="WebReader")


@tool
async def read_web_content(url: str) -> str:
    """
    Read and extract text content from a web page URL.

    This tool fetches the HTML content from a given URL and extracts the readable text,
    removing HTML tags and formatting for clean consumption by AI models.
    If you get a response code greater than or equal to 400, you may retry, but only up to 2 times. After that, return an error message.

    Args:
        url: The URL to read content from (must be http:// or https://)

    Returns:
        Clean text content from the web page, or error message if fetch fails
    """
    # Validate URL
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme not in ["http", "https"]:
            return (
                f"Error: Invalid URL '{url}'. Only HTTP and HTTPS URLs are supported."
            )
    except Exception as e:
        return f"Error: Invalid URL format '{url}': {str(e)}"

    logger.info(f"📖 Reading web content from: {url}")

    try:
        # Set up headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        # Fetch content with timeout
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.get(
                url, headers=headers, allow_redirects=True
            ) as response:
                if response.status >= 400:
                    return f"Error: HTTP {response.status} when accessing {url}"

                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    return f"Error: URL does not appear to contain HTML content (content-type: {content_type})"

                html_content = await response.text()

        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Get text content
        text_content = soup.get_text()

        # Clean up whitespace and formatting
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = " ".join(chunk for chunk in chunks if chunk)

        # Limit content length to avoid overwhelming the context
        max_length = 8000  # Reasonable limit for AI context
        if len(clean_text) > max_length:
            clean_text = (
                clean_text[:max_length] + "\n\n[Content truncated due to length...]"
            )

        if not clean_text.strip():
            return f"Warning: No readable text content found at {url}"

        logger.info(f"✅ Successfully read {len(clean_text)} characters from: {url}")

        return f"Content from {url}:\n\n{clean_text}"

    except asyncio.TimeoutError:
        return f"Error: Timeout when trying to access {url} (30 seconds)"
    except aiohttp.ClientError as e:
        return f"Error: Network error when accessing {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error reading web content from {url}: {str(e)}")
        return f"Error: Failed to read content from {url}: {str(e)}"
