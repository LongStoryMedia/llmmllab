"""
Web content reader tool for fetching and reading content from URLs.

This tool provides direct content reading from web URLs, complementing the web search tool.
It's designed to fetch and extract readable text content from web pages, including:
- Static HTML pages
- Plain text and markdown files (e.g., GitHub raw URLs)
- Single Page Applications (SPA) via Playwright rendering
"""

import asyncio
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_core.tools import tool
from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="WebReader")

# SPA detection thresholds
SPA_TEXT_THRESHOLD = 50  # Minimum characters of visible text
SPA_SCRIPT_RATIO = 0.5   # If script tags > 50% of content, likely SPA


def _is_spa_detected(html_content: str, text_content: str) -> bool:
    """
    Detect if a page is likely a Single Page Application (SPA).

    SPA indicators:
    - Very little visible text content
    - Heavy script tag usage
    - Empty root containers (div#root, #app, etc.)
    - Common framework markers

    Args:
        html_content: Raw HTML string
        text_content: Extracted text content (without tags)

    Returns:
        True if page appears to be an SPA requiring JavaScript rendering
    """
    # Check text content length
    if len(text_content.strip()) < SPA_TEXT_THRESHOLD:
        logger.debug("SPA detection: Very little text content")
        return True

    # Parse HTML to analyze structure
    soup = BeautifulSoup(html_content, "html.parser")

    # Check script tag dominance
    script_tags = soup.find_all(["script", "style"])
    script_text = "".join(tag.get_text() for tag in script_tags)
    if len(html_content) > 0 and len(script_text) / len(html_content) > SPA_SCRIPT_RATIO:
        logger.debug("SPA detection: High script/content ratio")
        return True

    # Check for empty SPA root containers
    root_containers = [
        soup.find("div", id="root"),
        soup.find("div", id="app"),
        soup.find("div", id="vue-app"),
        soup.find("div", class_="app"),
    ]
    for container in root_containers:
        if container and not container.get_text(strip=True):
            logger.debug("SPA detection: Empty root container found")
            return True

    # Check for framework-specific markers
    framework_markers = [
        '<script type="module"',  # Vite/ES modules
        'window.__NUXT__',  # Nuxt.js
        '__VUE__',  # Vue.js
        '__reactInternalInstance',  # React
        "webpackJsonp",  # Webpack bundles
    ]
    for marker in framework_markers:
        if marker in html_content:
            logger.debug(f"SPA detection: Framework marker '{marker}' found")
            return True

    return False


def _extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content.

    Args:
        html_content: Raw HTML string

    Returns:
        Clean text content
    """
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

    return clean_text


async def _render_with_playwright(url: str) -> Optional[str]:
    """
    Render a URL using Playwright to capture JavaScript-rendered content.

    Args:
        url: URL to render

    Returns:
        Rendered HTML content, or None if Playwright is unavailable
    """
    try:
        # Lazy import Playwright
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright not installed. Install with: pip install playwright")
        return None

    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to URL
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for common content indicators
            for selector in ["body", "main", "#root", ".app", "article"]:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    break
                except Exception:
                    continue

            # Get rendered HTML
            html_content = await page.content()
            await browser.close()

            logger.info("✅ SPA rendered successfully with Playwright")
            return html_content

    except Exception as e:
        logger.error(f"Playwright rendering failed: {str(e)}")
        return None


@tool
async def read_web_content(url: str, render_js: bool = False) -> str:
    """
    Read and extract text content from a web page URL.

    This tool fetches content from a given URL and extracts readable text,
    handling multiple content types:
    - HTML pages (static and SPA)
    - Plain text and markdown files
    - JSON and other text-based formats

    For SPA detection, the tool automatically identifies JavaScript-rendered
    pages and can use Playwright for rendering when render_js=True.

    Args:
        url: The URL to read content from (must be http:// or https://)
        render_js: If True, use Playwright to render JavaScript content.
                   If False, will auto-detect SPA and may fallback to Playwright.

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

    # Track if we've tried Playwright
    playwright_tried = False

    async def _fetch_and_process(use_playwright: bool = False) -> str:
        """Internal fetch and process function."""
        nonlocal playwright_tried

        try:
            # Set up headers to mimic a real browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            html_content = None
            content_type = None

            if use_playwright:
                playwright_tried = True
                rendered = await _render_with_playwright(url)
                if rendered:
                    html_content = rendered
                else:
                    return "Error: Playwright rendering failed"
            else:
                # Fetch content with aiohttp
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

                        # Handle non-HTML text content (plain text, markdown, JSON)
                        if "text/plain" in content_type or "text/markdown" in content_type:
                            logger.info("Detected plain text content, reading raw content")
                            text_content = await response.text()
                            return f"Content from {url}:\n\n{text_content}"

                        # Handle JSON content
                        if "application/json" in content_type:
                            logger.info("Detected JSON content, reading raw content")
                            json_content = await response.text()
                            return f"Content from {url}:\n\n{json_content}"

                        # For HTML content, get the raw HTML
                        if "text/html" not in content_type:
                            # Try to read anyway if it's some other text type
                            if "text/" in content_type:
                                logger.info(f"Reading {content_type} content as text")
                                text_content = await response.text()
                                return f"Content from {url}:\n\n{text_content}"

                            return f"Error: URL does not appear to contain HTML content (content-type: {content_type})"

                        html_content = await response.text()

            # Parse and extract text
            if html_content:
                text_content = _extract_text_from_html(html_content)

                # SPA detection and fallback
                if not render_js and not playwright_tried:
                    if _is_spa_detected(html_content, text_content):
                        logger.info("SPA detected, attempting Playwright rendering")
                        return await _fetch_and_process(use_playwright=True)

                logger.info(f"✅ Successfully extracted {len(text_content)} characters from: {url}")
                return f"Content from {url}:\n\n{text_content}"

            return "Error: Failed to retrieve content from URL"

        except asyncio.TimeoutError:
            return f"Error: Timeout when trying to access {url} (30 seconds)"
        except aiohttp.ClientError as e:
            return f"Error: Network error when accessing {url}: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error reading web content from {url}: {str(e)}")
            return f"Error: Failed to read content from {url}: {str(e)}"

    return await _fetch_and_process(use_playwright=render_js)