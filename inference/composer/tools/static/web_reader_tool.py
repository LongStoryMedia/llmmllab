"""
Web content reader tool for fetching and reading content from URLs.

This tool provides direct content reading from web URLs, complementing the web search tool.
It's designed to fetch and extract readable text content from web pages, including:
- Static HTML pages
- Plain text and markdown files (e.g., GitHub raw URLs)
- Single Page Applications (SPA) via Playwright rendering
"""

import asyncio
from typing import Optional, Tuple

import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_core.tools import tool
from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="WebReader")

SPA_TEXT_THRESHOLD = 50
SPA_SCRIPT_RATIO = 0.5
MAX_CONTENT_LENGTH = 0  # 0 = no truncation

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

FRAMEWORK_MARKERS = (
    '<script type="module"',
    "window.__NUXT__",
    "__VUE__",
    "__reactInternalInstance",
    "webpackJsonp",
)


def _analyze_html(html_content: str) -> Tuple[str, bool]:
    """
    Parse HTML once. Return (cleaned_text, is_spa).

    SPA signals — computed before stripping chrome so we can inspect script
    dominance and empty root containers:
      - visible text length
      - script/style byte ratio
      - empty #root / #app / #vue-app / .app containers
      - framework markers in the raw HTML string
    """
    soup = BeautifulSoup(html_content, "html.parser")

    script_text = "".join(tag.get_text() for tag in soup.find_all(["script", "style"]))
    script_ratio = (len(script_text) / len(html_content)) if html_content else 0.0
    empty_root = any(
        container is not None and not container.get_text(strip=True)
        for container in (
            soup.find("div", id="root"),
            soup.find("div", id="app"),
            soup.find("div", id="vue-app"),
            soup.find("div", class_="app"),
        )
    )
    has_framework_marker = any(marker in html_content for marker in FRAMEWORK_MARKERS)

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    raw_text = soup.get_text()
    lines = (line.strip() for line in raw_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = " ".join(chunk for chunk in chunks if chunk)

    is_spa = (
        len(clean_text.strip()) < SPA_TEXT_THRESHOLD
        or script_ratio > SPA_SCRIPT_RATIO
        or empty_root
        or has_framework_marker
    )
    return clean_text, is_spa


def _extract_text_from_html(html_content: str) -> str:
    """Public helper (tests import this): parse HTML and return cleaned text."""
    text, _ = _analyze_html(html_content)
    return text


def _is_spa_detected(html_content: str, text_content: str) -> bool:
    """Public helper (tests import this). text_content is ignored — kept for back-compat."""
    _, is_spa = _analyze_html(html_content)
    return is_spa


def _truncate(text: str) -> str:
    if MAX_CONTENT_LENGTH and len(text) > MAX_CONTENT_LENGTH:
        return text[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated due to length...]"
    return text


async def _render_with_playwright(url: str) -> Optional[str]:
    """Render a URL with Playwright; return HTML, or None if unavailable/failed."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright not installed. Install with: pip install playwright")
        return None

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)

            for selector in ("body", "main", "#root", ".app", "article"):
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    break
                except Exception:
                    continue

            html_content = await page.content()
            await browser.close()
            logger.info("✅ SPA rendered successfully with Playwright")
            return html_content

    except Exception as e:
        logger.error(f"Playwright rendering failed: {str(e)}")
        return None


async def _process_html(url: str, html_content: str, allow_spa_fallback: bool) -> str:
    """Extract text from HTML; optionally fall back to Playwright on SPA pages."""
    text_content, is_spa = _analyze_html(html_content)

    if allow_spa_fallback and is_spa:
        logger.info("SPA detected, attempting Playwright rendering")
        rendered = await _render_with_playwright(url)
        if rendered:
            text_content, _ = _analyze_html(rendered)
        # If Playwright failed, fall through with the original text_content.

    text_content = _truncate(text_content)
    logger.info(f"✅ Successfully extracted {len(text_content)} characters from: {url}")
    return f"Content from {url}:\n\n{text_content}"


@tool
async def read_web_content(url: str, render_js: bool = False) -> str:
    """
    Read and extract text content from a web page URL.

    Handles HTML (static and SPA), plain text/markdown, JSON, and other text-based
    content. Auto-detects JavaScript-rendered pages and can fall back to Playwright;
    pass render_js=True to force Playwright up front.

    Args:
        url: The URL to read content from (must be http:// or https://)
        render_js: If True, skip aiohttp and render with Playwright directly.

    Returns:
        Clean text content from the web page, or error message if fetch fails
    """
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme not in ("http", "https"):
            return (
                f"Error: Invalid URL '{url}'. Only HTTP and HTTPS URLs are supported."
            )
    except Exception as e:
        return f"Error: Invalid URL format '{url}': {str(e)}"

    logger.info(f"📖 Reading web content from: {url}")

    if render_js:
        rendered = await _render_with_playwright(url)
        if not rendered:
            return "Error: Playwright rendering failed"
        return await _process_html(url, rendered, allow_spa_fallback=False)

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.get(
                url, headers=BROWSER_HEADERS, allow_redirects=True
            ) as response:
                if response.status >= 400:
                    return f"Error: HTTP {response.status} when accessing {url}"

                content_type = response.headers.get("content-type", "").lower()
                body = await response.text()

                if "text/html" in content_type:
                    return await _process_html(url, body, allow_spa_fallback=True)

                if "application/json" in content_type or "text/" in content_type:
                    return f"Content from {url}:\n\n{_truncate(body)}"

                return f"Error: URL does not appear to contain readable text (content-type: {content_type})"

    except asyncio.TimeoutError:
        return f"Error: Timeout when trying to access {url} (30 seconds)"
    except aiohttp.ClientError as e:
        return f"Error: Network error when accessing {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error reading web content from {url}: {str(e)}")
        return f"Error: Failed to read content from {url}: {str(e)}"
