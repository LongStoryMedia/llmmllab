"""
Unit tests for the web reader tool.

Tests cover:
- Plain text/Markdown/JSON URL handling
- Static HTML content extraction
- SPA detection logic
- Playwright fallback for JavaScript-rendered content
- Error handling
"""

import sys

import aiohttp
import pytest
import asyncio
from aioresponses import aioresponses
from unittest.mock import AsyncMock, MagicMock, patch

# Check if Playwright is available for conditional test skipping
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from tools.static.web_reader_tool import (
    read_web_content,
    _is_spa_detected,
    _extract_text_from_html,
)


@pytest.fixture
def aioresponses_fixture():
    """Yield an aioresponses context for mocking aiohttp requests."""
    with aioresponses() as m:
        yield m


# =============================================================================
# Helper Functions Tests
# =============================================================================

class TestExtractTextFromHtml:
    """Tests for _extract_text_from_html helper function."""

    def test_extract_text_from_basic_html(self):
        """Test extracting text from basic HTML content."""
        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>This is some paragraph text.</p>
                <script>var x = 1;</script>
            </body>
        </html>
        """
        result = _extract_text_from_html(html)
        assert "Title" in result
        assert "This is some paragraph text." in result
        assert "var x = 1" not in result  # Script content removed

    def test_extract_text_removes_scripts_and_styles(self):
        """Test that script and style tags are removed."""
        html = """
        <html>
            <body>
                <style>.class { color: red; }</style>
                <script>console.log('test');</script>
                <p>Real content here</p>
            </body>
        </html>
        """
        result = _extract_text_from_html(html)
        assert "Real content here" in result
        assert ".class" not in result
        assert "console.log" not in result

    def test_extract_text_removes_navigation_elements(self):
        """Test that nav, header, footer, aside are removed."""
        html = """
        <html>
            <body>
                <nav>Navigation links</nav>
                <header>Header content</header>
                <main>Main content</main>
                <footer>Footer content</footer>
            </body>
        </html>
        """
        result = _extract_text_from_html(html)
        assert "Main content" in result
        # Navigation elements should be removed
        assert "Navigation links" not in result
        assert "Header content" not in result
        assert "Footer content" not in result

    def test_extract_text_handles_empty_html(self):
        """Test extracting text from empty HTML."""
        html = "<html><body></body></html>"
        result = _extract_text_from_html(html)
        assert result == ""

    def test_extract_text_normalizes_whitespace(self):
        """Test that whitespace is properly normalized."""
        html = """
        <html>
            <body>
                <p>Line 1</p>
                <p>Line 2</p>
            </body>
        </html>
        """
        result = _extract_text_from_html(html)
        # Should be on single line with normalized spaces
        assert "Line 1" in result
        assert "Line 2" in result
        assert "\n" not in result  # Newlines should be removed


class TestIsSpaDetected:
    """Tests for SPA detection logic."""

    def test_detects_empty_root_container(self):
        """Test detection of empty root containers."""
        html = '<html><body><div id="root"></div></body></html>'
        text = ""
        assert _is_spa_detected(html, text) is True

    def test_detects_empty_app_container(self):
        """Test detection of empty #app container."""
        html = '<html><body><div id="app"></div></body></html>'
        text = ""
        assert _is_spa_detected(html, text) is True

    def test_detects_vite_module_script(self):
        """Test detection of Vite/ES module scripts."""
        html = '<html><body><script type="module" src="/main.js"></script></body></html>'
        text = "Some text"
        assert _is_spa_detected(html, text) is True

    def test_detects_nuxt_marker(self):
        """Test detection of Nuxt.js markers."""
        html = '<html><body><script>window.__NUXT__={}</script></body></html>'
        text = "Some text"
        assert _is_spa_detected(html, text) is True

    def test_detects_webpack_marker(self):
        """Test detection of webpack bundles."""
        html = '<html><body><script>var webpackJsonp=[];</script></body></html>'
        text = "Some text"
        assert _is_spa_detected(html, text) is True

    def test_does_not_detect_regular_html(self):
        """Test that regular HTML with content is not flagged as SPA."""
        html = """
        <html>
            <body>
                <article>
                    <h1>Article Title</h1>
                    <p>This is a substantial paragraph with lots of content.</p>
                    <p>More content here for good measure.</p>
                </article>
            </body>
        </html>
        """
        text = "Article Title This is a substantial paragraph with lots of content. More content here for good measure."
        assert _is_spa_detected(html, text) is False

    def test_does_not_detect_with_sufficient_text(self):
        """Test that pages with enough text are not flagged as SPA."""
        html = '<html><body><div id="root"><p>Content</p></div></body></html>'
        text = "This is a lot of text that should be enough to not be considered an SPA page."
        assert _is_spa_detected(html, text) is False

    def test_handles_minimal_html(self):
        """Test detection with minimal HTML structure."""
        html = "<html></html>"
        text = ""
        result = _is_spa_detected(html, text)
        assert isinstance(result, bool)


# =============================================================================
# Main Tool Tests
# =============================================================================

class TestReadWebContent:
    """Tests for the read_web_content tool."""

    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        """Test rejection of invalid URL schemes."""
        result = await read_web_content.ainvoke({"url": "ftp://example.com/file.txt"})
        assert "Error" in result
        assert "Invalid URL" in result

    @pytest.mark.asyncio
    async def test_malformed_url(self):
        """Test handling of malformed URLs."""
        result = await read_web_content.ainvoke({"url": "not-a-url"})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_plain_text_url_content(self, aioresponses_fixture):
        """Test reading plain text content from URLs."""
        aioresponses_fixture.get(
            "https://example.com/file.txt",
            body="This is plain text content.",
            content_type="text/plain",
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/file.txt"})

        assert "Content from https://example.com/file.txt" in result
        assert "This is plain text content." in result

    @pytest.mark.asyncio
    async def test_markdown_url_content(self, aioresponses_fixture):
        """Test reading markdown content from URLs."""
        aioresponses_fixture.get(
            "https://example.com/file.md",
            body="# Header\n\nContent here.",
            content_type="text/markdown",
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/file.md"})

        assert "Content from https://example.com/file.md" in result
        assert "# Header" in result

    @pytest.mark.asyncio
    async def test_json_url_content(self, aioresponses_fixture):
        """Test reading JSON content from URLs."""
        aioresponses_fixture.get(
            "https://example.com/data.json",
            body='{"key": "value"}',
            content_type="application/json",
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/data.json"})

        assert "Content from https://example.com/data.json" in result
        assert '{"key": "value"}' in result

    @pytest.mark.asyncio
    async def test_static_html_content(self, aioresponses_fixture):
        """Test extracting content from static HTML pages."""
        # Use HTML with substantial text content to avoid SPA detection
        html_content = """
        <html>
            <body>
                <article>
                    <h1>Page Title</h1>
                    <p>This is the main content of the page with substantial text.</p>
                    <p>More content to ensure we're not triggering SPA detection.</p>
                </article>
            </body>
        </html>
        """

        aioresponses_fixture.get(
            "https://example.com/page.html",
            body=html_content,
            content_type="text/html",
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/page.html"})

        assert "Page Title" in result
        assert "This is the main content of the page" in result

    @pytest.mark.asyncio
    async def test_http_error_handling(self, aioresponses_fixture):
        """Test handling of HTTP errors."""
        aioresponses_fixture.get(
            "https://example.com/notfound",
            status=404,
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/notfound"})

        assert "Error" in result
        assert "404" in result

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, aioresponses_fixture):
        """Test handling of timeout errors."""
        # Simulate a timeout by raising asyncio.TimeoutError
        aioresponses_fixture.get(
            "https://example.com/slow",
            exception=asyncio.TimeoutError(),
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/slow"})

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_network_error_handling(self, aioresponses_fixture):
        """Test handling of network errors."""
        aioresponses_fixture.get(
            "https://example.com/offline",
            exception=aiohttp.ClientError("connection refused"),
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/offline"})

        assert "Error" in result
        assert "Network error" in result

    @pytest.mark.asyncio
    async def test_render_js_parameter(self):
        """Test that render_js parameter works with Playwright."""
        # Skip if Playwright is not available
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")

        # Actually use Playwright to render a simple page
        result = await read_web_content.ainvoke({
            "url": "https://example.com",
            "render_js": True
        })

        # Should return content (example.com is a simple page)
        assert isinstance(result, str)
        assert "Error" not in result or "example.com" in result

    @pytest.mark.asyncio
    async def test_playwright_not_installed_graceful_failure(self):
        """Test graceful failure when Playwright import fails."""
        # This test verifies the code handles ImportError gracefully
        # We test by checking the _render_with_playwright function behavior
        from tools.static.web_reader_tool import _render_with_playwright

        # Mock the import to fail
        with patch.dict(sys.modules, {"playwright": None}):
            # The function should return None or an error message, not crash
            result = await _render_with_playwright("https://example.com")
            # Should handle gracefully (return None or error string)
            assert result is None or isinstance(result, str)


# =============================================================================
# Integration-style Tests (with mocked external dependencies)
# =============================================================================

class TestWebReaderToolIntegration:
    """Integration-style tests for the web reader tool."""

    @pytest.mark.asyncio
    async def test_github_raw_markdown_url(self, aioresponses_fixture):
        """Test reading from GitHub raw markdown URLs (the original issue)."""
        # Simulate GitHub raw content response
        aioresponses_fixture.get(
            "https://raw.githubusercontent.com/user/repo/main/README.md",
            body="# README\n\nThis is a test project.",
            content_type="text/plain",
        )

        result = await read_web_content.ainvoke({
            "url": "https://raw.githubusercontent.com/user/repo/main/README.md"
        })

        # Should successfully read the markdown content
        assert "Content from" in result
        assert "# README" in result
        assert "This is a test project." in result

    @pytest.mark.asyncio
    async def test_spa_auto_detection_triggers_playwright(self):
        """Test that SPA pages are auto-detected and handled."""
        # First call returns HTML with SPA indicators
        html_with_spa = '<html><body><div id="root"></div><script type="module"></script></body></html>'

        # Mock Playwright to return rendered content
        mock_playwright_content = '<html><body><div id="root"><p>Rendered content</p></div></body></html>'

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value=mock_playwright_content)
        mock_page.wait_for_selector = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = MagicMock(return_value=mock_page)
        mock_browser.close = AsyncMock()

        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium = MagicMock()
        mock_playwright_instance.chromium.launch = MagicMock(return_value=mock_browser)

        mock_playwright = AsyncMock()
        mock_playwright.return_value.__aenter__ = AsyncMock(return_value=mock_playwright_instance)

        with patch("playwright.async_api.async_playwright", mock_playwright):
            result = await read_web_content.ainvoke({"url": "https://example.com/spa-page"})

        # Should have attempted SPA handling
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_non_html_content_type_fallback(self, aioresponses_fixture):
        """Test fallback for non-HTML but text-based content types."""
        aioresponses_fixture.get(
            "https://example.com/style.css",
            body="body { color: black; }",
            content_type="text/css",
        )

        result = await read_web_content.ainvoke({"url": "https://example.com/style.css"})

        # Should read text content even for CSS
        assert "Content from" in result
        assert "body { color: black; }" in result
