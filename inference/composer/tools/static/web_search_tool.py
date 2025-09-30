"""
Static web search tool using SearxNG provider.

This tool performs web searches with consistent behavior using
SearxNG as the search provider (running in the same cluster).

Configuration:
- Default engines: Google, Bing, DuckDuckGo, Startpage for comprehensive coverage
- Structured results API for reliable parsing
- Configurable engines, categories, and search parameters
- Specialized search tools for academic, news, technical, and shopping searches

Usage:
    # Default general web search
    tool = WebSearchTool()
    result = await tool._arun("machine learning trends 2025")

    # Custom engines
    tool = WebSearchTool(engines=["google", "duckduckgo"])

    # Specialized search tools
    academic_tool = create_academic_search_tool()
    news_tool = create_news_search_tool()
    tech_tool = create_technical_search_tool()
    shopping_tool = create_shopping_search_tool()

Available Engines (see https://docs.searxng.org/dev/engines/index.html):
- Web: google, bing, duckduckgo, startpage, yahoo, yandex
- Academic: google_scholar, arxiv, crossref, semantic_scholar
- News: google_news, bing_news, yahoo_news, reddit
- Technical: github, stackoverflow, gitlab
- Shopping: google_shopping, bing_shopping, amazon, ebay
- And many more specialized engines
"""

import asyncio
import json
import os
import re
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_community.utilities.searx_search import SearxSearchWrapper

from models import SearchResult, SearchResultContent

from ...monitoring.logging import composer_logger


def get_default_searx_config() -> Dict[str, Any]:
    """Get default configuration for SearxNG with sane defaults."""
    return {
        # Core search engines - prioritizing reliability and coverage
        "engines": [
            "google",  # Most comprehensive results
            "bing",  # Good alternative coverage
            "duckduckgo",  # Privacy-focused, good general results
            "startpage",  # Google results without tracking
            "github",  # For code and technical searches
            "arxiv",  # Academic papers
        ],
        # Search parameters
        "k": 10,  # Number of results to fetch
        "language": "en",  # English language results
        # Categories for general web search
        "categories": ["general"],
        # Additional parameters for better results
        "params": {
            "format": "json",
            "language": "en",
            "safesearch": 1,  # Moderate safe search
            "time_range": "",  # No time restriction by default
        },
        # Headers for better request handling
        "headers": {
            "User-Agent": "LLMMLLab-WebSearch/1.0",
        },
    }


class SearxNG:
    """Wrapper for Searx Search API with optimized configuration."""

    def __init__(
        self,
        searx_host: Optional[str] = None,
        engines: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.searx_host = searx_host or os.getenv("SEARX_HOST", "")

        # Merge default config with custom config
        default_config = get_default_searx_config()
        if config:
            # Deep merge configuration
            merged_config = {**default_config}
            merged_config.update(config)
            if "params" in config and "params" in default_config:
                merged_config["params"] = {
                    **default_config["params"],
                    **config["params"],
                }
        else:
            merged_config = default_config

        # Override engines if provided
        if engines:
            merged_config["engines"] = engines

        self.wrapper = SearxSearchWrapper(
            searx_host=self.searx_host,
            engines=merged_config["engines"],
            k=merged_config["k"],
            params=merged_config["params"],
            headers=merged_config["headers"],
            categories=merged_config["categories"],
        )
        self.logger = composer_logger.logger

        self.logger.debug(
            f"SearxNG initialized with engines: {merged_config['engines']}"
        )

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Execute search using Searx Search API."""
        results = []
        error = None
        try:
            if not query.strip():
                return SearchResult(
                    is_from_url_in_user_query=False,
                    query=query,
                    contents=results,
                    error="Empty query",
                )
            # Use the results() method for structured data instead of run()
            # This gives us proper metadata and structured results
            structured_results = self.wrapper.results(
                query=query,
                num_results=max_results,
                engines=None,  # Use configured engines
            )

            # Convert structured results to our format
            for i, result in enumerate(structured_results):
                if (
                    "Result" in result
                    and result["Result"] == "No good Search Result was found"
                ):
                    continue

                url = result.get("link", "No URL")
                if url.endswith("robots.txt"):
                    self.logger.debug(f"Skipping robots.txt URL: {url}")
                    continue

                title = result.get("title", "No title")
                content = result.get("snippet", "No content")

                results.append(
                    SearchResultContent(
                        url=url,
                        title=title,
                        content=content,
                        relevance=1.0 - (0.05 * i),
                    )
                )

            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=results,
                error=error,
            )

        except Exception as e:
            error = f"Error with Searx search: {e}"
            self.logger.error(error)

            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=results,
                error=error,
            )


class WebSearchTool(BaseTool):
    """Static tool for performing web searches using SearxNG provider.

    Configured with optimized engines (Google, Bing, DuckDuckGo, Startpage)
    for reliable web search results.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for information using a search query via SearxNG. "
        "Uses multiple search engines (Google, Bing, DuckDuckGo, Startpage) "
        "for comprehensive results. Returns formatted search results with titles, URLs, and content snippets."
    )

    def __init__(
        self,
        engines: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.engines = engines
        self.config = config

    async def _arun(self, query: str) -> str:
        """Async implementation of web search using SearxNG provider."""
        try:
            # Import search provider directly

            # Use SearxNG provider with configured engines and settings
            provider = SearxNG(
                engines=getattr(self, "engines", None),
                config=getattr(self, "config", None),
            )

            search_result = await provider.search(query, 5)

            if search_result and search_result.contents:
                formatted_results = [
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": (
                            content.content[:300] + "..."
                            if len(content.content) > 300
                            else content.content
                        ),
                        "relevance": content.relevance,
                    }
                    for content in search_result.contents
                ]

                return json.dumps(
                    {
                        "status": "success",
                        "results": formatted_results,
                        "query": query,
                        "count": len(formatted_results),
                    },
                    indent=2,
                )

            return json.dumps(
                {
                    "status": "success",
                    "results": [],
                    "query": query,
                    "message": "No search results found",
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "error": str(e), "query": query}, indent=2
            )

    def _run(self, query: str, **kwargs) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))


# Convenience factory functions for specialized search configurations


def create_academic_search_tool() -> WebSearchTool:
    """Create a WebSearchTool optimized for academic and research content."""
    return WebSearchTool(
        engines=[
            "google_scholar",  # Academic papers and citations
            "arxiv",  # Pre-print research papers
            "crossref",  # Academic publication metadata
            "google",  # General academic content
        ],
        config={
            "categories": ["science"],
            "params": {
                "format": "json",
                "language": "en",
                "safesearch": 0,  # Disable for academic content
            },
        },
    )


def create_news_search_tool() -> WebSearchTool:
    """Create a WebSearchTool optimized for news and current events."""
    return WebSearchTool(
        engines=[
            "google_news",  # Google News
            "bing_news",  # Bing News
            "yahoo_news",  # Yahoo News
            "reddit",  # Community discussions
        ],
        config={
            "categories": ["news"],
            "params": {
                "format": "json",
                "language": "en",
                "time_range": "month",  # Recent news within a month
                "safesearch": 1,
            },
        },
    )


def create_technical_search_tool() -> WebSearchTool:
    """Create a WebSearchTool optimized for technical documentation and code."""
    return WebSearchTool(
        engines=[
            "github",  # Code repositories and issues
            "stackoverflow",  # Programming Q&A
            "google",  # Technical documentation
            "duckduckgo",  # Alternative technical results
        ],
        config={
            "categories": ["it"],
            "params": {
                "format": "json",
                "language": "en",
                "safesearch": 0,  # Technical content may include code
            },
        },
    )


def create_shopping_search_tool() -> WebSearchTool:
    """Create a WebSearchTool optimized for product and shopping searches."""
    return WebSearchTool(
        engines=[
            "google_shopping",  # Google Shopping results
            "bing_shopping",  # Bing Shopping
            "amazon",  # Amazon products
            "ebay",  # eBay listings
        ],
        config={
            "categories": ["shopping"],
            "params": {
                "format": "json",
                "language": "en",
                "safesearch": 1,
            },
        },
    )
