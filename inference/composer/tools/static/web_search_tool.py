"""
Static web search tool using SearxNG provider.

This tool performs web searches with consistent behavior using
SearxNG as the search provider (running in the same cluster).
"""

import asyncio
import json
import os
import re
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_community.utilities.searx_search import SearxSearchWrapper

from models import SearchResult, SearchResultContent

from ...monitoring.logging import composer_logger


class SearxNG:
    """Wrapper for Searx Search API."""

    def __init__(self, searx_host: Optional[str] = None, **kwargs):
        self.searx_host = searx_host or os.getenv("SEARX_HOST", "")
        self.wrapper = SearxSearchWrapper(searx_host=self.searx_host, **kwargs)
        self.logger = composer_logger.logger

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
            # Searx returns a string by default
            raw_results = self.wrapper.run(query)

            # Parse the string results

            # Split by numbered sections (1., 2., etc.)
            sections = re.split(r"\n\d+\.\s", raw_results)
            if sections and sections[0].startswith("1. "):
                sections[0] = sections[0][3:]  # Remove "1. " from first section

            for i, section in enumerate(sections[:max_results]):
                if not section.strip():
                    continue

                # Try to extract URL
                url_match = re.search(r"URL:\s+(https?://\S+)", section)
                url = url_match.group(1) if url_match else "No URL"

                if url.endswith("robots.txt"):
                    self.logger.debug(f"Skipping robots.txt URL: {url}")
                    continue

                # Use first line as title
                lines = section.strip().split("\n")
                title = lines[0] if lines else "No title"

                # Rest is description
                description = (
                    "\n".join(lines[1:]) if len(lines) > 1 else "No description"
                )
                if url_match:
                    # Remove URL line from description
                    description = re.sub(
                        r"URL:\s+https?://\S+\n?", "", description
                    ).strip()

                results.append(
                    SearchResultContent(
                        url=url,
                        title=title,
                        content=description,
                        relevance=1.0 - (0.05 * i),
                    )
                )
        except Exception as e:
            error = f"Error with Searx search: {e}"
            self.logger.error(error)

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=results, error=error
        )


class WebSearchTool(BaseTool):
    """Static tool for performing web searches using SearxNG provider."""

    name: str = "web_search"
    description: str = (
        "Search the web for information using a search query via SearxNG. Returns formatted search results."
    )

    async def _arun(self, query: str) -> str:
        """Async implementation of web search using SearxNG provider."""
        try:
            # Import search provider directly

            # Use SearxNG provider (running in the same cluster) - no query formatting needed
            provider = SearxNG()

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
