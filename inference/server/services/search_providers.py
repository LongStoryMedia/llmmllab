"""
Standardized interface for web search providers.
This module provides a unified interface for all search providers,
making it easier to add new providers and aggregate results.
"""

import json
import os
import re
from typing import Optional, cast
from abc import ABC, abstractmethod

from langchain_community.utilities.brave_search import BraveSearchWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

# Prefer the new package if available; fall back to community package
try:
    # New recommended location
    from langchain_google_community import GoogleSearchAPIWrapper  # type: ignore
except Exception:  # noqa: BLE001
    from langchain_community.utilities.google_search import (  # type: ignore
        GoogleSearchAPIWrapper,
    )
from langchain_community.utilities.searx_search import SearxSearchWrapper

from models import SearchResult, SearchResultContent
from models.web_search_providers import WebSearchProviders
from server.config import logger


class StandardSearchProvider(ABC):
    """
    Abstract base class for standardized search providers.
    All search providers should inherit from this class to ensure a consistent interface.
    """

    @abstractmethod
    async def search(self, query: str, max_results: int) -> SearchResult:
        """
        Execute a search query and return standardized results.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            SearchResult object containing the search results
        """

    @staticmethod
    def _create_search_result(
        url: str, title: str, content: str, relevance: float = 1.0
    ) -> SearchResultContent:
        """
        Create a standardized search result object.

        Args:
            url: URL of the search result
            title: Title of the search result
            content: Content/description of the search result
            relevance: Relevance score (0.0 to 1.0)

        Returns:
            A SearchResultContent object
        """
        return SearchResultContent(
            url=url, title=title, content=content, relevance=relevance
        )


class BraveSearchProviderWrapper(StandardSearchProvider):
    """Wrapper for Brave Search API."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY", "")
        # Pass kwargs directly - BraveSearchWrapper will handle API key
        self.wrapper = BraveSearchWrapper(**kwargs)

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Execute search using Brave Search API."""
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
            # Brave search wrapper returns a formatted string
            search_response = self.wrapper.run(query)

            # Parse the string results - Brave returns stringified json
            # that is an array of objects which have "title", "link", and "snippet"
            entries = cast(list[dict[str, str]], json.loads(search_response))
            for e in entries[:max_results]:
                title = e.get("title")
                url = e.get("link")
                content = e.get("snippet")

                if not url:
                    continue

                results.append(
                    self._create_search_result(
                        url=url, title=title or "", content=content or ""
                    )
                )
        except Exception as e:
            error = f"Error with Brave search: {e}"
            logger.error(error)
            # Log additional context for debugging
            if "422" in str(e):
                logger.error(f"Brave search HTTP 422 error - likely rate limited or invalid query: {query}")
            elif "401" in str(e):
                logger.error("Brave search authentication failed - check API key")
            elif "403" in str(e):
                logger.error("Brave search forbidden - check API permissions")

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=results, error=error
        )


class DuckDuckGoSearchProviderWrapper(StandardSearchProvider):
    """Wrapper for DuckDuckGo Search API."""

    def __init__(self, **kwargs):
        # Lazy init to avoid ImportError (ddgs) crashing the flow
        self._kwargs = kwargs
        self.wrapper: Optional[DuckDuckGoSearchAPIWrapper] = None  # type: ignore[type-arg]
        self._init_error: Optional[str] = None

    def _parse_ddg_text_response(self, response: str) -> list[dict[str, str]]:
        """Parse DuckDuckGo text response into structured format."""
        entries = []
        
        # DuckDuckGo returns text like: "Title - Description URL"
        # Split by lines and parse each result
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract title, snippet, and URL from the text
            # Format is usually: "Title - Description"
            if ' - ' in line:
                parts = line.split(' - ', 1)
                title = parts[0].strip()
                snippet = parts[1].strip() if len(parts) > 1 else ""
                
                # Generate a search URL since DDG doesn't provide direct links
                url = f"https://duckduckgo.com/?q={title.replace(' ', '+')}"
                
                entries.append({
                    "title": title,
                    "snippet": snippet,
                    "link": url
                })
            else:
                # Fallback: treat the whole line as title
                entries.append({
                    "title": line,
                    "snippet": "",
                    "link": f"https://duckduckgo.com/?q={line.replace(' ', '+')}"
                })
        
        return entries

    def _ensure_wrapper(self) -> None:
        if self.wrapper is not None or self._init_error is not None:
            return
        try:
            self.wrapper = DuckDuckGoSearchAPIWrapper(**self._kwargs)
        except Exception as e:  # noqa: BLE001
            self._init_error = str(e)
            logger.error(f"DuckDuckGo initialization failed: {self._init_error}")

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Execute search using DuckDuckGo Search API."""
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

            self._ensure_wrapper()
            if not self.wrapper:
                reason = self._init_error or "Unknown initialization error"
                return SearchResult(
                    is_from_url_in_user_query=False,
                    query=query,
                    contents=results,
                    error=(
                        "DuckDuckGo search unavailable: "
                        + reason
                        + ". Install ddgs: pip install -U ddgs."
                    ),
                )

            search_response = self.wrapper.run(query)

            # DuckDuckGo returns text, not JSON - parse it directly
            try:
                # DuckDuckGo wrapper returns formatted text, not JSON
                # Parse the text response directly
                entries = self._parse_ddg_text_response(search_response)
            except Exception as parse_err:
                logger.error(
                    f"DuckDuckGo text parsing failed: {parse_err}. Response: {search_response[:200]}..."
                )
                raise ValueError(f"DuckDuckGo text parsing failed: {parse_err}")

            for e in entries[:max_results]:
                title = e.get("title")
                url = e.get("link")
                content = e.get("snippet")
                if not url:
                    continue
                results.append(
                    self._create_search_result(
                        url=url, title=title or "", content=content or ""
                    )
                )
        except Exception as e:  # noqa: BLE001
            error = f"Error with DuckDuckGo search: {e}"
            logger.error(error)
            # Log additional context for debugging
            if "JSONDecodeError" in str(e) or "malformed JSON" in str(e):
                logger.error(f"DuckDuckGo returned non-JSON response for query: {query}")
            elif "DNS" in str(e) or "Name or service not known" in str(e):
                logger.error("DuckDuckGo DNS resolution failed - network connectivity issue")

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=results, error=error
        )


class SearxSearchProviderWrapper(StandardSearchProvider):
    """Wrapper for Searx Search API."""

    def __init__(self, searx_host: Optional[str] = None, **kwargs):
        self.searx_host = searx_host or os.getenv("SEARX_HOST", "")
        self.wrapper = SearxSearchWrapper(searx_host=self.searx_host, **kwargs)

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
                    self._create_search_result(
                        url=url,
                        title=title,
                        content=description,
                        relevance=1.0 - (0.05 * i),
                    )
                )
        except Exception as e:
            error = f"Error with Searx search: {e}"
            logger.error(error)

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=results, error=error
        )


class GoogleSerperSearchProviderWrapper(StandardSearchProvider):
    """Wrapper for Google Serper Search API."""

    def __init__(self, serper_api_key: Optional[str] = None, **kwargs):
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY", "")
        self.wrapper = GoogleSerperAPIWrapper(
            serper_api_key=self.serper_api_key, **kwargs
        )

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Execute search using Google Serper API."""
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
            raw_results = self.wrapper.results(query, k=max_results)

            # Handle Serper's result format
            if isinstance(raw_results, dict) and "organic" in raw_results:
                organic_results = raw_results["organic"][:max_results]

                for i, result in enumerate(organic_results):
                    results.append(
                        self._create_search_result(
                            url=result.get("link", "No link"),
                            title=result.get("title", "No title"),
                            content=result.get("snippet", "No description"),
                            relevance=1.0 - (0.05 * i),
                        )
                    )
            else:
                warning_msg = f"Unexpected Serper result format: {type(raw_results)}"
                logger.warning(warning_msg)
                error = warning_msg
        except Exception as e:
            error = f"Error with Google Serper search: {e}"
            logger.error(error)

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=results, error=error
        )


class GoogleSearchProviderWrapper(StandardSearchProvider):
    """Wrapper for Google Search API."""

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        **kwargs,
    ):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self.google_cse_id = google_cse_id or os.getenv("GOOGLE_SEARCH_CX", "")
        # Defer expensive/fragile initialization until first use to avoid crashing the whole search flow
        self._kwargs = kwargs
        self.wrapper: Optional[GoogleSearchAPIWrapper] = None  # type: ignore[type-arg]
        self._init_error: Optional[str] = None

    def _ensure_wrapper(self) -> None:
        """Lazily initialize the GoogleSearchAPIWrapper, capturing missing deps as a soft error."""
        if self.wrapper is not None or self._init_error is not None:
            return
        try:
            self.wrapper = GoogleSearchAPIWrapper(  # type: ignore[call-arg]
                google_api_key=self.google_api_key,
                google_cse_id=self.google_cse_id,
                **self._kwargs,
            )
        except Exception as e:  # Missing deps etc.
            self._init_error = str(e)
            logger.error(f"GoogleSearch initialization failed: {self._init_error}")

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Execute search using Google Search API."""
        results = []
        error = None
        try:
            # Ensure the wrapper is ready; if not, return a graceful error without raising
            self._ensure_wrapper()
            if not self.wrapper:
                reason = self._init_error or "Unknown initialization error"
                error = (
                    "Google search unavailable: "
                    + reason
                    + ". Install google-api-python-client>=2.100.0 and set GOOGLE_SEARCH_API_KEY/GOOGLE_SEARCH_CX to enable."
                )
                return SearchResult(
                    is_from_url_in_user_query=False,
                    query=query,
                    contents=results,
                    error=error,
                )

            entries = cast(
                list[dict[str, str]],
                self.wrapper.results(query, num_results=max_results),
            )

            for e in entries[:max_results]:
                title = e.get("title")
                url = e.get("link")
                content = e.get("snippet")

                if not url:
                    continue

                results.append(
                    self._create_search_result(
                        url=url, title=title or "", content=content or ""
                    )
                )
        except Exception as e:
            error = f"Error with Google search: {e}"
            logger.error(error)

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=results, error=error
        )


class SearchProviderFactory:
    """Factory for creating search provider instances."""

    @staticmethod
    def create_provider(
        provider_type: WebSearchProviders, max_results: int
    ) -> StandardSearchProvider:
        """
        Create a search provider instance based on the provider type.

        Args:
            provider_type: The type of search provider to create
            max_results: Maximum number of results for the provider

        Returns:
            A StandardSearchProvider instance
        """
        if provider_type == WebSearchProviders.BRAVE:
            return BraveSearchProviderWrapper(search_kwargs={"count": max_results})
        elif provider_type == WebSearchProviders.DDG:
            return DuckDuckGoSearchProviderWrapper(max_results=max_results)
        elif provider_type == WebSearchProviders.SEARX:
            return SearxSearchProviderWrapper(k=max_results)
        elif provider_type == WebSearchProviders.SERPER:
            return GoogleSerperSearchProviderWrapper(k=max_results)
        elif provider_type == WebSearchProviders.GOOGLE:
            return GoogleSearchProviderWrapper(k=max_results)

        raise ValueError(f"Unknown search provider type: {provider_type}")
