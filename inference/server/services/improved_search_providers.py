"""
Enhanced search providers with better error handling and fallback mechanisms.
"""

import json
import os
import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from models import SearchResult, SearchResultContent
from models.web_search_providers import WebSearchProviders

logger = logging.getLogger(__name__)


class ImprovedSearchProvider(ABC):
    """Base class for search providers with enhanced error handling."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")
        self.is_available = False
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the search provider. Set self.is_available = True if successful."""
        pass

    @abstractmethod
    async def _execute_search(
        self, query: str, max_results: int
    ) -> List[SearchResultContent]:
        """Execute the actual search. Only called if provider is available."""
        pass

    async def search(self, query: str, max_results: int) -> SearchResult:
        """Execute search with comprehensive error handling."""
        if not self.is_available:
            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=[],
                error=f"{self.provider_name} is not available",
            )

        if not query.strip():
            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=[],
                error="Empty query",
            )

        try:
            self.logger.info(
                f"Executing search with {self.provider_name}: {query[:100]}..."
            )
            results = await self._execute_search(query, max_results)
            self.logger.info(f"{self.provider_name} returned {len(results)} results")

            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=results,
                error=None,
            )

        except Exception as e:
            error_msg = f"{self.provider_name} search failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=[],
                error=error_msg,
            )

    def _create_search_result(
        self, url: str, title: str, content: str, relevance: float = 0.5
    ) -> SearchResultContent:
        """Create a standardized search result."""
        return SearchResultContent(
            url=url.strip() if url else "",
            title=title.strip() if title else "No Title",
            content=content.strip() if content else "",
            relevance=relevance,
        )


class ImprovedDuckDuckGoProvider(ImprovedSearchProvider):
    """Enhanced DuckDuckGo search provider with better error handling."""

    def __init__(self):
        super().__init__("DuckDuckGo")

    def _initialize(self) -> None:
        """Initialize DuckDuckGo search wrapper."""
        try:
            from langchain_community.utilities.duckduckgo_search import (
                DuckDuckGoSearchAPIWrapper,
            )

            self.wrapper = DuckDuckGoSearchAPIWrapper()
            self.is_available = True
            self.logger.info("DuckDuckGo search provider initialized successfully")
        except ImportError as e:
            self.logger.error(
                f"DuckDuckGo initialization failed - missing dependency: {e}"
            )
        except Exception as e:
            self.logger.error(f"DuckDuckGo initialization failed: {e}")

    async def _execute_search(
        self, query: str, max_results: int
    ) -> List[SearchResultContent]:
        """Execute DuckDuckGo search."""
        results = []

        try:
            search_response = self.wrapper.run(query)

            # Handle different response formats
            if isinstance(search_response, str):
                try:
                    # Try to parse as JSON
                    entries = json.loads(search_response)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    if search_response.strip():
                        results.append(
                            self._create_search_result(
                                url="",
                                title="DuckDuckGo Result",
                                content=search_response[
                                    :500
                                ],  # Truncate long responses
                            )
                        )
                    return results

            elif isinstance(search_response, list):
                entries = search_response
            else:
                self.logger.warning(
                    f"Unexpected response type: {type(search_response)}"
                )
                return results

            # Process entries
            for entry in entries[:max_results]:
                if isinstance(entry, dict):
                    url = entry.get("link", entry.get("url", ""))
                    title = entry.get("title", "")
                    content = entry.get("snippet", entry.get("content", ""))

                    if url or title or content:  # At least one field should be present
                        results.append(
                            self._create_search_result(
                                url=url,
                                title=title,
                                content=content,
                            )
                        )

        except Exception as e:
            self.logger.error(f"Error processing DuckDuckGo response: {e}")
            raise

        return results


class ImprovedGoogleProvider(ImprovedSearchProvider):
    """Enhanced Google search provider with multiple fallback methods."""

    def __init__(self):
        super().__init__("Google")

    def _initialize(self) -> None:
        """Initialize Google search with fallback options."""
        # Try Google API client first
        try:
            from langchain_community.utilities.google_search import (
                GoogleSearchAPIWrapper,
            )

            api_key = os.getenv("GOOGLE_CSE_ID")
            search_engine_id = os.getenv("GOOGLE_API_KEY")

            if api_key and search_engine_id:
                self.wrapper = GoogleSearchAPIWrapper(
                    google_api_key=api_key,
                    google_cse_id=search_engine_id,
                )
                self.search_method = "api"
                self.is_available = True
                self.logger.info("Google API search provider initialized successfully")
                return

        except ImportError:
            self.logger.warning("google-api-python-client not installed")
        except Exception as e:
            self.logger.warning(f"Google API initialization failed: {e}")

        # Try Serper API as fallback
        try:
            from langchain_community.utilities.google_serper import (
                GoogleSerperAPIWrapper,
            )

            serper_key = os.getenv("SERPER_API_KEY")
            if serper_key:
                self.wrapper = GoogleSerperAPIWrapper(serper_api_key=serper_key)
                self.search_method = "serper"
                self.is_available = True
                self.logger.info(
                    "Google Serper search provider initialized successfully"
                )
                return

        except ImportError:
            self.logger.warning("Google Serper not available")
        except Exception as e:
            self.logger.warning(f"Google Serper initialization failed: {e}")

        self.logger.warning("No Google search methods available")

    async def _execute_search(
        self, query: str, max_results: int
    ) -> List[SearchResultContent]:
        """Execute Google search using available method."""
        results = []

        try:
            search_response = self.wrapper.run(query)

            # Handle different response formats based on method
            if self.search_method == "serper":
                # Serper returns a structured response
                if isinstance(search_response, str):
                    try:
                        data = json.loads(search_response)
                        organic_results = data.get("organic", [])

                        for result in organic_results[:max_results]:
                            results.append(
                                self._create_search_result(
                                    url=result.get("link", ""),
                                    title=result.get("title", ""),
                                    content=result.get("snippet", ""),
                                )
                            )
                    except json.JSONDecodeError:
                        # Fallback to text processing
                        pass

            else:
                # Standard Google API response
                if isinstance(search_response, str):
                    try:
                        entries = json.loads(search_response)
                        for entry in entries[:max_results]:
                            if isinstance(entry, dict):
                                results.append(
                                    self._create_search_result(
                                        url=entry.get("link", ""),
                                        title=entry.get("title", ""),
                                        content=entry.get("snippet", ""),
                                    )
                                )
                    except json.JSONDecodeError:
                        # Treat as plain text result
                        if search_response.strip():
                            results.append(
                                self._create_search_result(
                                    url="",
                                    title="Google Search Result",
                                    content=search_response[:500],
                                )
                            )

        except Exception as e:
            self.logger.error(f"Error processing Google response: {e}")
            raise

        return results


class RobustSearchProviderFactory:
    """Factory for creating robust search providers with proper fallbacks."""

    _providers_cache: Dict[WebSearchProviders, ImprovedSearchProvider] = {}

    @classmethod
    async def create_provider(
        cls, provider_type: WebSearchProviders, max_results: int = 10
    ) -> ImprovedSearchProvider:
        """Create a search provider with caching."""

        if provider_type in cls._providers_cache:
            cached_provider = cls._providers_cache[provider_type]
            if cached_provider.is_available:
                return cached_provider
            else:
                # Remove failed provider from cache
                del cls._providers_cache[provider_type]

        # Create new provider
        if provider_type == "ddg":
            provider = ImprovedDuckDuckGoProvider()
        elif provider_type == "google":
            provider = ImprovedGoogleProvider()
        else:
            # Default fallback
            provider = ImprovedDuckDuckGoProvider()

        # Cache if successful
        if provider.is_available:
            cls._providers_cache[provider_type] = provider

        return provider

    @classmethod
    def get_available_providers(cls) -> List[WebSearchProviders]:
        """Get list of available search providers."""
        available = []

        # Test each provider type
        test_providers = [
            "ddg",
            "google",
        ]

        for provider_type in test_providers:
            try:
                if provider_type == "ddg":
                    provider = ImprovedDuckDuckGoProvider()
                elif provider_type == "google":
                    provider = ImprovedGoogleProvider()
                else:
                    continue

                if provider.is_available:
                    available.append(provider_type)

            except Exception:
                continue

        return available
