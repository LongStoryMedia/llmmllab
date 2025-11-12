"""
Web search tool using SearxNG provider with LangGraph Command pattern.

This module provides a single, streamlined web search tool that integrates cleanly
with LangGraph workflows using strong typing and efficient state access.

Features:
- Single function-based tool using @tool decorator
- Strong typing with WorkflowState instead of generic Dict
- Efficient user_config access from injected state (no database calls)
- Command pattern for proper state updates
- SearxNG provider with comprehensive engine support

Configuration:
- Default engines: Google, Bing, DuckDuckGo, Startpage for comprehensive coverage
- Structured results API for reliable parsing
- User-specific preferences from WorkflowState.user_config.web_search
- Configurable engines, categories, and search parameters

Usage in LangGraph workflows:
    # Tool is automatically available when registered in tool registry
    # LangGraph handles injection of tool_call_id and WorkflowState

Available Engines (see https://docs.searxng.org/dev/engines/index.html):
- Web: google, bing, duckduckgo, startpage, yahoo, yandex
- Academic: google_scholar, arxiv, crossref, semantic_scholar
- News: google_news, bing_news, yahoo_news, reddit
- Technical: github, stackoverflow, gitlab
- Shopping: google_shopping, bing_shopping, amazon, ebay
- And many more specialized engines
"""

from calendar import c
import json
import os
from typing import Annotated, List, Literal, Optional

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger
from models import SearchResult, SearchResultContent, WebSearchConfig

# Global cache to track recent searches and prevent duplicates
_search_cache = {}
_duplicate_counts = {}  # Track how many times each query was requested
_max_cache_size = 100
_max_duplicate_attempts = 3  # Hard stop after 3 duplicate attempts

# Import from langchain_community (preferred) then fallback to langchain_classic
try:  # pragma: no cover - import resolution
    from langchain_community.utilities.searx_search import SearxSearchWrapper  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment variability
    try:
        from langchain_classic.utilities.searx_search import SearxSearchWrapper  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Neither langchain_community nor langchain_classic SearxSearchWrapper available. Install langchain-community >=0.2.0."
        ) from e


class SearxNG:
    """Wrapper for Searx Search API using WebSearchConfig."""

    def __init__(
        self,
        web_config: WebSearchConfig,
        categories: List[
            Literal[
                "general",
                "news",
                "science",
                "it",
                "shopping",
                "images",
                "videos",
                "music",
                "files",
                "social",
            ]
        ],
    ):
        self.web_config = web_config
        self.searx_host = web_config.searx_host or os.getenv("SEARX_HOST", "")
        self.categories = categories or list[str](web_config.categories)

        # Build SearxSearchWrapper parameters directly from WebSearchConfig
        params = {
            "format": "json",
            "language": web_config.language,
            "safesearch": web_config.safesearch,
            "time_range": web_config.time_range or "",
        }

        headers = {
            "User-Agent": web_config.user_agent or "LLMMLLab-WebSearch/1.0",
        }

        self.wrapper = SearxSearchWrapper(
            searx_host=self.searx_host,
            engines=web_config.engines,
            k=web_config.max_results,
            params=params,
            headers=headers,
            categories=self.categories,  # type: ignore
        )
        self.logger = llmmllogger.logger

        self.logger.debug(f"SearxNG initialized with engines: {web_config.engines}")

    async def search(
        self,
        query: str,
        max_results: int,
        categories: List[
            Literal[
                "general",
                "news",
                "science",
                "it",
                "shopping",
                "images",
                "videos",
                "music",
                "files",
                "social",
            ]
        ] = [],
    ) -> SearchResult:
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
                categories=categories,  # type: ignore
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


# Single web search tool using simplified signature for testing
@tool
async def web_search(
    query: Annotated[str, "The search query to execute"],
    num_results: Annotated[Optional[int], "Number of search results to return"] = None,
    categories: Annotated[
        List[
            Literal[
                "general",
                "news",
                "science",
                "it",
                "shopping",
                "images",
                "videos",
                "music",
                "files",
                "social",
            ]
        ],
        "Search categories to include",
    ] = [],
) -> str:
    """
    Search the web for information and automatically add results to workflow state.

    This tool performs comprehensive web searches using multiple search engines
    and returns structured results. Use this tool when you need current information
    from the internet about any topic.

    Args:
        query: The search query to execute
        num_results: Number of search results to return (overrides user config if provided)

    Returns:
        Search results with titles, URLs, content snippets, and relevance scores
    """
    from models.default_configs import (  # pylint: disable=import-outside-toplevel
        DEFAULT_WEB_SEARCH_CONFIG,
    )

    logger = llmmllogger.logger.bind(component="WebSearch")

    # ANTI-RECURSION: Check for duplicate searches to prevent infinite loops
    global _search_cache
    query_normalized = query.strip().lower()

    if query_normalized in _search_cache:
        # Track duplicate attempts
        _duplicate_counts[query_normalized] = (
            _duplicate_counts.get(query_normalized, 0) + 1
        )
        duplicate_count = _duplicate_counts[query_normalized]

        logger.warning(
            f"🔄 BLOCKED duplicate web search for: '{query}' (attempt #{duplicate_count}) - returning cached results"
        )

        # Hard stop after too many duplicate attempts - force agent to use what it has
        if duplicate_count >= _max_duplicate_attempts:
            return (
                "🛑 **SEARCH LIMIT REACHED** 🛑\n\n"
                f"The query '{query}' has been searched multiple times already. "
                "Please use the information from previous searches to provide your response. "
                "No further searches for this query will be performed.\n\n"
                "**Final Answer Required:** Based on the search results already provided, "
                "please synthesize and present your findings to the user."
            )

        cached_result = _search_cache[query_normalized]

        # Add explicit duplicate notice to help agent understand it should stop
        duplicate_notice = (
            f"⚠️ **DUPLICATE SEARCH DETECTED (#{duplicate_count})** ⚠️\n\n"
            f"This query '{query}' has already been searched in this conversation. "
            "Using previous results to avoid redundant searches.\n\n"
            "**Previous Search Results:**\n\n"
        )
        return duplicate_notice + cached_result

    # Clean cache if it gets too large
    if len(_search_cache) > _max_cache_size:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_search_cache.keys())[: -(_max_cache_size // 2)]
        for key in keys_to_remove:
            del _search_cache[key]
        logger.debug(f"Cleaned search cache, removed {len(keys_to_remove)} old entries")

    try:
        # For testing without ToolRuntime - use default config
        # TODO: Implement proper LangGraph agent context to support ToolRuntime
        web_config = DEFAULT_WEB_SEARCH_CONFIG
        logger.debug(
            "Using default web search config - ToolRuntime temporarily removed for testing"
        )
        if not num_results:
            num_results = DEFAULT_WEB_SEARCH_CONFIG.max_results

        # Use SearxNG provider with WebSearchConfig
        provider = SearxNG(web_config=web_config, categories=categories)
        search_result = await provider.search(query, num_results)
        if search_result and search_result.contents:
            # Format results for display with more substantial content
            formatted_results = []
            for content in search_result.contents:
                # Provide much more content (up to 1500 characters) instead of just 300
                # This gives the AI enough context to work with while still being manageable
                content_text = content.content
                if len(content_text) > 1500:
                    # Find a good breaking point (sentence end) near 1500 chars
                    truncate_pos = 1500
                    sentence_ends = [". ", "! ", "? ", ". "]
                    for end in sentence_ends:
                        last_sentence = content_text.rfind(end, 1200, 1500)
                        if last_sentence != -1:
                            truncate_pos = last_sentence + len(end)
                            break
                    content_text = content_text[:truncate_pos].rstrip() + "..."

                formatted_results.append(
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": content_text,
                        "relevance": content.relevance,
                    }
                )

            # Create response message with improved formatting
            response_message = f"🔍 **Web Search Results for: '{query}'**\n\n"
            response_message += f"Found {len(formatted_results)} relevant results:\n\n"

            for i, result in enumerate(formatted_results, 1):
                response_message += f"**Result {i}: {result['title']}**\n"
                response_message += f"📍 URL: {result['url']}\n"
                response_message += f"📄 Content: {result['content']}\n"
                response_message += f"⭐ Relevance: {result['relevance']:.2f}\n"
                response_message += "---\n\n"

            # Add helpful note about getting full content if needed
            response_message += "💡 **Note**: If you need the complete content from any of these articles, "
            response_message += "use the `read_web_content` tool with the specific URL."

            logger.info(
                f"Web search completed successfully with {len(formatted_results)} results",
                query=query,
                result_count=len(formatted_results),
            )

            # Cache the successful result and initialize duplicate count
            _search_cache[query_normalized] = response_message
            _duplicate_counts[query_normalized] = 0

            # Return string result - ToolNode will automatically create ToolMessage
            return response_message

        else:
            # No results found
            if search_result and search_result.error:
                response_message = f"⚠️ Web search error: {search_result.error}"
            else:
                response_message = f"🔍 No results found for query: '{query}'"

            logger.warning(f"Web search returned no results", query=query)

            # Cache the no-results response to prevent repeated failed searches
            _search_cache[query_normalized] = response_message
            _duplicate_counts[query_normalized] = 0

            return response_message

    except Exception as e:
        error_message = f"❌ Web search failed: {str(e)}"
        logger.error(f"Web search error: {e}", query=query, error=str(e))

        # Cache the error response to prevent repeated failed searches
        _search_cache[query_normalized] = error_message
        _duplicate_counts[query_normalized] = 0

        return error_message
