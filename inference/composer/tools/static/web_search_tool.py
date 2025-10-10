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
    # Create tool using user_id - configuration retrieved from data layer
    tool = create_web_search_tool(user_id="user_123")
    result = await tool._arun("machine learning trends 2025")

    # Direct instantiation
    tool = WebSearchTool(user_id="user_123")
    result = await tool._arun("search query")

    # Specialized search tools using user configuration
    academic_tool = create_academic_search_tool(user_id="user_123")
    news_tool = create_news_search_tool(user_id="user_123")
    tech_tool = create_technical_search_tool(user_id="user_123")
    shopping_tool = create_shopping_search_tool(user_id="user_123")

User Configuration Integration:
- Configuration retrieved from shared data layer via storage.user_config.get_user_config(user_id)
- User-specific web search preferences merged with system defaults at data layer
- Ensures user preferences are always respected for engines, categories, limits, etc.
- Specialized search behavior should be configured through user preferences rather than factory overrides

Available Engines (see https://docs.searxng.org/dev/engines/index.html and https://github.com/searxng/searxng/tree/master/searx/engines):
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
from typing import Optional, List, Dict, Any, Annotated

from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

# Attempt import from langchain_classic (newer split) then fallback to langchain_community
try:  # pragma: no cover - import resolution
    from langchain_classic.utilities.searx_search import SearxSearchWrapper  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment variability
    try:
        from langchain_community.utilities.searx_search import SearxSearchWrapper  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Neither langchain_classic nor langchain_community SearxSearchWrapper available. Install langchain-community >=0.2.0."
        ) from e


from models import SearchResult, SearchResultContent, WebSearchConfig

from composer.monitoring.logging import composer_logger


class SearxNG:
    """Wrapper for Searx Search API using WebSearchConfig."""

    def __init__(self, web_config: WebSearchConfig):
        self.web_config = web_config
        self.searx_host = web_config.searx_host or os.getenv("SEARX_HOST", "")

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
            categories=web_config.categories,
        )
        self.logger = composer_logger.logger

        self.logger.debug(f"SearxNG initialized with engines: {web_config.engines}")

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

    Retrieves user-specific WebSearchConfig from shared data layer with defaults
    merged automatically. Uses actual user_id to get proper configuration.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for information using a search query via SearxNG. "
        "Configurable search engines, categories, and parameters for optimal results. "
        "Returns formatted search results with titles, URLs, and content snippets."
    )
    
    # Declare user_id as a proper Pydantic field
    user_id: str

    def __init__(self, user_id: str):
        super().__init__(user_id=user_id)
        # Create logger without assigning to self (Pydantic doesn't allow it)
        # Use it directly when needed
    
    @property
    def logger(self):
        """Get logger for this tool instance."""
        return composer_logger.logger.bind(component="WebSearchTool")

    async def _get_web_search_config(self) -> WebSearchConfig:
        """Get web search configuration from user config via shared data layer."""
        from db import storage  # pylint: disable=import-outside-toplevel
        from models.default_configs import (  # pylint: disable=import-outside-toplevel
            DEFAULT_WEB_SEARCH_CONFIG,
        )

        try:

            # Get complete user config with defaults merged at data layer
            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(self.user_id)
            if not user_config:
                return DEFAULT_WEB_SEARCH_CONFIG
            return user_config.web_search
        except Exception as e:
            self.logger.error(
                "Failed to retrieve user web search config",
                user_id=self.user_id,
                error=str(e),
            )

            return DEFAULT_WEB_SEARCH_CONFIG

    async def _arun(
        self, 
        query: str,
        **kwargs: Any
    ) -> Command:
        """Async implementation of web search using SearxNG provider.
        
        Returns a Command that updates the WorkflowState with search results.
        """
        # Get tool_call_id from kwargs or use a default
        tool_call_id = kwargs.get('tool_call_id', 'web_search_call')
        
        try:
            # Get web search configuration from user config
            web_config = await self._get_web_search_config()

            # Use SearxNG provider with WebSearchConfig
            provider = SearxNG(web_config=web_config)
            search_result = await provider.search(query, web_config.max_results)

            if search_result and search_result.contents:
                # Format results for display
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

                # Create response message for the conversation
                response_message = json.dumps(
                    {
                        "status": "success",
                        "results": formatted_results,
                        "query": query,
                        "count": len(formatted_results),
                    },
                    indent=2,
                )

                # Return Command that updates state with search results
                return Command(update={
                    "web_search_results": [search_result],
                    "search_query": query,
                    "messages": [
                        ToolMessage(response_message, tool_call_id=tool_call_id)
                    ]
                })

            # No results case
            response_message = json.dumps(
                {
                    "status": "success",
                    "results": [],
                    "query": query,
                    "message": "No search results found",
                },
                indent=2,
            )

            return Command(update={
                "search_query": query,
                "messages": [
                    ToolMessage(response_message, tool_call_id=tool_call_id)
                ]
            })

        except Exception as e:
            error_message = json.dumps(
                {"status": "error", "error": str(e), "query": query}, indent=2
            )
            
            return Command(update={
                "search_query": query,
                "messages": [
                    ToolMessage(error_message, tool_call_id=tool_call_id)
                ]
            })

    def _run(self, query: str, **kwargs) -> Command:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query, **kwargs))


# Factory functions for creating WebSearchTool instances


def create_web_search_tool(
    user_id: str,
) -> WebSearchTool:
    """Create a WebSearchTool that uses user configuration from data layer.

    Args:
        user_id: User ID for configuration retrieval

    Returns:
        Configured WebSearchTool instance
    """
    return WebSearchTool(user_id=user_id)


# Convenience factory functions for specialized search configurations


def create_academic_search_tool(user_id: str) -> WebSearchTool:
    """Create a WebSearchTool that uses user configuration from data layer.

    Note: Academic search behavior should be configured through user preferences
    in the user_config.web_search settings rather than factory function overrides.
    This ensures user preferences are always respected.

    Args:
        user_id: User ID for configuration retrieval

    Returns:
        WebSearchTool using user's web search configuration
    """
    return WebSearchTool(user_id=user_id)


def create_news_search_tool(user_id: str) -> WebSearchTool:
    """Create a WebSearchTool that uses user configuration from data layer.

    Note: News search behavior should be configured through user preferences.
    """
    return WebSearchTool(user_id=user_id)


def create_technical_search_tool(user_id: str) -> WebSearchTool:
    """Create a WebSearchTool that uses user configuration from data layer.

    Note: Technical search behavior should be configured through user preferences.
    """
    return WebSearchTool(user_id=user_id)


def create_shopping_search_tool(user_id: str) -> WebSearchTool:
    """Create a WebSearchTool that uses user configuration from data layer.

    Note: Shopping search behavior should be configured through user preferences.
    """
    return WebSearchTool(user_id=user_id)


# Function-based tool using Command pattern (preferred approach)
@tool
async def web_search_with_state_update(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict[str, Any], InjectedState]
) -> Command:
    """
    Search the web for information and automatically add results to workflow state.
    
    This tool performs web searches using SearxNG and returns a Command that updates
    the WorkflowState with search results, enabling automatic routing to search 
    synthesis nodes.
    
    Uses the official LangGraph Command pattern for state updates as documented in:
    https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#short-term-memory
    
    Args:
        query: The search query to execute
        tool_call_id: Injected tool call ID for message tracking (auto-injected by LangGraph)
        state: Injected workflow state for accessing user context (auto-injected by LangGraph)
        
    Returns:
        Command object that updates state with search results
    """
    from db import storage  # pylint: disable=import-outside-toplevel
    from models.default_configs import (  # pylint: disable=import-outside-toplevel
        DEFAULT_WEB_SEARCH_CONFIG,
    )
    
    logger = composer_logger.logger.bind(component="WebSearchWithStateUpdate")
    
    try:
        # Get user_id from injected state
        user_id = state.get("user_id", "default_user")
        
        # Get web search configuration from user config
        try:
            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)
            web_config = user_config.web_search if user_config else DEFAULT_WEB_SEARCH_CONFIG
        except Exception as e:
            logger.warning(f"Failed to get user config, using defaults: {e}")
            web_config = DEFAULT_WEB_SEARCH_CONFIG

        # Use SearxNG provider with WebSearchConfig
        provider = SearxNG(web_config=web_config)
        search_result = await provider.search(query, web_config.max_results)

        if search_result and search_result.contents:
            # Format results for display
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

            # Create response message for the conversation
            response_message = json.dumps(
                {
                    "status": "success",
                    "results": formatted_results,
                    "query": query,
                    "count": len(formatted_results),
                },
                indent=2,
            )

            logger.info(
                f"Web search completed successfully with {len(formatted_results)} results",
                query=query[:100],
                result_count=len(formatted_results)
            )

            # Return Command that updates state with search results
            return Command(update={
                "web_search_results": [search_result],
                "search_query": query,
                "messages": [
                    ToolMessage(response_message, tool_call_id=tool_call_id)
                ]
            })

        # No results case
        response_message = json.dumps(
            {
                "status": "success",
                "results": [],
                "query": query,
                "message": "No search results found",
            },
            indent=2,
        )

        logger.info("Web search completed with no results", query=query[:100])

        return Command(update={
            "search_query": query,
            "messages": [
                ToolMessage(response_message, tool_call_id=tool_call_id)
            ]
        })

    except Exception as e:
        error_message = json.dumps(
            {"status": "error", "error": str(e), "query": query}, indent=2
        )
        
        logger.error(f"Web search failed: {e}", query=query[:100])
        
        return Command(update={
            "search_query": query,
            "messages": [
                ToolMessage(error_message, tool_call_id=tool_call_id)
            ]
        })
