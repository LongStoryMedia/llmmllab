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

import json
import os
from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger
from models import SearchResult, SearchResultContent, WebSearchConfig

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
        self.logger = llmmllogger.logger

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


# Single web search tool using Command pattern with strong typing
@tool
async def web_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[WorkflowState, InjectedState],
) -> Command:
    """
    Search the web for information and automatically add results to workflow state.

    This tool performs comprehensive web searches using multiple search engines
    and returns structured results. Use this tool when you need current information
    from the internet about any topic.

    Args:
        query: The search query to execute

    Returns:
        Search results with titles, URLs, content snippets, and relevance scores
    """
    from models.default_configs import (  # pylint: disable=import-outside-toplevel
        DEFAULT_WEB_SEARCH_CONFIG,
    )

    logger = llmmllogger.logger.bind(component="WebSearch")

    try:
        # Get user_config directly from injected state (much more efficient!)
        if state.user_config and hasattr(state.user_config, "web_search"):
            web_config = state.user_config.web_search
            logger.debug("Using web search config from injected state")
        else:
            web_config = DEFAULT_WEB_SEARCH_CONFIG
            logger.debug("Using default web search config - no user_config in state")

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
                result_count=len(formatted_results),
            )

            # Return Command that updates state with search results
            return Command(
                update={
                    "web_search_results": [search_result],
                    "search_query": query,
                    "messages": [
                        ToolMessage(response_message, tool_call_id=tool_call_id)
                    ],
                }
            )

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

        return Command(
            update={
                "search_query": query,
                "messages": [ToolMessage(response_message, tool_call_id=tool_call_id)],
            }
        )

    except Exception as e:
        error_message = json.dumps(
            {"status": "error", "error": str(e), "query": query}, indent=2
        )

        logger.error(f"Web search failed: {e}", query=query[:100])

        return Command(
            update={
                "search_query": query,
                "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)],
            }
        )
