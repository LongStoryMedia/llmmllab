"""
Static web search tool using DuckDuckGo provider.

This tool performs web searches with consistent behavior using
DuckDuckGo as the search provider (no API key required).
"""

import asyncio
import json

from langchain_core.tools import BaseTool

from models.web_search_providers import WebSearchProviders


class WebSearchTool(BaseTool):
    """Static tool for performing web searches using DuckDuckGo provider."""
    name: str = "web_search"
    description: str = "Search the web for information using a search query. Returns formatted search results."

    async def _arun(self, query: str) -> str:
        """Async implementation of web search using DuckDuckGo provider."""
        try:
            # Import search provider directly
            from server.services.search_providers import SearchProviderFactory
            
            # Use DuckDuckGo as default provider (no API key required)
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.DDG, 
                max_results=3
            )
            
            search_result = await provider.search(query, 3)
            
            if search_result and search_result.contents:
                formatted_results = [
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": content.content[:200] + "..." if len(content.content) > 200 else content.content,
                        "relevance": content.relevance
                    }
                    for content in search_result.contents
                ]
                
                return json.dumps({
                    "status": "success",
                    "results": formatted_results,
                    "query": query
                }, indent=2)
            else:
                return json.dumps({
                    "status": "success",
                    "results": [],
                    "query": query,
                    "message": "No search results found"
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "status": "error", 
                "error": str(e),
                "query": query
            }, indent=2)
    
    def _run(self, query: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))