"""
Static web search tool using SearxNG provider.

This tool performs web searches with consistent behavior using
SearxNG as the search provider (running in the same cluster).
"""

import asyncio
import json

from langchain_core.tools import BaseTool

from models.web_search_providers import WebSearchProviders


class WebSearchTool(BaseTool):
    """Static tool for performing web searches using SearxNG provider."""
    name: str = "web_search"
    description: str = "Search the web for information using a search query via SearxNG. Returns formatted search results."

    async def _arun(self, query: str) -> str:
        """Async implementation of web search using SearxNG provider."""
        try:
            # Import search provider directly
            from server.services.search_providers import SearchProviderFactory
            
            # Use SearxNG provider (running in the same cluster) - no query formatting needed
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.SEARX, 
                max_results=5
            )
            
            search_result = await provider.search(query, 5)
            
            if search_result and search_result.contents:
                formatted_results = [
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": content.content[:300] + "..." if len(content.content) > 300 else content.content,
                        "relevance": content.relevance
                    }
                    for content in search_result.contents
                ]
                
                return json.dumps({
                    "status": "success",
                    "results": formatted_results,
                    "query": query,
                    "count": len(formatted_results)
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
    
    def _run(self, query: str, **kwargs) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query))