"""
Web Searclass WebSearchAgent:web content retrieval and processing.
Provides core business logic for web search operations and content extraction.
"""

from typing import List, Optional, Dict, Any
import asyncio

from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class WebSearchAge        """
        Extract content from specific URLs.
        
        Args:
            urls: List of URLs to extract content from
            user_id: User identifier
            query: Search query for context
            conversation_id: Conversation ID for context
            timeout: Extraction timeout per URL
            
        Returns:
            List of extracted content results
        """
    Web Search Agent for web content retrieval with configurable search depth.
    
    Provides core business logic for web search, content extraction, and result processing.
    Supports both shallow (quick) and deep (comprehensive) search strategies.
    """

    def __init__(self):
        """Initialize web search agent."""
        self.logger = composer_logger.logger.bind(component="WebSearchAgent")

    async def perform_search(
        self,
        query: str,
        user_id: str,
        search_depth: str = "SHALLOW",
        max_results: int = 5,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Perform web search with configurable depth and result processing.
        
        Args:
            query: Search query string
            user_id: User identifier for configuration
            search_depth: "SHALLOW" for quick search, "DEEP" for comprehensive
            max_results: Maximum number of results to return
            timeout: Timeout in seconds for search operations
            
        Returns:
            Search results with extracted content and metadata
        """
        try:
            self.logger.info(
                "Performing web search",
                user_id=user_id,
                query=query[:100],  # Truncate for logging
                search_depth=search_depth,
                max_results=max_results
            )

            # Get user configuration for search preferences
            try:
                from db import storage  # pylint: disable=import-outside-toplevel
                user_config = await storage.get_service(storage.user_config).get_user_config(user_id)
                
                # Use user's search preferences if available
                if hasattr(user_config, 'search'):
                    max_results = min(max_results, user_config.search.max_results or max_results)
                    timeout = user_config.search.timeout or timeout
                    
            except Exception as e:
                self.logger.warning(f"Could not get user search config: {e}")

            # Perform search based on depth
            if search_depth.upper() == "DEEP":
                results = await self._perform_deep_search(query, max_results, timeout)
            else:
                results = await self._perform_shallow_search(query, max_results, timeout)

            # Process and format results
            processed_results = await self._process_search_results(results, search_depth)

            self.logger.info(
                "Web search completed",
                user_id=user_id,
                results_count=len(processed_results.get("results", [])),
                search_depth=search_depth
            )

            return processed_results

        except Exception as e:
            self.logger.error(
                "Web search failed",
                user_id=user_id,
                query=query[:100],
                error=str(e)
            )
            raise NodeExecutionError(f"Web search failed: {e}") from e

    async def _perform_shallow_search(
        self, 
        query: str, 
        max_results: int, 
        timeout: int
    ) -> List[Dict[str, Any]]:
        """
        Perform shallow/quick web search for immediate results.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            timeout: Search timeout
            
        Returns:
            List of search result dictionaries
        """
        try:
            # Use web search tool for proper search functionality
            from composer.tools.static.web_search_tool import SearxNG  # pylint: disable=import-outside-toplevel
            from models import WebSearchConfig  # pylint: disable=import-outside-toplevel
            
            # Get default web search config
            web_config = WebSearchConfig()
            search_tool = SearxNG(web_config)
            
            # Perform quick search
            search_result = await asyncio.wait_for(
                search_tool.search(query=query, max_results=max_results),
                timeout=timeout
            )
            
            # Convert SearchResult to list format expected by caller
            if search_result and search_result.contents:
                return [
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": content.content,
                        "snippet": content.content[:200] + "..." if len(content.content) > 200 else content.content
                    }
                    for content in search_result.contents
                ]
            return []

        except asyncio.TimeoutError:
            self.logger.warning(f"Shallow search timed out after {timeout}s")
            return []
        except Exception as e:
            self.logger.error(f"Shallow search failed: {e}")
            return []

    async def _perform_deep_search(
        self, 
        query: str, 
        max_results: int, 
        timeout: int
    ) -> List[Dict[str, Any]]:
        """
        Perform deep/comprehensive web search with detailed content extraction.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            timeout: Search timeout
            
        Returns:
            List of detailed search result dictionaries
        """
        try:
            # Use web search tool for comprehensive search
            from composer.tools.static.web_search_tool import SearxNG  # pylint: disable=import-outside-toplevel
            from models import WebSearchConfig  # pylint: disable=import-outside-toplevel
            
            # Get enhanced web search config for deep search
            web_config = WebSearchConfig(
                max_results=max_results * 2,  # Get more initial results for filtering
                engines=["google", "bing", "duckduckgo"],  # Multiple engines for comprehensive search
                safesearch=1,
                time_range=""
            )
            search_tool = SearxNG(web_config)
            
            # Perform comprehensive search
            search_result = await asyncio.wait_for(
                search_tool.search(query=query, max_results=max_results * 2),
                timeout=timeout * 2  # Allow more time for deep search
            )
            
            # Convert SearchResult to list format and filter for quality
            if search_result and search_result.contents:
                results = [
                    {
                        "title": content.title,
                        "url": content.url,
                        "content": content.content,
                        "snippet": content.content[:300] + "..." if len(content.content) > 300 else content.content,
                        "metadata": {
                            "source": "deep_search",
                            "search_engines": web_config.engines,
                            "content_length": len(content.content)
                        }
                    }
                    for content in search_result.contents
                ]
                
                # Filter and rank results for quality
                filtered_results = self._filter_deep_search_results(results, max_results)
                return filtered_results
            return []

        except asyncio.TimeoutError:
            self.logger.warning(f"Deep search timed out after {timeout * 2}s")
            return []
        except Exception as e:
            self.logger.error(f"Deep search failed: {e}")
            return []

    def _filter_deep_search_results(
        self, 
        results: List[Dict[str, Any]], 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank deep search results for quality and relevance.
        
        Args:
            results: Raw search results
            max_results: Maximum results to return
            
        Returns:
            Filtered and ranked results
        """
        # Score results based on content quality metrics
        scored_results = []
        
        for result in results:
            score = 0
            
            # Score based on content length (prefer substantial content)
            content_length = len(result.get("content", ""))
            if content_length > 1000:
                score += 3
            elif content_length > 500:
                score += 2
            elif content_length > 100:
                score += 1
            
            # Score based on metadata availability
            if result.get("title"):
                score += 1
            if result.get("description"):
                score += 1
            if result.get("keywords"):
                score += 1
                
            # Prefer results with structured content
            if result.get("structured_data"):
                score += 2
                
            scored_results.append((score, result))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results[:max_results]]

    async def _process_search_results(
        self, 
        results: List[Dict[str, Any]], 
        search_depth: str
    ) -> Dict[str, Any]:
        """
        Process and format search results for consumption by other nodes.
        
        Args:
            results: Raw search results
            search_depth: Search depth used
            
        Returns:
            Processed search results with metadata
        """
        processed_results = {
            "results": results,
            "search_depth": search_depth,
            "result_count": len(results),
            "summary": f"Found {len(results)} results using {search_depth.lower()} search",
            "urls": [result.get("url", "") for result in results if result.get("url")],
            "total_content_length": sum(len(result.get("content", "")) for result in results)
        }
        
        # Generate content synthesis for deep searches
        if search_depth.upper() == "DEEP" and results:
            processed_results["content_synthesis"] = await self._synthesize_content(results)
        
        return processed_results

    async def _synthesize_content(self, results: List[Dict[str, Any]]) -> str:
        """
        Synthesize content from multiple search results into coherent summary.
        
        Args:
            results: Search results with extracted content
            
        Returns:
            Synthesized content summary
        """
        try:
            # Extract key content from each result
            content_pieces = []
            for result in results:
                title = result.get("title", "")
                content = result.get("content", "")[:500]  # Truncate for synthesis
                url = result.get("url", "")
                
                if content:
                    piece = f"From {title or url}:\n{content}"
                    content_pieces.append(piece)
            
            if not content_pieces:
                return "No content available for synthesis."
            
            # Simple synthesis - in a full implementation, this would use an LLM
            synthesis = f"Synthesized from {len(content_pieces)} sources:\n\n"
            synthesis += "\n\n---\n\n".join(content_pieces[:3])  # Limit to top 3
            
            if len(content_pieces) > 3:
                synthesis += f"\n\n... and {len(content_pieces) - 3} more sources"
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Content synthesis failed: {e}")
            return "Content synthesis unavailable."

    async def extract_urls_content(
        self, 
        urls: List[str], 
        user_id: str,
        query: str = "",
        conversation_id: int = 0,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Extract content from specific URLs.
        
        Args:
            urls: List of URLs to extract content from
            user_id: User identifier
            timeout: Extraction timeout per URL
            
        Returns:
            List of extracted content results
        """
        try:
            self.logger.info(
                "Extracting content from URLs",
                user_id=user_id,
                url_count=len(urls)
            )

            # Import extraction service
            from server.services.web_extraction_service import WebExtractionService  # pylint: disable=import-outside-toplevel
            
            search_service = WebExtractionService()
            
            # Extract content from each URL
            extraction_results = []
            for url in urls:
                try:
                    result = await asyncio.wait_for(
                        search_service.extract_content_from_url(url),
                        timeout=timeout
                    )
                    if result:
                        extraction_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to extract from {url}: {e}")
                    continue

            self.logger.info(
                "URL content extraction completed",
                user_id=user_id,
                successful_extractions=len(extraction_results)
            )

            return extraction_results

        except Exception as e:
            self.logger.error(
                "URL content extraction failed",
                user_id=user_id,
                error=str(e)
            )
            raise NodeExecutionError(f"URL content extraction failed: {e}") from e