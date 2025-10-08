"""
Single Source Content Agent for URL-based content retrieval and synthesis.
"""

import asyncio
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from models import WebSearchConfig

from composer.monitoring.logging import composer_logger


class SingleSourceAgent:
    """
    Agent for retrieving and synthesizing content from a single URL.

    Provides focused content extraction and synthesis from individual URLs,
    designed to be orchestrated in parallel for comprehensive web search.
    """

    def __init__(self):
        """Initialize single source content agent."""
        self.logger = composer_logger.logger.bind(component="SingleSourceAgent")

    async def retrieve_and_synthesize(
        self,
        url: str,
        query: str,
        user_id: str,
        search_config: Optional["WebSearchConfig"] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve content from a single URL and synthesize it relative to the query.

        Args:
            url: URL to retrieve content from
            query: Search query for context-aware synthesis
            user_id: User identifier for configuration
            search_config: WebSearchConfig for retrieval parameters

        Returns:
            Content retrieval and synthesis results
        """
        try:
            self.logger.info(
                "Retrieving content from single source",
                user_id=user_id,
                url=url,
                query=query[:100] if query else "No query",
            )

            # Validate URL
            if not self._is_valid_url(url):
                return self._create_error_result(url, "Invalid URL format", query)

            # Get content extraction configuration
            timeout = (
                int(search_config.timeout)
                if search_config and search_config.timeout
                else 30
            )

            # Extract content from URL
            content_data = await self._extract_url_content(url, timeout)
            if not content_data or content_data.get("error"):
                error_msg = (
                    content_data.get("error", "Failed to extract content")
                    if content_data
                    else "No content retrieved"
                )
                return self._create_error_result(url, error_msg, query)

            # Synthesize content relative to query
            synthesis = await self._synthesize_content(
                content_data, query, user_id, search_config
            )

            # Create successful result
            result = {
                "url": url,
                "title": content_data.get("title", ""),
                "content": content_data.get("content", ""),
                "synthesis": synthesis,
                "word_count": len(content_data.get("content", "").split()),
                "extraction_time": content_data.get("extraction_time", 0),
                "relevance_score": await self._calculate_relevance(content_data, query),
                "status": "success",
                "error": None,
            }

            self.logger.info(
                "Single source retrieval completed",
                user_id=user_id,
                url=url,
                word_count=result["word_count"],
                relevance=result["relevance_score"],
            )

            return result

        except asyncio.TimeoutError:
            self.logger.warning(f"Content retrieval timed out for URL: {url}")
            return self._create_error_result(url, "Content retrieval timeout", query)
        except Exception as e:
            self.logger.error(f"Single source retrieval failed for {url}: {e}")
            return self._create_error_result(url, f"Retrieval error: {str(e)}", query)

    async def _extract_url_content(
        self, url: str, timeout: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract content from URL using appropriate extraction method.

        Args:
            url: URL to extract content from
            timeout: Extraction timeout in seconds

        Returns:
            Extracted content data or None if failed
        """
        try:
            # Use SearxNG deep URL extraction if available
            from composer.tools.static.web_search_tool import (
                SearxNG,
            )  # pylint: disable=import-outside-toplevel
            from models.web_search_config import (
                WebSearchConfig,
            )  # pylint: disable=import-outside-toplevel

            # Configure for single URL extraction
            extraction_config = WebSearchConfig(
                max_results=1,
                timeout=float(timeout),
                max_urls_deep=1,
                engines=["google"],  # Minimal engine for URL extraction
            )

            searx = SearxNG(extraction_config)

            # Perform URL-specific search to get content
            # This is a simplified approach - in practice you'd want direct URL content extraction
            domain = urlparse(url).netloc
            site_query = f"site:{domain} {url.split('/')[-1]}"

            search_result = await asyncio.wait_for(
                searx.search(query=site_query, max_results=1), timeout=timeout
            )

            if search_result and search_result.contents:
                content_item = search_result.contents[0]
                return {
                    "title": content_item.title,
                    "content": content_item.content,
                    "extraction_time": timeout,  # Placeholder
                    "url": content_item.url,
                    "error": None,
                }
            else:
                return {"error": "No content extracted from URL"}

        except asyncio.TimeoutError:
            return {"error": "Content extraction timeout"}
        except Exception as e:
            self.logger.error(f"Content extraction failed for {url}: {e}")
            return {"error": f"Extraction failed: {str(e)}"}

    async def _synthesize_content(
        self,
        content_data: Dict[str, Any],
        query: str,
        user_id: str,  # Currently unused in simplified implementation
        search_config: Optional[
            "WebSearchConfig"
        ] = None,  # Currently unused in simplified implementation
    ) -> str:
        """
        Synthesize extracted content relative to the search query.

        Args:
            content_data: Extracted content data
            query: Search query for context
            user_id: User identifier (reserved for future use)
            search_config: WebSearchConfig for synthesis parameters (reserved for future use)

        Returns:
            Synthesized content summary
        """
        try:
            # Simple content synthesis without external pipeline dependencies
            content = content_data.get("content", "")
            title = content_data.get("title", "")

            if not content:
                return "No content available for synthesis"

            # Create basic synthesis
            try:
                # Basic synthesis approach - can be enhanced with LLM pipeline later
                if len(content) > 500:
                    synthesized = f"Content analysis for query '{query}': {title}. Summary: {content[:300]}... [Content continues with {len(content) - 300} more characters]"
                else:
                    synthesized = (
                        f"Content analysis for query '{query}': {title}. {content}"
                    )
                return synthesized
            except Exception as synthesis_error:
                self.logger.warning(
                    f"Content synthesis failed, using raw content: {synthesis_error}"
                )
                return f"Raw content from {content_data.get('url', 'source')}: {content[:400]}..."

        except Exception as e:
            self.logger.error(f"Content synthesis failed: {e}")
            return f"Content from {content_data.get('url', 'source')}: {content_data.get('content', '')[:200]}..."

    async def _calculate_relevance(
        self, content_data: Dict[str, Any], query: str
    ) -> float:
        """
        Calculate relevance score between content and query.

        Args:
            content_data: Extracted content data
            query: Search query

        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            if not query:
                return 0.5  # Neutral relevance for no query

            content = content_data.get("content", "").lower()
            title = content_data.get("title", "").lower()
            query_lower = query.lower()

            # Simple keyword-based relevance scoring
            query_terms = query_lower.split()

            title_matches = sum(1 for term in query_terms if term in title)
            content_matches = sum(1 for term in query_terms if term in content)

            # Weight title matches more heavily
            total_matches = (title_matches * 2) + content_matches
            max_possible = len(query_terms) * 3  # 2 for title + 1 for content

            relevance = (
                min(total_matches / max_possible, 1.0) if max_possible > 0 else 0.0
            )

            return round(relevance, 2)

        except Exception:
            return 0.5  # Default neutral relevance

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if URL appears valid
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _create_error_result(
        self, url: str, error_message: str, query: str
    ) -> Dict[str, Any]:
        """
        Create standardized error result.

        Args:
            url: URL that failed
            error_message: Error description
            query: Original search query

        Returns:
            Standardized error result dictionary
        """
        return {
            "url": url,
            "title": "",
            "content": "",
            "synthesis": f"Failed to retrieve content: {error_message}",
            "word_count": 0,
            "extraction_time": 0,
            "relevance_score": 0.0,
            "status": "error",
            "error": error_message,
            "query": query,
        }
