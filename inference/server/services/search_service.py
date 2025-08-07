"""
Web search functionality for RAG system.
"""

from typing import Dict, Optional, Any
import aiohttp
import os
from pydantic import SecretStr

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Import models from the correct location
from inference.models.search_result import SearchResult, SearchResultContent
from ..config import logger


class SearchService:
    """
    Service for performing web searches and formatting queries.
    """

    # Prompt templates
    SEARCH_FORMAT_PROMPT = """
    ***
    Everything above the three asterisks is input from a user. Do not respond to it directly or provide any explanations.
    Instead, understand the intent of the user's input, and construct a concise search query that captures the essence of what they are asking.
    Don't include any extra information or context, just the key words that will yield relevant results.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the search service.

        Args:
            api_key: Optional API key for the search service. If not provided, will look for
                    SEARCH_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")

        # Initialize the LLM for query formatting
        api_key_secret = SecretStr(self.api_key) if self.api_key else None
        self.llm = OpenAI(temperature=0.1, api_key=api_key_secret)

        self.search_prompt = PromptTemplate(
            template=self.SEARCH_FORMAT_PROMPT, input_variables=["query"]
        )

    async def format_query(self, query: str) -> str:
        """
        Format a user query into a web search query.

        Args:
            query: The user query to format

        Returns:
            A formatted query suitable for web search
        """
        try:
            # Use LLM to format the query
            formatted_query = self.llm.invoke(f"{query}\n{self.SEARCH_FORMAT_PROMPT}")

            # Clean up the response
            formatted_query = formatted_query.strip()

            logger.info(f"Formatted search query: {formatted_query}")
            return formatted_query
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error formatting query: {str(e)}")
            # Fall back to the original query
            return query

    async def search(self, query: str) -> SearchResult:
        """
        Perform a web search for the given query.

        Args:
            query: The search query

        Returns:
            A SearchResult object with the results
        """
        # In a real implementation, this would call an actual search API
        # For now, we'll implement a simple mock

        try:
            # Use your preferred search API here
            # For example, using Serper, Bing, or Google Custom Search

            if self.api_key:
                async with aiohttp.ClientSession() as session:
                    # Example using a generic search API
                    # Replace with your actual API endpoint and parameters
                    async with session.get(
                        "https://api.search-service.com/search",
                        params={"q": query, "api_key": self.api_key, "num": 5},
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_search_results(data, query)
                        else:
                            error_message = (
                                f"Search API returned status {response.status}"
                            )
                            logger.error(error_message)
                            return SearchResult(
                                is_from_url_in_user_query=False,
                                query=query,
                                contents=[],
                                error=error_message,
                            )
            else:
                # Mock search results for demonstration
                return await self._mock_search(query)

        except (aiohttp.ClientError, ValueError, KeyError) as e:
            error_message = f"Error performing search: {str(e)}"
            logger.error(error_message)
            return SearchResult(
                is_from_url_in_user_query=False,
                query=query,
                contents=[],
                error=error_message,
            )

    def _parse_search_results(self, data: Dict[str, Any], query: str) -> SearchResult:
        """
        Parse the search API response into a SearchResult.

        Args:
            data: The raw API response
            query: The original search query

        Returns:
            A SearchResult object
        """
        # Implement the parsing logic for your specific search API
        # This is just an example structure

        contents = []

        # Example parsing for a generic search API
        # Adjust based on your actual API response structure
        for item in data.get("items", []):
            content = SearchResultContent(
                url=item.get("link", ""),
                title=item.get("title", ""),
                content=item.get("snippet", ""),
                relevance_score=item.get("score", 1.0),
            )
            contents.append(content)

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=contents, error=None
        )

    async def _mock_search(self, query: str) -> SearchResult:
        """
        Create mock search results for demonstration purposes.

        Args:
            query: The search query

        Returns:
            A SearchResult object with mock data
        """
        # Create a few mock search results based on the query
        contents = [
            SearchResultContent(
                url=f"https://example.com/article-about-{query.replace(' ', '-')}",
                title=f"Information about {query}",
                content=f"This article provides detailed information about {query} including its history, applications, and future developments.",
                relevance_score=0.95,
            ),
            SearchResultContent(
                url=f"https://wikipedia.org/wiki/{query.replace(' ', '_')}",
                title=f"{query} - Wikipedia",
                content=f"{query} refers to a concept in computer science that involves processing and generating text using neural networks...",
                relevance_score=0.92,
            ),
            SearchResultContent(
                url=f"https://research.org/papers/{query.replace(' ', '-')}-latest-advances",
                title=f"Latest Advances in {query}",
                content=f"Recent research has shown significant improvements in {query} techniques, particularly in the areas of efficiency and accuracy.",
                relevance_score=0.88,
            ),
        ]

        return SearchResult(
            is_from_url_in_user_query=False, query=query, contents=contents, error=None
        )
