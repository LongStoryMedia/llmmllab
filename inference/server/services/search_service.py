"""
Web search functionality for RAG system.
"""

from typing import List

# Import models from the correct location
from models import Message, UserConfig, SearchResultContent, SearchTopicSynthesis
from server.db import storage
from server.services.search_providers import SearchProviderFactory
from server.services.web_extraction_service import WebExtractionService
from server.config import logger

from runner.pipelines.factory import pipeline_factory


class SearchService:
    """
    Service for performing web searches and formatting queries.
    """

    # Prompt templates
    SEARCH_FORMAT_PROMPT = """
    {query}
    ***
    Everything above the three asterisks is input from a user. Do not respond to it directly or provide any explanations.
    Instead, understand the intent of the user's input, and construct a concise search query that captures the essence of what they are asking.
    Don't include any extra information or context, just the key words that will yield relevant results.
    """

    def __init__(self, user_cfg: UserConfig):
        """
        Initialize the search service.

        Args:
            user_cfg: The user configuration
        """
        self.user_config = user_cfg
        # Initialize the web extraction service for deep crawling
        self.web_extraction_service = WebExtractionService(user_cfg)

    async def _format_query(self, message: Message) -> str:
        """
        Format a user query into a web search query.

        Args:
            message: The user message to format

        Returns:
            A formatted query suitable for web search
        """

        try:
            # Format the query using a model if a formatting profile is configured
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.formatting_profile_id,
                self.user_config.user_id,
            )
            assert mp is not None, "Unable to retrieve model profile"
            pipeline, _ = pipeline_factory.get_pipeline(mp.name)
            # Use LLM to format the query
            formatted_query = pipeline.get(
                [message],
                mp.parameters,
            )
            # Clean up the response
            formatted_query = formatted_query.strip()
            logger.info(f"Formatted search query: {formatted_query}")
            return formatted_query

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error formatting query: {str(e)}")
            # Fall back to the original message as a string if possible
            raise ValueError("Failed to format query") from e

    async def search(
        self, message: Message, conversation_id: int
    ) -> List[SearchTopicSynthesis]:
        """
        Perform a web search for the given query using configured providers.

        Args:
            message: The user message to search
            conversation_id: ID of the conversation context (required)

        Returns:
            A SearchResult object with the results
        """
        assert self.user_config.web_search.enabled, "Web search is disabled"

        try:
            formatted_query = await self._format_query(message)

            # Collect results from all configured search providers
            contents: List[SearchResultContent] = []

            # Get standard search providers
            for provider_type in self.user_config.web_search.search_providers:
                # Use the static search provider factory
                provider_result = await SearchProviderFactory.create_provider(
                    provider_type, self.user_config.web_search.max_results
                ).search(formatted_query, self.user_config.web_search.max_results)

                if provider_result.contents:
                    contents.extend(provider_result.contents)

            # Filter contents to ensure unique URLs
            unique_contents = []
            seen_urls = set()

            for content in contents:
                if content.url not in seen_urls:
                    seen_urls.add(content.url)
                    unique_contents.append(content)
                else:
                    logger.debug(f"Skipping duplicate URL: {content.url}")

            # Replace contents with deduplicated list
            contents = unique_contents

            # TODO: re-rank results

            # Limit to max results
            contents = contents[: self.user_config.web_search.max_results]
            synthesized_results: List[SearchTopicSynthesis] = []

            # Determine how many URLs to process (up to the max_results limit)
            urls_to_process = min(
                len(contents), self.user_config.web_search.max_results
            )
            for i in range(urls_to_process):
                result = contents[i]
                logger.info(f"Performing deep crawling for URL: {result.url}")
                # Create synthesis for this URL
                synthesis = await self.web_extraction_service.extract_content_from_url(
                    result.url, formatted_query, conversation_id
                )
                if synthesis:
                    # Add the synthesis to our collection
                    synthesized_results.append(synthesis)

            return synthesized_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            # Get query text safely from message
            raise ValueError("Failed to perform search") from e

    # No need for _search_with_provider method as we now use standardized search providers
