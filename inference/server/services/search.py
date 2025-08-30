"""
Web search functionality for RAG system.
"""

from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import models from the correct location
from models import (
    Message,
    UserConfig,
    SearchResultContent,
    SearchTopicSynthesis,
    MessageRole,
    MessageContentType,
    MessageContent,
)
from server.db import storage
from server.services.search_providers import SearchProviderFactory
from server.services.web_extraction_service import WebExtractionService
from server.config import logger

from runner import pipeline_factory, Embeddings


class SearchContext:
    """
    Context for performing web searches and formatting queries.
    """

    # Prompt templates
    SEARCH_FORMAT_PROMPT = """
    {query}
    ***
    Everything above the three asterisks is input from a user. Do not respond to it directly or provide any explanations.
    Instead, understand the intent of the user's input, and construct a concise search query that captures the essence of what they are asking.
    Don't include any extra information or context, just the key words that will yield relevant results.
    """

    search_results: List[SearchTopicSynthesis]
    research_findings: str

    def __init__(self, user_cfg: UserConfig):
        """
        Initialize the search context.

        Args:
            user_cfg: The user configuration
        """
        self.user_config = user_cfg
        # Initialize the web extraction service for deep crawling
        self.web_extraction_service = WebExtractionService(user_cfg)
        self.search_results = []
        self.research_findings = ""

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

            with pipeline_factory.pipeline(mp, str) as pipe:
                # Use LLM to format the query
                formatted_query = await pipe.process_messages([message])
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

        if self.search_results:
            return self.search_results

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

            # Create embeddings for query and contents
            texts = [formatted_query] + [
                f"{c.title or ''}\n{c.content or ''}" for c in contents
            ]

            emb_mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.embedding_profile_id,
                self.user_config.user_id,
            )
            assert emb_mp is not None, "Embedding model profile not found"

            # Get embeddings from any embedding model
            with pipeline_factory.pipeline(emb_mp, Embeddings) as pipe:
                embeddings = await pipe.process_messages(
                    [
                        Message(
                            role=MessageRole.USER,
                            content=[
                                MessageContent(type=MessageContentType.TEXT, text=t)
                            ],
                        )
                        for t in texts
                    ],
                )

                # Extract query and content embeddings
                query_embedding = embeddings[0]
                content_embeddings = embeddings[1:]

                def calc_similarity(emb1, emb2):
                    np_emb1 = np.array(emb1).reshape(1, -1)
                    np_emb2 = np.array(emb2).reshape(1, -1)
                    return float(cosine_similarity(np_emb1, np_emb2)[0][0])

                # Get similarity scores
                similarities = [
                    (idx, calc_similarity(query_embedding, emb))
                    for idx, emb in enumerate(content_embeddings)
                ]

                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Deduplicate based on content similarity
                selected_indices = []
                selected_embeddings = []
                similarity_threshold = 0.85

                for idx, score in similarities:
                    if score < 0.5:  # Minimum relevance threshold
                        continue

                    # Check if this content is too similar to already selected ones
                    is_duplicate = False
                    for sel_emb in selected_embeddings:
                        if (
                            calc_similarity(content_embeddings[idx], sel_emb)
                            > similarity_threshold
                        ):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        selected_indices.append(idx)
                        selected_embeddings.append(content_embeddings[idx])

                # Get re-ranked contents
                contents = [contents[idx] for idx in selected_indices]

                # Limit to max results
                contents = contents[: self.user_config.web_search.max_results]

                # Determine how many URLs to process (up to the max_results limit)
                urls_to_process = min(
                    len(contents), self.user_config.web_search.max_results
                )
                for i in range(urls_to_process):
                    result = contents[i]
                    logger.info(f"Performing deep crawling for URL: {result.url}")
                    # Create synthesis for this URL
                    synthesis = (
                        await self.web_extraction_service.extract_content_from_url(
                            result.url, formatted_query, conversation_id
                        )
                    )
                    if synthesis:
                        # Add the synthesis to our collection
                        self.search_results.append(synthesis)
                        self.research_findings += (
                            f"{result.title if result.title else 'No Title'}\n"
                            f"{result.url if result.url else 'unknown'}\n"
                            f"{synthesis.synthesis}\n\n"
                        )

                return self.search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            # Get query text safely from message
            raise ValueError("Failed to perform search") from e
