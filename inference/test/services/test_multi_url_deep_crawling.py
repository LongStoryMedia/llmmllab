"""
Unit tests for multi-URL deep crawling in search service.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from models.web_search_config import WebSearchConfig
from models.user_config import UserConfig
from models.web_search_providers import WebSearchProviders
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.search_result import SearchResult
from models.search_result_content import SearchResultContent
from models.search_topic_synthesis import SearchTopicSynthesis
from server.context.search import SearchContext


class TestMultiURLDeepCrawling:
    """Test cases for multi-URL deep crawling."""

    @pytest.fixture
    def user_config(self):
        """Create a test user configuration."""
        # Create minimal config with the fields we need for testing
        return MagicMock(
            web_search=WebSearchConfig(
                enabled=True,
                auto_detect=True,
                max_results=3,
                include_results=True,
                search_providers=[WebSearchProviders.DDG],
                max_urls_deep=2,
            ),
            user_id="test-user",
        )

    @pytest.fixture
    def mock_provider(self):
        """Create a mock search provider."""
        mock = MagicMock()
        mock.search = AsyncMock()

        # Create search results with multiple URLs
        contents = [
            SearchResultContent(
                url="https://example.com/1", title="Test 1", content="Content 1"
            ),
            SearchResultContent(
                url="https://example.com/2", title="Test 2", content="Content 2"
            ),
            SearchResultContent(
                url="https://example.com/3", title="Test 3", content="Content 3"
            ),
        ]

        result = SearchResult(
            is_from_url_in_user_query=False, query="test query", contents=contents
        )

        mock.search.return_value = result
        return mock

    @pytest.fixture
    def mock_web_extraction(self):
        """Create a mock web extraction service."""
        mock = MagicMock()
        mock.extract_content_from_url = AsyncMock()

        # Set up synthesis return values for different URLs
        async def mock_extract(url, query, conversation_id):
            # Create different synthesis objects based on the URL
            from datetime import datetime

            synthesis = SearchTopicSynthesis(
                urls=[url, f"{url}/sub1", f"{url}/sub2"],
                topics=["topic1", "topic2"],
                synthesis=f"Synthesis for {url}",
                created_at=datetime.now(),
                conversation_id=conversation_id,
            )
            return synthesis

        mock.extract_content_from_url.side_effect = mock_extract
        return mock

    @patch("server.services.search_service._get_search_provider")
    async def test_multi_url_deep_crawling(
        self, mock_get_provider, user_config, mock_provider, mock_web_extraction
    ):
        """Test that deep crawling is performed for multiple URLs."""
        # Set up mocks
        mock_get_provider.return_value = mock_provider

        # Create the search service
        service = SearchContext(user_config)
        # Replace the web extraction service with our mock
        service.web_extraction_service = mock_web_extraction

        # Create a message for testing
        message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="test query")],
        )

        # Mock format query to return the original query
        service._format_query = AsyncMock(return_value="test query")

        # Call the search method
        result = await service.search(message, 123)

        # Verify that extract_content_from_url was called for each URL
        assert mock_web_extraction.extract_content_from_url.call_count == 3

        # Verify that the result contains synthesized content
        assert result.contents is not None
        # Should have original 3 results plus 3 synthesized results
        assert len(result.contents) == 6

        # Verify the calls to extract_content_from_url
        calls = mock_web_extraction.extract_content_from_url.call_args_list
        assert len(calls) == 3
        assert calls[0][0][0] == "https://example.com/1"  # First URL
        assert calls[1][0][0] == "https://example.com/2"  # Second URL
        assert calls[2][0][0] == "https://example.com/3"  # Third URL
